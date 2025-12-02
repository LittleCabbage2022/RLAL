# src/agents/sac_agent.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
import copy

from ..utils.tensor_utils import to_tensor, to_numpy
from ..utils.buffer import ReplayBuffer
from .networks import GaussianPolicy, QNetwork

class SACAgent:
    def __init__(self, state_dim, z_dim, action_dim, args: dict, device):
        """
        state_dim, z_dim, action_dim: ints
        args: dict with keys actor_lr, critic_lr, replay_size, batch_sac, gamma, tau, alpha_init, target_entropy
        device: torch.device or 'cuda'/'cpu'
        """
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        # --- 参数提取 ---
        actor_lr = float(args.get('actor_lr', 1e-4))    # 论文: 1e-4
        critic_lr = float(args.get('critic_lr', 1e-3))  # 论文: 1e-3
        alpha_fixed = float(args.get('alpha', 0.1))     # 论文: 0.1
        self.gamma = float(args.get('gamma', 0.99))
        self.tau = float(args.get('tau', 0.005))

        # 【新增】一致性约束系数，建议在 yaml 里加一个 consistency_beta: 1.0 或 0.5
        # 如果 yaml 没写，默认给 1.0，保证生效
        self.consistency_beta = float(args.get('consistency_beta', 1.0))

        self.state_dim = state_dim
        self.z_dim = z_dim
        self.action_dim = action_dim

        # --- 策略网络 & 优化器 ---
        # 论文要求 Adam, beta=(0.9, 0.999) (默认值), lr=1e-4
        self.policy = GaussianPolicy(state_dim, z_dim, action_dim, hidden1=300, hidden2=300).to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=actor_lr)

        # 【新增】Target Actor (目标策略网络)
        # 它的作用是提供一个“过去策略的平滑版本”作为参考锚点
        self.policy_target = copy.deepcopy(self.policy)
        # 冻结 Target Actor 的参数，不需要梯度
        for p in self.policy_target.parameters():
            p.requires_grad = False

        # --- Q网络 & 优化器 ---
        # 论文要求 lr=1e-3
        self.q1 = QNetwork(state_dim, z_dim, action_dim, hidden1=300, hidden2=300).to(self.device)
        self.q2 = QNetwork(state_dim, z_dim, action_dim, hidden1=300, hidden2=300).to(self.device)
        self.q1_target = QNetwork(state_dim, z_dim, action_dim, hidden1=300, hidden2=300).to(self.device)
        self.q2_target = QNetwork(state_dim, z_dim, action_dim, hidden1=300, hidden2=300).to(self.device)

        self.q1_optim = Adam(self.q1.parameters(), lr=critic_lr)
        self.q2_optim = Adam(self.q2.parameters(), lr=critic_lr)

        # 初始化 Target 网络权重
        # copy weights
        self._hard_update(self.q1_target, self.q1)
        self._hard_update(self.q2_target, self.q2)

        # 【新增】初始化 Actor Target
        self._hard_update(self.policy_target, self.policy)

        # replay buffer
        self.replay = ReplayBuffer(int(args.get('replay_size', 200000)))
        self.batch_size = int(args.get('batch_sac', 64))

        # alpha (temperature) auto tune
        self.automatic_entropy_tuning = args.get('automatic_entropy_tuning', False)
        if self.automatic_entropy_tuning:
            self.target_entropy = float(args.get('target_entropy', -float(action_dim)))
            self.log_alpha = torch.tensor(np.log(alpha_fixed), requires_grad=True, device=self.device)
            self.alpha_optim = Adam([self.log_alpha], lr=actor_lr)
            self.alpha = alpha_fixed # 初始值
        else:
            self.alpha = alpha_fixed # 固定值

    def select_action(self, state, z, deterministic=False):
        """
        state: numpy array or tensor (state_dim,) or (state_dim,)
        z: numpy array or tensor (z_dim,) or (1,z_dim) or tensor
        returns: numpy array action (action_dim,)
        """
        state_t = to_tensor(state).unsqueeze(0).to(self.device) if not torch.is_tensor(state) else state
        z_t = to_tensor(z).unsqueeze(0).to(self.device) if not torch.is_tensor(z) else z
        if state_t.dim() == 1: state_t = state_t.unsqueeze(0)
        if z_t.dim() == 1: z_t = z_t.unsqueeze(0)

        if deterministic:
            a = self.policy.sample_deterministic(state_t, z_t)
        else:
            a, _, _ = self.policy.sample(state_t, z_t)
        return to_numpy(a).squeeze(0)

    def observe(self, s, a, r, s2, done, epoch_idx):
        """
        store transition with epoch index for env_model conditioning
        """
        self.replay.push(s, a, r, s2, done, epoch_idx)

    def update(self, env_model=None, current_epoch=0, current_beta=None):
        """
        Single update step (per sampled batch).
        env_model: object with get_z_batch(epoch_indices) -> tensor or numpy of shape (B, z_dim)
        decoder: optional, not used here but left for future extensions
        """
        if len(self.replay) < self.batch_size:
            return None

        # sample batch (s, a, r, s2, d, epoch_idx)
        s_b, a_b, r_b, s2_b, d_b, et_b = self.replay.sample(self.batch_size)

        # to tensors
        s = to_tensor(s_b).to(self.device)
        a = to_tensor(a_b).to(self.device)
        r = to_tensor(r_b).unsqueeze(1).to(self.device)
        s2 = to_tensor(s2_b).to(self.device)
        d = to_tensor(d_b).unsqueeze(1).to(self.device)

        # get z batch from env_model (if provided)
        if env_model is not None:
            z_batch = env_model.get_z_batch(et_b).to(self.device).float()
        else:
            # Fallback (仅用于调试，实际上应该总是有 env_model)
            z_batch = torch.zeros((s.shape[0], self.z_dim), device=self.device)

        # next z (for s2), 在论文设定中，Transition 发生在一个 epoch 内，z 不变
        z_next = z_batch

        # 1. Update Critic (Q1, Q2)
        with torch.no_grad():
            # 计算 next action 和 log_prob
            next_a, next_logp, _ = self.policy.sample(s2, z_next)
            
            # 计算 target Q
            q1_next = self.q1_target(s2, z_next, next_a)
            q2_next = self.q2_target(s2, z_next, next_a)
            min_q_next = torch.min(q1_next, q2_next)
            
            # 获取当前 alpha 值
            alpha_val = self.log_alpha.exp() if self.automatic_entropy_tuning else self.alpha

            # Target Q Value
            target_q = r + (1.0 - d) * self.gamma * (min_q_next - alpha_val * next_logp)

        # Current Q Estimates
        q1_pred = self.q1(s, z_batch, a)
        q2_pred = self.q2(s, z_batch, a)
        
        q1_loss = F.mse_loss(q1_pred, target_q)
        q2_loss = F.mse_loss(q2_pred, target_q)

        # Optimize Q1
        self.q1_optim.zero_grad()
        q1_loss.backward()
        self.q1_optim.step()

        # Optimize Q2
        self.q2_optim.zero_grad()
        q2_loss.backward()
        self.q2_optim.step()

        # 2. Update Actor (Policy)
        # sample 返回: action(带噪声), log_prob, mean(不带噪声)
        # 我们用 mean 来做一致性约束，这更稳定

        # sample 返回: action(带噪声), log_prob, mean(不带噪声)
        # 我们用 mean 来做一致性约束
        new_a, logp, current_mean = self.policy.sample(s, z_batch)
        q1_new = self.q1(s, z_batch, new_a)
        q2_new = self.q2(s, z_batch, new_a)
        min_q_new = torch.min(q1_new, q2_new)
        
        alpha_val = self.log_alpha.exp() if self.automatic_entropy_tuning else self.alpha

        # 原始 SAC Loss
        sac_loss = (alpha_val * logp - min_q_new).mean()

        # 【新增】计算一致性 Loss (Temporal Consistency)
        # 获取 Target Actor 的均值 (无梯度)
        with torch.no_grad():
            _, _, target_mean = self.policy_target.sample(s, z_batch)
        
        # 惩罚当前策略均值与历史策略均值的距离
        consistency_loss = F.mse_loss(current_mean, target_mean)
        
        # D. 确定 Beta 值 (优先使用传入的 current_beta)
        # 兼容旧逻辑
        # if current_epoch < 40:
        #     beta = 0.0
        # else:
        #     beta = self.consistency_beta
        if current_beta is not None:
            beta = current_beta
        else:
            # 兼容旧逻辑
            if current_epoch < 40:
                beta = 0.0
            else:
                beta = self.consistency_beta

        # 总 Loss
        policy_loss = sac_loss + beta * consistency_loss

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        # 3. Update Alpha (Optional, default OFF)
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp().item()

        # 4. Soft Update Targets
        self._soft_update(self.q1_target, self.q1)
        self._soft_update(self.q2_target, self.q2)
        # 【新增】同时软更新 Policy Target
        self._soft_update(self.policy_target, self.policy)

        return {
            "q1_loss": q1_loss.item(),
            "policy_loss": policy_loss.item(),
            # 【新增】
            "cons_loss": consistency_loss.item(),
            "alpha": alpha_val.item() if isinstance(alpha_val, torch.Tensor) else alpha_val
        }

    def _soft_update(self, target, source):
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_(tp.data * (1.0 - self.tau) + sp.data * self.tau)

    def _hard_update(self, target, source):
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_(sp.data)

    def save(self, path):
        import os
        os.makedirs(path, exist_ok=True)
        torch.save(self.policy.state_dict(), f"{path}/policy.pth")
        torch.save(self.policy_target.state_dict(), f"{path}/policy_target.pth") # 新增
        torch.save(self.q1.state_dict(), f"{path}/q1.pth")
        torch.save(self.q2.state_dict(), f"{path}/q2.pth")

    def load(self, path):
        self.policy.load_state_dict(torch.load(f"{path}/policy.pth"))
        if os.path.exists(f"{path}/policy_target.pth"): # 兼容旧权重
            self.policy_target.load_state_dict(torch.load(f"{path}/policy_target.pth"))
        self.q1.load_state_dict(torch.load(f"{path}/q1.pth"))
        self.q2.load_state_dict(torch.load(f"{path}/q2.pth"))

# # 训练循环里我们将通过 env_model.get_z_for_epoch / get_z_batch 获取 z。


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.optim import Adam
# import numpy as np

# from ..utils.tensor_utils import to_tensor, to_numpy
# from ..utils.buffer import ReplayBuffer
# from .networks import GaussianPolicy, QNetwork

# class SACAgent:
#     def __init__(self, state_dim, z_dim, action_dim, args: dict, device):
#         """
#         Standard SAC Agent without Consistency Regularization.
#         """
#         self.device = device if isinstance(device, torch.device) else torch.device(device)
        
#         # 参数提取
#         actor_lr = float(args.get('actor_lr', 1e-4))
#         critic_lr = float(args.get('critic_lr', 1e-3))
#         alpha_fixed = float(args.get('alpha', 0.1))
#         self.gamma = float(args.get('gamma', 0.99))
#         self.tau = float(args.get('tau', 0.005))
        
#         self.state_dim = state_dim
#         self.z_dim = z_dim
#         self.action_dim = action_dim

#         # --- 策略网络 & 优化器 ---
#         self.policy = GaussianPolicy(state_dim, z_dim, action_dim, hidden1=300, hidden2=300).to(self.device)
#         self.policy_optim = Adam(self.policy.parameters(), lr=actor_lr)

#         # --- Q网络 & 优化器 ---
#         self.q1 = QNetwork(state_dim, z_dim, action_dim, hidden1=300, hidden2=300).to(self.device)
#         self.q2 = QNetwork(state_dim, z_dim, action_dim, hidden1=300, hidden2=300).to(self.device)
#         self.q1_target = QNetwork(state_dim, z_dim, action_dim, hidden1=300, hidden2=300).to(self.device)
#         self.q2_target = QNetwork(state_dim, z_dim, action_dim, hidden1=300, hidden2=300).to(self.device)

#         self.q1_optim = Adam(self.q1.parameters(), lr=critic_lr)
#         self.q2_optim = Adam(self.q2.parameters(), lr=critic_lr)

#         # 初始化 Target 网络权重
#         self._hard_update(self.q1_target, self.q1)
#         self._hard_update(self.q2_target, self.q2)

#         # Replay Buffer
#         self.replay = ReplayBuffer(int(args.get('replay_size', 200000)))
#         self.batch_size = int(args.get('batch_sac', 64))

#         # Entropy Tuning
#         self.automatic_entropy_tuning = args.get('automatic_entropy_tuning', False)
#         if self.automatic_entropy_tuning:
#             self.target_entropy = float(args.get('target_entropy', -float(action_dim)))
#             self.log_alpha = torch.tensor(np.log(alpha_fixed), requires_grad=True, device=self.device)
#             self.alpha_optim = Adam([self.log_alpha], lr=actor_lr)
#             self.alpha = alpha_fixed
#         else:
#             self.alpha = alpha_fixed

#     def select_action(self, state, z, deterministic=False):
#         state_t = to_tensor(state).unsqueeze(0).to(self.device) if not torch.is_tensor(state) else state
#         z_t = to_tensor(z).unsqueeze(0).to(self.device) if not torch.is_tensor(z) else z
#         if state_t.dim() == 1: state_t = state_t.unsqueeze(0)
#         if z_t.dim() == 1: z_t = z_t.unsqueeze(0)

#         if deterministic:
#             a = self.policy.sample_deterministic(state_t, z_t)
#         else:
#             a, _, _ = self.policy.sample(state_t, z_t)
#         return to_numpy(a).squeeze(0)

#     def observe(self, s, a, r, s2, done, epoch_idx):
#         self.replay.push(s, a, r, s2, done, epoch_idx)

#     def update(self, env_model=None):
#         if len(self.replay) < self.batch_size:
#             return None

#         # 采样数据
#         s_b, a_b, r_b, s2_b, d_b, et_b = self.replay.sample(self.batch_size)
        
#         s = to_tensor(s_b).to(self.device)
#         a = to_tensor(a_b).to(self.device)
#         r = to_tensor(r_b).unsqueeze(1).to(self.device)
#         s2 = to_tensor(s2_b).to(self.device)
#         d = to_tensor(d_b).unsqueeze(1).to(self.device)

#         # 获取环境表征 z
#         if env_model is not None:
#             z_batch = env_model.get_z_batch(et_b).to(self.device).float()
#         else:
#             z_batch = torch.zeros((s.shape[0], self.z_dim), device=self.device)
        
#         z_next = z_batch 

#         # --- 1. Update Critic ---
#         with torch.no_grad():
#             next_a, next_logp, _ = self.policy.sample(s2, z_next)
#             q1_next = self.q1_target(s2, z_next, next_a)
#             q2_next = self.q2_target(s2, z_next, next_a)
#             min_q_next = torch.min(q1_next, q2_next)
            
#             alpha_val = self.log_alpha.exp() if self.automatic_entropy_tuning else self.alpha
#             target_q = r + (1.0 - d) * self.gamma * (min_q_next - alpha_val * next_logp)

#         q1_pred = self.q1(s, z_batch, a)
#         q2_pred = self.q2(s, z_batch, a)
#         q1_loss = F.mse_loss(q1_pred, target_q)
#         q2_loss = F.mse_loss(q2_pred, target_q)

#         self.q1_optim.zero_grad()
#         q1_loss.backward()
#         self.q1_optim.step()

#         self.q2_optim.zero_grad()
#         q2_loss.backward()
#         self.q2_optim.step()

#         # --- 2. Update Actor (Policy) ---
#         # 移除了一致性约束逻辑
#         new_a, logp, _ = self.policy.sample(s, z_batch)
        
#         q1_new = self.q1(s, z_batch, new_a)
#         q2_new = self.q2(s, z_batch, new_a)
#         min_q_new = torch.min(q1_new, q2_new)
        
#         alpha_val = self.log_alpha.exp() if self.automatic_entropy_tuning else self.alpha

#         # Standard SAC Loss
#         policy_loss = (alpha_val * logp - min_q_new).mean()

#         self.policy_optim.zero_grad()
#         policy_loss.backward()
#         self.policy_optim.step()

#         # --- 3. Update Alpha ---
#         if self.automatic_entropy_tuning:
#             alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
#             self.alpha_optim.zero_grad()
#             alpha_loss.backward()
#             self.alpha_optim.step()
#             self.alpha = self.log_alpha.exp().item()

#         # --- 4. Soft Update Targets ---
#         self._soft_update(self.q1_target, self.q1)
#         self._soft_update(self.q2_target, self.q2)
#         # 移除了 policy_target 的更新

#         return {
#             "q1_loss": q1_loss.item(),
#             "policy_loss": policy_loss.item(),
#             "alpha": alpha_val.item() if isinstance(alpha_val, torch.Tensor) else alpha_val
#         }

#     def _soft_update(self, target, source):
#         for tp, sp in zip(target.parameters(), source.parameters()):
#             tp.data.copy_(tp.data * (1.0 - self.tau) + sp.data * self.tau)

#     def _hard_update(self, target, source):
#         for tp, sp in zip(target.parameters(), source.parameters()):
#             tp.data.copy_(sp.data)

#     def save(self, path):
#         import os
#         os.makedirs(path, exist_ok=True)
#         torch.save(self.policy.state_dict(), f"{path}/policy.pth")
#         torch.save(self.q1.state_dict(), f"{path}/q1.pth")
#         torch.save(self.q2.state_dict(), f"{path}/q2.pth")

#     def load(self, path):
#         self.policy.load_state_dict(torch.load(f"{path}/policy.pth"))
#         self.q1.load_state_dict(torch.load(f"{path}/q1.pth"))
#         self.q2.load_state_dict(torch.load(f"{path}/q2.pth"))