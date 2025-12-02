# src/envs/prune_env.py

import torch
import numpy as np
from ..utils.flops_utils import calculate_conv_flops, calculate_block_flops, count_total_flops

class PruneEnv:
    def __init__(self, model_wrapper, train_loader, reward_loader, device, 
                K=20, prune_ratio_limit=0.5, 
                hard_ratio_limit=0.8, min_channels=4):
        self.model = model_wrapper
        self.train_loader = train_loader
        self.reward_loader = reward_loader
        self.device = device
        self.K = K

        # 保存约束参数
        self.hard_ratio_limit = hard_ratio_limit
        self.min_channels = min_channels

        # 持久化迭代器
        self.train_iter = iter(self.train_loader)

        self.num_layers = self.model.get_num_prunable_layers()
        
        # 更新后的总 FLOPs 计算 (Block level)
        self.original_flops = count_total_flops(self.model)
        self.target_flops = self.original_flops * (1.0 - prune_ratio_limit)
        
        print(f"Env Init: Original={self.original_flops/1e6:.2f}M, Target={self.target_flops/1e6:.2f}M")
        self.reset()

    def reset(self):
        self.model.restore_initial()
        self.current_idx = 0
        self.pruned_actions = []
        self.current_flops_cost = 0.0
        return self._build_state()

    def _get_current_and_future_flops(self, layer_idx):
        """辅助函数：计算当前层最大FLOPs和后续层预期FLOPs"""
        info = self.model.get_layer_static_info(layer_idx)
        
        # 1. 当前 Block 最大 FLOPs (Conv1+Conv2)
        current_max = calculate_block_flops(info, info['cin1'])
        
        # 2. 后续层
        future_orig = 0.0
        future_expected = 0.0
        
        for i in range(layer_idx + 1, self.num_layers):
            f_inf = self.model.get_layer_static_info(i)
            # 使用原始 cin1 估算未来 Block 的 FLOPs
            ff = calculate_block_flops(f_inf, f_inf['cin1'])
            future_orig += ff
            # future_expected += ff * 0.5 # 这一行被注释掉，不使用乐观估计
            future_expected += 0.0 # Placeholder if not using optimistic
            
        return current_max, future_orig, future_expected

    def _build_state(self):
        idx = self.current_idx
        info = self.model.get_layer_static_info(idx)
        
        cur_flops, fut_flops_orig, _ = self._get_current_and_future_flops(idx)
        
        state = np.array([
            idx / self.num_layers,
            # 2. C_l (输入通道数 - 结构固定值)
            info['cin1'] / 2048.0, 
            # 3. C_{l+1} (输出通道数 - 结构固定值)
            info['cout2'] / 2048.0,
            info['s1'],
            info['k1'],
            cur_flops / self.original_flops,
            self.current_flops_cost / self.original_flops,
            fut_flops_orig / self.original_flops,
            self.pruned_actions[-1] if idx > 0 else 0.0
        ], dtype=np.float32)
        
        return state

    def _calc_action_bounds(self, layer_idx):
        # 获取FLOPs信息 (future_expected 在这里没用上，因为被注释了)
        current_max, future_orig, _ = self._get_current_and_future_flops(layer_idx)
        
        used = self.current_flops_cost
        target = self.target_flops

        # 1. 预算约束 (Strict Paper Formulas)
        if current_max > 1e-6:
            # --- a_min (Eq. 12) ---
            # 假设未来层全剪光(0)，我现在最少剪多少？
            numerator_min = target - used
            min_keep_ratio = numerator_min / current_max
            a_min_budget = 1.0 - min_keep_ratio
            
            # --- a_max (Eq. 13) ---
            # 假设未来层全保留(100%，即 future_orig)，我现在最多剪多少？
            # numerator_max = target - used - future_flops_full
            numerator_max = target - used - future_orig
            max_keep_ratio = numerator_max / current_max
            a_max_budget = 1.0 - max_keep_ratio
        else:
            a_min_budget = 0.0
            a_max_budget = 1.0

        # 2. 物理约束 (Physical Constraints)
        info = self.model.get_layer_static_info(layer_idx)
        channel_limit = 1.0 - (self.min_channels / float(info['cout1']))
        a_max_physical = min(self.hard_ratio_limit, channel_limit)
        a_max_physical = max(0.0, a_max_physical)

        # 3. 整合 (取交集)    
        # 下界
        final_min = np.clip(a_min_budget, 0.0, a_max_physical)
        
        # 上界：取 预算约束 和 物理约束 的较小值
        final_max = min(a_max_budget, a_max_physical)
        final_max = np.clip(final_max, 0.0, 1.0)
        
        # 兜底
        if final_min > final_max:
            final_min = final_max

        return final_min, final_max

    def step(self, action):
        a_min, a_max = self._calc_action_bounds(self.current_idx)
        real_action = float(np.clip(action, a_min, a_max))
        
        self.pruned_actions.append(real_action)
        
        # 获取当前层信息
        info = self.model.get_layer_static_info(self.current_idx)
        keep_ratio = 1.0 - real_action
        
        # --- 计算实际消耗的 FLOPs (Block Level) ---
        # 剪枝后的中间通道数 (Conv1 Out / Conv2 In)
        pruned_mid = max(1, int(info['cout1'] * keep_ratio))
        
        # F1: Cin -> Mid
        f1 = calculate_conv_flops(
            info['h1'], info['w1'], info['k1'], info['s1'], 
            info['cin1'], pruned_mid
        )
        # F2: Mid -> Cout2
        f2 = calculate_conv_flops(
            info['h2'], info['w2'], info['k2'], info['s2'], 
            pruned_mid, info['cout2']
        )
        
        self.current_flops_cost += (f1 + f2)
        
        self.current_idx += 1
        done = (self.current_idx >= self.num_layers)
        
        reward = 0.0
        if done:
            # 统一生成 Mask
            masks = self.model.masks_from_action_ratios(self.pruned_actions)
            
            # # 1. 微调 K 步 (包含自动 BN 更新)
            # self._finetune_k_steps(masks)

            # A. 备份当前状态 (包含 Soft Weights 和当前的 BN 统计量)
            self.model.backup_weights()

            try:
                # B. 应用硬剪枝 (权重变 0)
                # 注意：_recalibrate_bn 内部需要 train 模式
                self._recalibrate_bn(masks) # 这一步会更新 BN，使其适应硬剪枝后的结构
                
                # C. 计算 Reward
                reward = self._validate(masks)
                
            finally:
                # D. 无论如何，恢复到备份状态
                # 这样 Phase 3 的后续训练依然是基于 Soft Weights 的
                self.model.restore_weights()

        next_state = np.zeros(9, dtype=np.float32) if done else self._build_state()

        return next_state, reward, done, {
            'real_action': real_action,
            'a_min': a_min,
            'a_max': a_max
        }

    def _recalibrate_bn(self, masks):
        """
        仅前向传播校准 BN 统计量。
        不计算梯度，不更新权重。严格符合 'Frozen Weights'。
        """
        self.model.base.train() # 必须开启 Train 模式以更新 running_mean/var
        self.model.apply_hard_prune(masks)
        
        with torch.no_grad(): # 关键：关闭梯度计算
            for _ in range(self.K):
                try:
                    x, _ = next(self.train_iter)
                except StopIteration:
                    self.train_iter = iter(self.train_loader)
                    x, _ = next(self.train_iter)
                
                x = x.to(self.device)
                self.model.base(x) # Forward 触发 BN 更新
                
                # Forward 可能会轻微改变 BN 参数的数值稳定性，保险起见再次 Apply Mask
                self.model.apply_hard_prune(masks)

    # def _finetune_k_steps(self, masks):
    #     optimizer = torch.optim.SGD(self.model.base.parameters(), lr=0.01)
    #     self.model.base.train()
    #     self.model.apply_hard_prune(masks)
        
    #     for _ in range(self.K):
    #         try:
    #             x, y = next(self.train_iter)
    #         except StopIteration:
    #             self.train_iter = iter(self.train_loader)
    #             x, y = next(self.train_iter)
                
    #         x, y = x.to(self.device), y.to(self.device)
    #         optimizer.zero_grad()
    #         out = self.model.base(x)
    #         loss = torch.nn.functional.cross_entropy(out, y)
    #         loss.backward()
    #         optimizer.step()
    #         self.model.apply_hard_prune(masks)

    def _validate(self, masks):
        self.model.base.eval()
        self.model.apply_hard_prune(masks)
        
        correct = 0
        total = 0
        with torch.no_grad():
            for i, (x, y) in enumerate(self.reward_loader):
                x, y = x.to(self.device), y.to(self.device)
                out = self.model.base(x)
                pred = out.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        return correct / (total + 1e-8)


# # src/envs/prune_env.py

# import torch
# import numpy as np
# from ..utils.flops_utils import calculate_conv_flops, count_total_flops

# class PruneEnv:
#     def __init__(self, model_wrapper, train_loader, reward_loader, device, K=20, prune_ratio_limit=0.5):
#         self.model = model_wrapper
#         self.train_loader = train_loader
#         self.reward_loader = reward_loader
#         self.device = device
#         self.K = K
        
#         # 创建持久化的迭代器，保证微调时数据是流动的，而不是每次都从头开始
#         self.train_iter = iter(self.train_loader)

#         self.num_layers = self.model.get_num_prunable_layers()
#         self.original_flops = count_total_flops(self.model)
#         self.target_flops = self.original_flops * (1.0 - prune_ratio_limit)
        
#         print(f"Env Init: Original={self.original_flops/1e6:.2f}M, Target={self.target_flops/1e6:.2f}M")
#         self.reset()

#     def reset(self):
#         self.model.restore_initial()
#         self.current_idx = 0
#         self.pruned_actions = []
#         self.current_flops_cost = 0.0
        
#         # 动态通道追踪
#         info = self.model.get_layer_static_info(0)
#         self.current_in_channels = info['cin'] 

#         return self._build_state()

#     def _get_current_and_future_flops(self, layer_idx):
#         """辅助函数：计算当前层最大FLOPs和后续层预期FLOPs"""
#         info = self.model.get_layer_static_info(layer_idx)
        
#         # 1. 当前层最大 FLOPs (基于动态输入通道)
#         current_max = calculate_conv_flops(
#             info['h'], info['w'], info['k'], info['s'], 
#             self.current_in_channels, info['cout']
#         )
        
#         # 2. 后续层 FLOPs (原始值，用于 State 归一化)
#         future_orig = 0.0
#         # 3. 后续层 FLOPs (预期值，用于 Bounds 计算)
#         future_expected = 0.0
        
#         for i in range(layer_idx + 1, self.num_layers):
#             f_inf = self.model.get_layer_static_info(i)
#             ff = calculate_conv_flops(
#                 f_inf['h'], f_inf['w'], f_inf['k'], f_inf['s'], 
#                 f_inf['cin'], f_inf['cout']
#             )
#             future_orig += ff
#             # 假设后续层平均剪掉 50% (乐观估计)
#             future_expected += ff * 0.5 
            
#         return current_max, future_orig, future_expected

#     def _build_state(self):
#         idx = self.current_idx
#         info = self.model.get_layer_static_info(idx)
        
#         # 获取FLOPs信息
#         cur_flops, fut_flops_orig, _ = self._get_current_and_future_flops(idx)
        
#         # 构建 State 向量
#         state = np.array([
#             idx / self.num_layers,                  # Idx
#             self.current_in_channels / 2048.0,      # Cin
#             info['cout'] / 2048.0,                  # Cout
#             info['s'],                              # Stride
#             info['k'],                              # Kernel
#             cur_flops / self.original_flops,        # Current FLOPs
#             self.current_flops_cost / self.original_flops, # Used FLOPs
#             fut_flops_orig / self.original_flops,   # Future FLOPs (Orig)
#             self.pruned_actions[-1] if idx > 0 else 0.0    # Last Action
#         ], dtype=np.float32)
        
#         return state

#     def _calc_action_bounds(self, layer_idx):
#         # 获取静态信息
#         info = self.model.get_layer_static_info(layer_idx)
        
#         # 1. 当前层 FLOPs
#         current_flops = calculate_conv_flops(
#             info['h'], info['w'], info['k'], info['s'], 
#             self.current_in_channels, info['cout']
#         )
        
#         # 2. 未来层 FLOPs (全量，用于 Eq. 13)
#         future_flops_full = 0.0
#         for i in range(layer_idx + 1, self.num_layers):
#             f_inf = self.model.get_layer_static_info(i)
#             future_flops_full += calculate_conv_flops(
#                 f_inf['h'], f_inf['w'], f_inf['k'], f_inf['s'], 
#                 f_inf['cin'], f_inf['cout']
#             )
            
#         used = self.current_flops_cost
#         target = self.target_flops

#         # 1. 预算约束 (Strict Paper Formulas)
#         if current_flops > 1e-6:
#             # --- 计算 a_min (Eq. 12) ---
#             # 论文公式：a_min = 1 - (Target - Used) / Current
#             # 含义：假设未来层全部被剪光(为0)，我现在最少要剪多少？
#             # 实际上它起到的作用是：当预算(Target)快用完时，强迫当前层剪枝。
#             numerator_min = target - used
#             # 注意：如果 numerator_min > current_flops，说明预算非常充裕
#             # keep_ratio > 1.0 -> prune_ratio < 0 -> a_min = 0
#             min_keep_ratio = numerator_min / current_flops
#             a_min_paper = 1.0 - min_keep_ratio
            
#             # --- 计算 a_max (Eq. 13) ---
#             # 论文公式：a_max = 1 - (Target - Used - Future_Full) / Current
#             # 含义：假设未来层完全不剪，我现在最多能剪多少，才不至于让总 FLOPs 低于 Target？
#             numerator_max = target - used - future_flops_full
#             max_keep_ratio = numerator_max / current_flops
#             a_max_paper = 1.0 - max_keep_ratio
#         else:
#             a_min_paper = 0.0
#             a_max_paper = 1.0

#         # 2. 物理约束 (Physical Constraints) - 依然建议保留以防崩塌
#         HARD_RATIO_LIMIT = 1
#         MIN_CHANNELS = 0

#         channel_limit = 1.0 - (MIN_CHANNELS / float(info['cout']))
#         a_max_physical = min(HARD_RATIO_LIMIT, channel_limit)
#         a_max_physical = max(0.0, a_max_physical)

#         # 3. 整合
#         # 下界：限制在 [0, 1] 且受物理上界约束
#         final_min = np.clip(a_min_paper, 0.0, a_max_physical)
        
#         # 上界：取 论文公式(13) 和 物理上界 的较小值
#         # 确保既不违反论文的 Target 下限约束，也不违反物理存活约束
#         final_max = min(a_max_paper, a_max_physical)
#         final_max = np.clip(final_max, 0.0, 1.0)
        
#         # 兜底：如果 min > max，说明无论怎么剪都无法同时满足所有约束
#         # 优先级：物理存活 > 预算下界 > 预算上界
#         if final_min > final_max:
#             final_min = final_max

#         return final_min, final_max

#     # def _calc_action_bounds(self, layer_idx):
#     #     info = self.model.get_layer_static_info(layer_idx)
        
#     #     # 1. 当前层 FLOPs
#     #     current_flops = calculate_conv_flops(
#     #         info['h'], info['w'], info['k'], info['s'], 
#     #         self.current_in_channels, info['cout']
#     #     )
        
#     #     # 2. 计算未来层 FLOPs (区分全量和期望)
#     #     future_flops_full = 0.0
#     #     future_flops_expected = 0.0 # 这里的 0.5 是核心区别
        
#     #     for i in range(layer_idx + 1, self.num_layers):
#     #         f_inf = self.model.get_layer_static_info(i)
#     #         ff = calculate_conv_flops(
#     #             f_inf['h'], f_inf['w'], f_inf['k'], f_inf['s'], 
#     #             f_inf['cin'], f_inf['cout']
#     #         )
#     #         future_flops_full += ff
#     #         future_flops_expected += ff * (1.0 - 0.5) # 假设未来层平均保留 50%

#     #     used = self.current_flops_cost
#     #     target = self.target_flops

#     #     # 1. 预算约束 (Optimistic Heuristic)
#     #     if current_flops > 1e-6:
#     #         # [下界 a_min] 修改版
#     #         # 含义：给未来层预留 50% 的预算后，我现在最少剪多少？
#     #         # 结果：前期 a_min 就会变成正数，逼迫 Agent 剪枝
#     #         numerator_min = target - used - future_flops_expected # <--- 关键区别
#     #         min_keep_ratio = numerator_min / current_flops
#     #         a_min_budget = 1.0 - min_keep_ratio
            
#     #         # [上界 a_max] 保持不变 (Eq. 13)
#     #         numerator_max = target - used - future_flops_full
#     #         max_keep_ratio = numerator_max / current_flops
#     #         a_max_budget = 1.0 - max_keep_ratio
#     #     else:
#     #         a_min_budget = 0.0
#     #         a_max_budget = 1.0

#     #     # 2. 物理约束 (同上)
#     #     HARD_RATIO_LIMIT = 0.8
#     #     MIN_CHANNELS = 4
#     #     channel_limit = 1.0 - (MIN_CHANNELS / float(info['cout']))
#     #     a_max_physical = min(HARD_RATIO_LIMIT, channel_limit)
#     #     a_max_physical = max(0.0, a_max_physical)

#     #     # 3. 整合
#     #     final_min = np.clip(a_min_budget, 0.0, a_max_physical)
#     #     final_max = min(a_max_budget, a_max_physical)
#     #     final_max = np.clip(final_max, 0.0, 1.0)
        
#     #     if final_min > final_max:
#     #         final_min = final_max

#     #     return final_min, final_max

#     def step(self, action):
#         a_min, a_max = self._calc_action_bounds(self.current_idx)
#         real_action = float(np.clip(action, a_min, a_max))
        
#         self.pruned_actions.append(real_action)
        
#         # 更新 FLOPs 和 通道信息
#         info = self.model.get_layer_static_info(self.current_idx)
#         keep_ratio = 1.0 - real_action
        
#         # 重新计算一遍当前 FLOPs (或者复用之前的计算结果，这里为了清晰重算一下)
#         current_flops = calculate_conv_flops(
#             info['h'], info['w'], info['k'], info['s'], 
#             self.current_in_channels, info['cout']
#         )
#         self.current_flops_cost += current_flops * keep_ratio
        
#         # 更新下一层的 Cin
#         self.current_in_channels = max(1, int(info['cout'] * keep_ratio))
        
#         self.current_idx += 1
#         done = (self.current_idx >= self.num_layers)
        
#         reward = 0.0
#         if done:
#             # 统一生成一次 Mask，后续复用
#             masks = self.model.masks_from_action_ratios(self.pruned_actions)
            
#             # 1. 微调 (K steps)
#             self._finetune_k_steps(masks)
            
#             # # 2. BN 校准 (Forward only)
#             # self._recalibrate_bn(masks, batch_count=30)
            
#             # 3. 验证
#             reward = self._validate(masks)

#         next_state = np.zeros(9, dtype=np.float32) if done else self._build_state()

#         return next_state, reward, done, {
#             'real_action': real_action,
#             'a_min': a_min,
#             'a_max': a_max
#         }

#     def _finetune_k_steps(self, masks):
#         """微调 K 步，使用持久化的迭代器"""
#         optimizer = torch.optim.SGD(self.model.base.parameters(), lr=0.01)
#         self.model.base.train()
#         self.model.apply_hard_prune(masks) # 初始 Apply
        
#         for _ in range(self.K):
#             try:
#                 x, y = next(self.train_iter)
#             except StopIteration:
#                 # 如果遍历完一遍，重新生成迭代器
#                 self.train_iter = iter(self.train_loader)
#                 x, y = next(self.train_iter)
                
#             x, y = x.to(self.device), y.to(self.device)
#             optimizer.zero_grad()
#             out = self.model.base(x)
#             loss = torch.nn.functional.cross_entropy(out, y)
#             loss.backward()
#             optimizer.step()
            
#             # SGD 更新后需重新 Apply Mask
#             self.model.apply_hard_prune(masks)

#     # def _recalibrate_bn(self, masks, batch_count=20):
#     #     """校准 BN 统计量"""
#     #     self.model.base.train()
#     #     # 只需在循环前 Apply 一次，因为这里不涉及梯度更新权重
#     #     self.model.apply_hard_prune(masks)
        
#     #     with torch.no_grad():
#     #         for i, (x, _) in enumerate(self.train_loader):
#     #             if i >= batch_count: break
#     #             x = x.to(self.device)
#     #             self.model.base(x)

#     def _validate(self, masks):
#         self.model.base.eval()
#         self.model.apply_hard_prune(masks)
        
#         correct = 0
#         total = 0
#         with torch.no_grad():
#             for i, (x, y) in enumerate(self.reward_loader):
#                 # if i > 20: break
#                 x, y = x.to(self.device), y.to(self.device)
#                 out = self.model.base(x)
#                 pred = out.argmax(dim=1)
#                 correct += (pred == y).sum().item()
#                 total += y.size(0)
#         return correct / (total + 1e-8)