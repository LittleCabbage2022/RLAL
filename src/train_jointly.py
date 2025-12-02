# 文件路径: src/train_jointly.py
# nohup python -u -m src.run_train --config configs/full.yaml >train.log 2>&1 &
# python -m src.run_train --config configs/full.yaml
import os
import time
import numpy as np
import torch
import random
import yaml
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from .data.dataloader import get_cifar10_loaders
from .models.model_wrapper import ModelWrapper
from .envs.prune_env import PruneEnv
from .agents.sac_agent import SACAgent
from .agents.env_model import EnvModel
from .agents.decoder import RewardDecoder
from .utils.tensor_utils import to_tensor
from .utils.buffer import ReplayBuffer

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(config):
    seed = int(config.get('seed', 42))
    set_seed(seed)

    # GPU 设置
    gpu_id = int(config.get('gpu_id', 0))
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    save_dir = config.get('save_dir', 'checkpoints')

    # 初始化 Writer
    # 将日志存在 checkpoints/你的实验名/runs 下
    log_dir = os.path.join(save_dir, 'runs')
    writer = SummaryWriter(log_dir=log_dir)
    print(f">> TensorBoard logging to {log_dir}")

    # 数据加载
    print(">> Data: Splitting 45k (Train) / 5k (Reward) / 10k (Test)")
    train_loader, reward_loader, val_loader = get_cifar10_loaders(
        batch_size=int(config['batch_size']), 
        seed=seed,
        split_for_reward=True  # 开启切分
    )

    # --- [修改 1] 初始化 ModelWrapper，传入 input_shape 用于计算 FLOPs ---
    input_shape = tuple(config.get('input_shape', [1, 3, 32, 32]))
    model = ModelWrapper(device, save_dir=save_dir, input_shape=input_shape)

    # --- [修改 2] 初始化 Env，传入 target_prune_rate ---
    target_prune_rate = float(config.get('target_prune_rate', 0.5))
    K_steps = int(config.get('K_train_steps', 20))
    # 读取约束参数
    hard_limit = float(config.get('hard_ratio_limit', 0.8))
    min_ch = int(config.get('min_channels', 4))

    env = PruneEnv(
        model, 
        train_loader,   # 45k 用于微调
        reward_loader,  # 5k 用于算 Reward (原 val_loader 位置)
        device, 
        K=K_steps,
        prune_ratio_limit=target_prune_rate,
        hard_ratio_limit=hard_limit,
        min_channels=min_ch
    )

    # --- [修改 3] 正确获取 State Dim ---
    init_state = env.reset()
    state_dim = len(init_state)
    print(f"Environment initialized. State dim: {state_dim}, Target Prune Rate: {target_prune_rate}")

    action_dim = 1
    z_dim = int(config.get('z_dim', 128))
    
    # 自动计算总 epochs
    warmup_epochs = int(config.get('warmup_epochs', 10))
    fill_epochs = int(config.get('fill_epochs', 10))
    agent_epochs = int(config.get('agent_epochs', 70))
    weights_only_epochs = int(config.get('weights_only_epochs', 100))
    
    calc_total = warmup_epochs + fill_epochs + agent_epochs + weights_only_epochs
    config_total = int(config.get('total_epochs', 0))
    if config_total > 0 and config_total != calc_total:
        print(f"Warning: Configured total_epochs ({config_total}) != Sum ({calc_total}). Using Sum.")
    
    num_epochs = calc_total

    # 全局定义优化器和 Scheduler，防止动量丢失
    lr = float(config.get('lr', 0.1))
    weight_decay = float(config.get('weight_decay', 5e-4))
    
    # 定义优化器 (针对 Model 权重)
    # 注意：Agent 训练阶段我们会冻结这个优化器，只用它来更新权重
    model_optimizer = torch.optim.SGD(
        model.base.parameters(), 
        lr=lr, 
        momentum=0.9, 
        weight_decay=weight_decay
    )
    
    # 定义学习率调度器 (MultiStepLR 是 ResNet 的标准配置)
    # 假设总 epoch 200，通常在 100 和 150 衰减
    milestones = [int(num_epochs * 0.5), int(num_epochs * 0.75)]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(model_optimizer, milestones=milestones, gamma=0.1)

    # 初始化模块
    env_model = EnvModel(num_epochs=num_epochs, embed_dim=z_dim, gru_hidden=z_dim, device=device)
    decoder = RewardDecoder(state_dim=state_dim, action_dim=action_dim, z_dim=z_dim, hidden=300).to(device)
    decoder_opt = torch.optim.Adam(
        list(env_model.parameters()) + list(decoder.parameters()), 
        lr=float(config.get('decoder_lr', 1e-3))
    )

    agent = SACAgent(state_dim=state_dim, z_dim=z_dim, action_dim=action_dim, args=config, device=device)
    # Replay Buffer 大小
    agent.replay = ReplayBuffer(int(config.get('replay_size', 200000)))

    # 参数
    lalign_beta = float(config.get('lalign_beta', 1e-4))
    weight_train_epochs_per_weights_phase = int(config.get('weight_train_epochs_per_weights_phase', 1))
    
    P = int(config.get('episodes_per_epoch', 10))
    decoder_updates = int(config.get('decoder_updates_per_epoch', 100))
    agent_updates = int(config.get('agent_updates_per_epoch', 200))

    current_epoch = 0

    # ================= 1. Warmup Phase =================
    print("\n=== [Phase 1] Warmup (Weights Training) ===")
    for e in range(warmup_epochs):
        current_epoch += 1
        # 传入全局 optimizer
        train_one_epoch_weights(model, train_loader, device, model_optimizer)
        scheduler.step() # 更新 LR
        
        if current_epoch % int(config.get('eval_interval', 5)) == 0:
            acc = evaluate(model, val_loader, device)
            print(f"Epoch {current_epoch} Val Acc: {acc:.4f} | LR: {scheduler.get_last_lr()[0]:.5f}")
            writer.add_scalar('Eval/Accuracy', acc, current_epoch)

    print(">> Warmup done. Updating base weights for Agent...")
    model.checkpoint_current_weights()

    # ================= 2. Fill Buffer Phase =================
    print("\n=== [Phase 2] Fill Buffer (Pure Random Exploration) ===")
    for e in range(fill_epochs):
        current_epoch += 1
        print(f"Epoch {current_epoch}/{num_epochs}")
        for p in range(P):
            state = env.reset()
            done = False
            ep_reward = 0.0
            while not done:
                z = env_model.get_z_for_epoch(current_epoch - 1).to(device)
                
                # 【修改点】：不要用 agent.select_action，改用纯随机
                # 你的动作空间是 [0, 1] 之间的连续值
                # 直接生成一个均匀分布的随机动作
                raw_action = np.random.uniform(0.0, 1.0) 
                
                # Step
                next_state, reward, done, info = env.step(raw_action)
                real_action = info['real_action']
                
                # 存入 Buffer
                agent.observe(state, np.array([real_action], dtype=np.float32), reward, next_state, float(done), current_epoch - 1)
                state = next_state
                ep_reward += reward
            print(f"  Fill Episode {p+1}/{P}: Reward={ep_reward:.4f}")

    # ================= 3. Agent Training Phase =================
    print("\n=== [Phase 3] Joint Agent Training & Pruning Search ===")
    for e in range(agent_epochs):
        current_epoch += 1
        print(f"Epoch {current_epoch}/{num_epochs}")
        
        # 3.1 收集数据
        for p in range(P):
            state = env.reset()
            done = False
            ep_reward = 0.0
            actions_record = []
            
            while not done:
                z = env_model.get_z_for_epoch(current_epoch - 1).to(device)
                action = agent.select_action(state, z, deterministic=False)
                
                next_state, reward, done, info = env.step(float(action[0]) if hasattr(action, '__len__') else float(action))
                
                # --- 记录真实动作 ---
                real_action = info['real_action']
                actions_record.append(real_action)
                
                agent.observe(state, np.array([real_action], dtype=np.float32), reward, next_state, float(done), current_epoch - 1)
                state = next_state
                ep_reward += reward
            
            # 更新最佳策略
            model.update_best_policy(actions_record, ep_reward)
            print(f"  Episode {p+1}/{P}: Reward={ep_reward:.4f}")

            # 计算全局步数，这样 TensorBoard 横坐标就是平滑的
            global_step = (current_epoch - 1) * P + p
            writer.add_scalar('Pruning/Episode_Reward', ep_reward, global_step)
            writer.add_scalar('Pruning/Avg_Pruning_Rate', np.mean(actions_record), global_step)

        # 3.2 训练 Decoder
        decoder_losses = []
        for _ in range(decoder_updates):
            if len(agent.replay) < agent.batch_size: break
            s, a, r, s2, d, et = agent.replay.sample(agent.batch_size)
            
            s_t = to_tensor(s).to(device)
            a_t = to_tensor(a).to(device)
            r_t = to_tensor(r).unsqueeze(1).to(device)
            z_batch = env_model.get_z_batch(et).to(device)
            
            rhat = decoder(s_t, a_t, z_batch)
            loss = F.mse_loss(rhat, r_t)
            
            decoder_opt.zero_grad()
            loss.backward()
            decoder_opt.step()
            decoder_losses.append(loss.item())
        
        if decoder_losses:
            print(f"  >> Decoder Loss: {np.mean(decoder_losses):.6f}")

        # 3.3 训练 Agent (SAC)
        # 【新增】计算当前的 Beta 值
        # 从 config 获取最大值，默认 500 (你测出的最佳值)
        max_beta = float(config.get('consistency_beta', 500.0))
        # current_beta = get_dynamic_beta(current_epoch, max_beta)
        current_beta = None

        sac_logs = []
        for _ in range(agent_updates):
            out = agent.update(env_model=env_model, current_epoch=current_epoch, current_beta=current_beta) # Decoder 暂时没在 update 里用，如果需要可用
            if out: sac_logs.append(out)
     
        if sac_logs:
            # 提取各项 loss 的平均值
            avg_q1 = np.mean([x['q1_loss'] for x in sac_logs])
            avg_policy = np.mean([x['policy_loss'] for x in sac_logs])
            avg_alpha = np.mean([x['alpha'] for x in sac_logs])
            # 使用 .get() 防止旧代码报错
            avg_cons = np.mean([x.get('cons_loss', 0.0) for x in sac_logs])

            print(f"  >> SAC Q1 Loss: {avg_q1:.4f}, Policy Loss: {avg_policy:.4f}, Consistency Loss: {avg_cons:.4f}")

            # 写入 (使用 current_epoch 作为横坐标)
            writer.add_scalar('Agent/Loss_Q1', avg_q1, current_epoch)
            writer.add_scalar('Agent/Loss_Policy_Total', avg_policy, current_epoch)
            writer.add_scalar('Agent/Loss_Consistency', avg_cons, current_epoch) # 重点监控这个！
            writer.add_scalar('Agent/Alpha', avg_alpha, current_epoch)

        # 3.4 训练权重 (Normal Training) 
        # 这里也用全局 optimizer，保持动量
        # Step 8: 获取当前认为最好的剪枝结构 (Mask)
        current_best_masks = model.get_masks_from_best_policy()
        
        # Step 9: 更新模型参数
        if current_best_masks is None:
            # 如果还没找到 best policy (极少见)，退化为普通训练
            train_one_epoch_weights(model, train_loader, device, model_optimizer)
        else:
            # [关键] 使用 Lalign Loss 进行训练
            # 这会让保留下来的权重更强，被剪掉的权重趋向于 0
            avg_ce, avg_align = train_one_epoch_weights_with_lalign(
                model, 
                train_loader, 
                device, 
                lalign_beta,        # 使用配置文件里的 beta (如 1e-5)
                current_best_masks, # 传入当前的 Best Mask
                model_optimizer     # 传入全局优化器
            )
            # 写入 TensorBoard
            writer.add_scalar('Weights/Loss_CE', avg_ce, current_epoch)
            writer.add_scalar('Weights/Loss_Align_Raw', avg_align, current_epoch)
            writer.add_scalar('Weights/Loss_Align_Weighted', avg_align * lalign_beta, current_epoch)
        
        scheduler.step()

        # 在这一轮权重训练(Lalign)结束后，更新存档点
        # 这样下一个 Epoch 的 Agent 就能基于更强的模型进行剪枝
        model.checkpoint_current_weights()

        if current_epoch % int(config.get('eval_interval', 5)) == 0:
            acc = evaluate(model, val_loader, device)
            current_lr = scheduler.get_last_lr()[0]
            print(f"Epoch {current_epoch} Val Acc: {acc:.4f} | LR: {current_lr:.5f}")
            writer.add_scalar('Eval/Accuracy', acc, current_epoch)

    # ================= 4. Weights-only (Lalign) Phase =================
    print("\n=== [Phase 4] Weights Training with Lalign (Alignment) ===")
    masks_best = model.get_masks_from_best_policy()
    
    if masks_best is not None:
        print("  >> Best policy loaded. Applying Soft Alignment...")
        for e in range(weights_only_epochs):
            current_epoch += 1
            
            # 使用最佳 Mask 进行 Lalign 训练
            # 注意：apply_hard_prune 只是把权重置0，这里我们每次 iter 都要确保这一点
            model.apply_hard_prune(masks_best)
            
            for _ in range(weight_train_epochs_per_weights_phase):
                avg_ce, avg_align = train_one_epoch_weights_with_lalign(
                    model, train_loader, device, lalign_beta, masks_best, model_optimizer
                )

            # 写入 TensorBoard (Phase 4 也要记录，看看 Loss 降没降)
            writer.add_scalar('Weights/Loss_CE', avg_ce, current_epoch)
            writer.add_scalar('Weights/Loss_Align_Raw', avg_align, current_epoch)
            writer.add_scalar('Weights/Loss_Align_Weighted', avg_align * lalign_beta, current_epoch)

            scheduler.step()

            if current_epoch % int(config.get('eval_interval', 5)) == 0:
                acc = evaluate(model, val_loader, device)
                print(f"Epoch {current_epoch}/{num_epochs} Val Acc: {acc:.4f}")
                writer.add_scalar('Eval/Accuracy', acc, current_epoch)
    else:
        print("  >> Warning: No best policy found! Skipping Lalign phase.")

    # # ================= 5. Fine-tuning Phase =================
    # print("\n=== [Phase 5] Final Fine-tuning ===")
    # # 理想情况下，这里应该进行真正的结构化剪枝（改变 channel 数）
    # # 目前作为验证，我们继续使用 Mask 训练
    # for e in range(finetune_epochs):
    #     current_epoch += 1
        
    #     # 每次训练前确保 Mask 生效（Soft Pruning 模拟）
    #     if masks_best is not None:
    #          model.apply_hard_prune(masks_best)
             
    #     train_one_epoch_weights(model, train_loader, device, model_optimizer)
    #     scheduler.step()
        
    #     if current_epoch % int(config.get('eval_interval', 5)) == 0:
    #         acc = evaluate(model, val_loader, device)
    #         print(f"Epoch {current_epoch}/{num_epochs} Val Acc: {acc:.4f}")

    # Save
    os.makedirs(config.get('save_dir', 'checkpoints'), exist_ok=True)
    model.save_checkpoint(config.get('save_dir', 'checkpoints'))
    writer.close()
    print("writer closed.")
    print("Training Complete.")

# ---------------- 辅助函数 ----------------
def train_one_epoch_weights(model_wrapper, train_loader, device, optimizer):
    model_wrapper.base.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model_wrapper.forward(x)
        loss = F.cross_entropy(out, y)
        loss.backward()
        optimizer.step()

def train_one_epoch_weights_with_lalign(model_wrapper, train_loader, device, lalign_beta, masks, optimizer):
    model_wrapper.base.train()

    # 累加器
    total_ce = 0.0
    total_align = 0.0
    steps = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model_wrapper.forward(x)
        ce_loss = F.cross_entropy(out, y)
        
        # 计算对齐正则项
        lalign_loss = model_wrapper.compute_lalign(masks)
        
        loss = ce_loss + lalign_beta * lalign_loss
        loss.backward()
        optimizer.step()
        
        # 记录
        total_ce += ce_loss.item()
        total_align += lalign_loss.item()
        steps += 1

        # 再次应用 Mask 保持投影
        # model_wrapper.apply_hard_prune(masks)
    return total_ce / steps, total_align / steps

def evaluate(model_wrapper, val_loader, device, max_batches=None):
    model_wrapper.base.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            if max_batches and i >= max_batches: break
            x, y = x.to(device), y.to(device)
            out = model_wrapper.forward(x)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total

def get_dynamic_beta(epoch, max_beta):
    # 阶段 1: 自由探索 (0 - 40 Epoch)
    if epoch < 40:
        return 0.0
    
    # 阶段 2: 线性爬坡 (40 - 90 Epoch)
    # 这里的 90 是 Phase 3 结束的时间点 (Warmup 10 + Fill 10 + Agent 70 = 90)
    # 让 Beta 在 Agent 训练结束时达到最大值，为了接下来的 Weights-Only Phase 做最稳的准备
    elif epoch < 90:
        # 计算进度 (0.0 ~ 1.0)
        ratio = (epoch - 40) / (90 - 40)
        return max_beta * ratio
        
    # 阶段 3: 保持最大 (90+ Epoch)
    else:
        return max_beta

if __name__ == '__main__':
    # 这里的 config 可以通过 argparse 覆盖，或者直接读文件
    cfg = yaml.safe_load(open('configs/default.yaml'))
    train(cfg)