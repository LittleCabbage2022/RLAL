# src/models/model_wrapper.py

import torch
import torch.nn as nn
import numpy as np
import os
import json
from copy import deepcopy
from .resnet_56_cifar import resnet56, BasicBlock

class ModelWrapper:
    def __init__(self, device, save_dir='checkpoints', input_shape=(1, 3, 32, 32)):
        self.device = torch.device(device)
        self.base = resnet56()
        self.base.to(self.device)
        
        # 保存初始权重用于 Reset
        self.init_state = deepcopy(self.base.state_dict())

        # 改为存储 Block 信息 (Conv1 + Conv2)
        self._prunable_blocks = None 
        self._layer_info = {}

        # 初始化层信息
        self.get_prunable_blocks() 
        self._fetch_layer_info(input_shape)

        self.best_policy = None 
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.best_policy_path = os.path.join(self.save_dir, 'best_policy.json')


    def get_prunable_blocks(self):
        """
        获取 ResNet 的 Residual Blocks。
        每个 Block 包含 conv1 (剪输出) 和 conv2 (剪输入)。
        """
        if self._prunable_blocks is None:
            blocks = []
            for m_name, module in self.base.named_modules():
                if isinstance(module, BasicBlock):
                    # 记录一对卷积层
                    blocks.append({
                        'name': m_name,
                        'conv1': module.conv1,
                        'conv2': module.conv2,
                        'bn1': module.bn1 
                    })
            self._prunable_blocks = blocks
            print(f"ModelWrapper: Detected {len(blocks)} prunable blocks (Conv1+Conv2 pairs).")
        return self._prunable_blocks

    def get_num_prunable_layers(self):
        return len(self.get_prunable_blocks())

    def _fetch_layer_info(self, input_shape):
        """
        运行一次虚拟前向传播，记录每一层的 Feature Map 尺寸。
        同时记录 Conv1 和 Conv2 的静态参数。
        """
        self.base.eval()
        x = torch.zeros(input_shape).to(self.device)
        
        handles = []
        temp_shapes = {}

        def get_hook(idx, layer_type):
            def hook(module, input, output):
                if idx not in temp_shapes: temp_shapes[idx] = {}
                # 记录特征图大小 (Batch, C, H, W)
                temp_shapes[idx][layer_type] = (input[0].shape[2], input[0].shape[3])
            return hook

        blocks = self.get_prunable_blocks()
        for i, blk in enumerate(blocks):
            # Hook Conv1 和 Conv2
            h1 = blk['conv1'].register_forward_hook(get_hook(i, 'conv1'))
            h2 = blk['conv2'].register_forward_hook(get_hook(i, 'conv2'))
            handles.extend([h1, h2])
        
        with torch.no_grad():
            self.base(x)
        
        for h in handles: h.remove()
        
        # 整理 Info
        for i, blk in enumerate(blocks):
            c1 = blk['conv1']
            c2 = blk['conv2']
            
            # 记录 Block 的综合信息
            # 我们主要关心 Conv1 的输出变化如何影响整体
            self._layer_info[i] = {
                # Conv1 的参数
                'h1': temp_shapes[i]['conv1'][0],
                'w1': temp_shapes[i]['conv1'][1],
                'k1': c1.kernel_size[0],
                's1': c1.stride[0],
                'cin1': c1.in_channels,
                'cout1': c1.out_channels, # 这是我们要剪的维度
                
                # Conv2 的参数 (它的输入维度 = Conv1 的输出维度)
                'h2': temp_shapes[i]['conv2'][0],
                'w2': temp_shapes[i]['conv2'][1],
                'k2': c2.kernel_size[0],
                's2': c2.stride[0],
                'cout2': c2.out_channels 
            }

    def get_layer_static_info(self, layer_idx):
        return self._layer_info[layer_idx]

    def masks_from_action_ratios(self, actions):
        """
        生成 Mask。
        基于 Conv1 的权重 L1-norm 进行排序。
        """
        masks = []
        blocks = self.get_prunable_blocks()
        
        for idx, ratio in enumerate(actions):
            c1 = blocks[idx]['conv1']
            out_ch = c1.out_channels
            
            # 至少保留 1 个通道 (防止除零错误，虽然 min_channels 会控制)
            keep = max(1, int(np.round(out_ch * (1.0 - float(ratio)))))
            
            # L1-norm ranking based on Conv1 weights
            weight = c1.weight.data.detach().abs().view(out_ch, -1).sum(dim=1).cpu().numpy()
            # 降序排列，取前 keep 个最大的
            topk_idx = np.argsort(-weight)[:keep] 
            
            mask = np.zeros(out_ch, dtype=np.float32)
            mask[topk_idx] = 1.0
            masks.append(mask)
        return masks

    def apply_hard_prune(self, masks):
        """
        同时应用到 Conv1 的输出 和 Conv2 的输入
        """
        blocks = self.get_prunable_blocks()
        for blk, mask in zip(blocks, masks):
            # 1. Prune Conv1 Output (dim 0)
            w1 = blk['conv1'].weight.data
            m_out = torch.tensor(mask, dtype=w1.dtype, device=w1.device).view(-1, 1, 1, 1)
            blk['conv1'].weight.data = w1 * m_out
            
            if blk['conv1'].bias is not None:
                blk['conv1'].bias.data *= torch.tensor(mask, dtype=w1.dtype, device=w1.device)
                
            # 2. Prune BN1 (Batch Norm 跟着 Conv1 走)
            if blk['bn1'] is not None:
                m_bn = torch.tensor(mask, dtype=w1.dtype, device=w1.device)
                blk['bn1'].weight.data *= m_bn
                blk['bn1'].bias.data *= m_bn
                blk['bn1'].running_mean.data *= m_bn
                blk['bn1'].running_var.data *= m_bn

            # 3. Prune Conv2 Input (dim 1)
            # Conv2 的权重形状是 (Out, In, K, K)，我们要剪 Dim 1
            w2 = blk['conv2'].weight.data
            m_in = torch.tensor(mask, dtype=w2.dtype, device=w2.device).view(1, -1, 1, 1)
            blk['conv2'].weight.data = w2 * m_in

    def compute_lalign(self, masks):
        """
        计算对齐正则化损失 (Alignment Loss).
        包含两种实现方式：
        1. Squared L2 (默认): 数学上是 Group Weight Decay，工程上收敛更稳。
        2. Strict L2 Norm: 数学上是 Group Lasso，稀疏性更强，但梯度更陡峭。
        """
        total_loss = torch.tensor(0.0, device=self.device)
        blocks = self.get_prunable_blocks()
        
        for blk, mask in zip(blocks, masks):
            # --- 1. 处理 Conv1 (剪输出通道) ---
            w1 = blk['conv1'].weight
            # 构造反向 Mask (1 - v_l)
            # view(-1, 1, 1, 1) 对应 (C_out, C_in, K, K) 的 C_out 维度
            m_out = torch.tensor(mask, dtype=w1.dtype, device=w1.device).view(-1, 1, 1, 1)
            # 计算被“剪掉”部分的权重残差
            residual_1 = (1.0 - m_out) * w1
            
            # --- 2. 处理 Conv2 (剪输入通道) ---
            w2 = blk['conv2'].weight
            # view(1, -1, 1, 1) 对应 (C_out, C_in, K, K) 的 C_in 维度
            m_in = torch.tensor(mask, dtype=w2.dtype, device=w2.device).view(1, -1, 1, 1)
            residual_2 = (1.0 - m_in) * w2
            
            # ==========================================================
            # [Option A] Squared L2 Norm (平方和)
            # ==========================================================
            # 公式: sum(w^2)
            # 特点: 梯度平滑 (2*beta*w)，对大权重惩罚重，对小权重惩罚轻。
            # 适用参数: lalign_beta 推荐 1e-4
            # ----------------------------------------------------------
            # total_loss = total_loss + torch.sum(residual_1 ** 2) + torch.sum(residual_2 ** 2)
            
            # ==========================================================
            # [Option B] Strict L2 Norm (Group Lasso)
            # ==========================================================
            # 公式: ||w||_2 (即 sqrt(sum(w^2)))，完全符合论文 Eq.10
            # 特点: 梯度恒定 (beta)，对趋近于0的小权重依然有强大的推力，稀疏性极强。
            # 风险: 如果 beta 太大，会强行“杀死”所有权重，导致欠拟合(Reward下降)。
            # 【重要】如果切换到此选项，请务必将 lalign_beta 降低到 1e-5 或 2e-5 !!!
            # ----------------------------------------------------------
            norm1 = torch.norm(residual_1, p=2)
            norm2 = torch.norm(residual_2, p=2)
            total_loss = total_loss + norm1 + norm2
            
        return total_loss

    def checkpoint_current_weights(self):
        self.init_state = deepcopy(self.base.state_dict())

    def restore_initial(self):
        self.base.load_state_dict(self.init_state)

    def update_best_policy(self, actions, reward):
        if self.best_policy is None or reward > float(self.best_policy['reward']):
            self.best_policy = {'actions': [float(a) for a in actions], 'reward': float(reward)}
            with open(self.best_policy_path, 'w') as f:
                json.dump(self.best_policy, f)
            print(f"[BestPolicy] Updated best reward: {reward:.4f}")
            return True
        return False

    def get_masks_from_best_policy(self):
        if self.best_policy is None:
            if os.path.exists(self.best_policy_path):
                with open(self.best_policy_path, 'r') as f:
                    self.best_policy = json.load(f)
        if self.best_policy is None: return None
        return self.masks_from_action_ratios(self.best_policy['actions'])

    def save_checkpoint(self, path=None):
        target_dir = path if path else self.save_dir
        os.makedirs(target_dir, exist_ok=True)
        torch.save(self.base.state_dict(), os.path.join(target_dir, 'student_model.pth'))
        if self.best_policy is not None:
            with open(os.path.join(target_dir, 'best_policy.json'), 'w') as f:
                json.dump(self.best_policy, f)

    def forward(self, x):
        return self.base(x)

    def backup_weights(self):
        """
        备份当前的模型参数（权重+BN统计量）。
        用于在评估/剪枝前保存“软权重”状态。
        """
        # 使用 CPU 备份以节省显存，或者用 GPU 换取速度
        # 这里为了安全起见，deepcopy 到 CPU
        self._backup_state = deepcopy(self.base.state_dict())

    def restore_weights(self):
        """
        恢复之前备份的模型参数。
        """
        if hasattr(self, '_backup_state') and self._backup_state is not None:
            self.base.load_state_dict(self._backup_state)
            #以此释放内存
            del self._backup_state
        else:
            print(">> Warning: No backup found to restore!")


# # src/models/model_wrapper.py

# import torch
# import torch.nn as nn
# import numpy as np
# import os
# import json
# from copy import deepcopy

# # 引入刚才定义的 ResNet
# from .resnet_56_cifar import resnet56, BasicBlock

# class ModelWrapper:
#     def __init__(self, device, input_shape=(1, 3, 32, 32)):
#         self.device = torch.device(device)

#         self.base = resnet56()
#         self.base.to(self.device)
        
#         # 保存初始权重用于 Reset
#         self.init_state = deepcopy(self.base.state_dict())

#         self._conv_layers = None
#         self._layer_info = {}

#         # 初始化层信息
#         # 注意：get_conv_layers 现在逻辑变了，会先执行
#         self.get_conv_layers() 
#         self._fetch_layer_info(input_shape)

#         self.best_policy = None 
#         self.best_policy_path = 'checkpoints/best_policy.json'
#         os.makedirs('checkpoints', exist_ok=True)

#     def get_conv_layers(self):
#         """
#         关键修改：
#         我们只希望 Agent 剪枝每个 Residual Block 内部的第一个卷积 (conv1)。
#         如果剪 conv2，输出维度变了，Shortcut 加法就会报错。
#         """
#         if self._conv_layers is None:
#             convs = []
#             # 遍历所有模块，找到 BasicBlock
#             for m_name, module in self.base.named_modules():
#                 if isinstance(module, BasicBlock):
#                     # 将 Block 内部的 conv1 加入列表
#                     # 名字拼接一下以便区分: layer1.0.conv1
#                     full_name = f"{m_name}.conv1"
#                     convs.append((full_name, module.conv1))
            
#             self._conv_layers = convs
#             # ResNet-56 应该有 9*3 = 27 个可剪枝层
#             print(f"ModelWrapper: Detected {len(convs)} prunable layers (BasicBlock.conv1 only).")
#         return self._conv_layers

#     def _fetch_layer_info(self, input_shape):
#         self.base.eval()
#         x = torch.zeros(input_shape).to(self.device)
        
#         handles = []
#         # 用来存临时 hook 数据的字典
#         temp_shapes = {}

#         def get_hook(idx):
#             def hook(module, input, output):
#                 # input[0] is (B, C, H, W)
#                 if idx not in temp_shapes:
#                     temp_shapes[idx] = {}
#                 temp_shapes[idx]['h'] = input[0].shape[2]
#                 temp_shapes[idx]['w'] = input[0].shape[3]
#             return hook

#         convs = self.get_conv_layers()
#         for i, (name, module) in enumerate(convs):
#             h = module.register_forward_hook(get_hook(i))
#             handles.append(h)
        
#         with torch.no_grad():
#             self.base(x)
        
#         for h in handles: h.remove()
        
#         # 整理 info
#         for i, (name, module) in enumerate(convs):
#             if i not in self._layer_info:
#                 self._layer_info[i] = {}
            
#             # 从 temp_shapes 获取动态的 H/W
#             self._layer_info[i]['h'] = temp_shapes[i]['h']
#             self._layer_info[i]['w'] = temp_shapes[i]['w']
            
#             # 静态信息
#             k = module.kernel_size[0] if isinstance(module.kernel_size, tuple) else module.kernel_size
#             s = module.stride[0] if isinstance(module.stride, tuple) else module.stride
            
#             self._layer_info[i].update({
#                 'k': k,
#                 's': s,
#                 'cin': module.in_channels,
#                 'cout': module.out_channels
#             })

#     def get_layer_static_info(self, layer_idx):
#         return self._layer_info[layer_idx]

#     def get_num_prunable_layers(self):
#         return len(self.get_conv_layers())

#     def restore_initial(self):
#         self.base.load_state_dict(self.init_state)
#         # 重新 fetch info 并不需要，除非结构变了，但 restore 只是恢复权重值
    
#     def checkpoint_current_weights(self):
#         """
#         将当前的权重保存为 init_state。
#         在 Warmup 结束或每个 Training Epoch 结束后调用。
#         这样 Env.reset() 就会恢复到这个训练好的状态，而不是随机状态。
#         """
#         self.init_state = deepcopy(self.base.state_dict())
#         # print(">> [ModelWrapper] Base weights updated for RL environment.")

#     def masks_from_action_ratios(self, actions):
#         """
#         生成 Mask。
#         注意：actions 列表长度 = prunable layers 数量 (27)。
#         """
#         masks = []
#         convs = self.get_conv_layers()
        
#         for idx, ratio in enumerate(actions):
#             name, module = convs[idx]
#             out_ch = module.weight.data.shape[0]
            
#             # 至少保留 1 个通道
#             keep = max(1, int(np.round(out_ch * (1.0 - float(ratio)))))
            
#             # L1-norm 排序筛选
#             weight = module.weight.data.detach().abs().view(out_ch, -1).sum(dim=1).cpu().numpy()
#             topk_idx = np.argsort(-weight)[:keep]
            
#             mask = np.zeros(out_ch, dtype=np.float32)
#             mask[topk_idx] = 1.0
#             masks.append(mask)
#         return masks

#     def compute_lalign(self, masks):
#         total = torch.tensor(0.0, device=self.device)
#         convs = self.get_conv_layers()
        
#         # 只计算被剪枝层的 Alignment Loss
#         for (name, module), mask in zip(convs, masks):
#             w = module.weight
#             m = torch.tensor(mask, dtype=w.dtype, device=w.device).view(-1, 1, 1, 1)
#             residual = (1.0 - m) * w
#             total = total + torch.sum(residual * residual)
#         return total

#     def apply_hard_prune(self, masks):
#         convs = self.get_conv_layers()
#         for (name, module), mask in zip(convs, masks):
#             w = module.weight.data
#             m = torch.tensor(mask, dtype=w.dtype, device=w.device).view(-1,1,1,1)
#             module.weight.data = w * m
#             if hasattr(module, 'bias') and module.bias is not None:
#                 b = module.bias.data
#                 module.bias.data = b * m.view(-1)

#     def update_best_policy(self, actions, reward):
#         if self.best_policy is None or reward > float(self.best_policy['reward']):
#             self.best_policy = {'actions': [float(a) for a in actions], 'reward': float(reward)}
#             with open(self.best_policy_path, 'w') as f:
#                 json.dump(self.best_policy, f)
#             print(f"[BestPolicy] Updated best reward: {reward:.4f}")
#             return True
#         return False

#     def get_masks_from_best_policy(self):
#         if self.best_policy is None:
#             if os.path.exists(self.best_policy_path):
#                 with open(self.best_policy_path, 'r') as f:
#                     self.best_policy = json.load(f)
#         if self.best_policy is None: return None
#         return self.masks_from_action_ratios(self.best_policy['actions'])

#     def save_checkpoint(self, path):
#         os.makedirs(path, exist_ok=True)
#         torch.save(self.base.state_dict(), os.path.join(path, 'student_model.pth'))
#         if self.best_policy is not None:
#             with open(os.path.join(path, 'best_policy.json'), 'w') as f:
#                 json.dump(self.best_policy, f)

#     def forward(self, x):
#         return self.base(x)