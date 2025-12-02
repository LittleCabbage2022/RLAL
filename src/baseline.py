# src/finetune.py
# nohup python -u -m src.baseline --config configs/full.yaml --checkpoint_dir checkpoints/baseline_test >baseline.log 2>&1

import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

from .data.dataloader import get_cifar10_loaders
from .models.model_wrapper import ModelWrapper

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def finetune(config_path, checkpoint_dir):
    # 1. 加载配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device(f"cuda:{config.get('gpu_id', 0)}" if torch.cuda.is_available() else "cpu")
    set_seed(config.get('seed', 42))
    
    print(f"=== Starting Finetuning (Stage 2) ===")
    print(f"Config: {config_path}")
    print(f"Checkpoint Dir: {checkpoint_dir}")

    # 2. 准备数据 (注意：必须包含 Normalize)
    train_loader, val_loader = get_cifar10_loaders(
        batch_size=config.get('batch_size', 128), 
        seed=config.get('seed', 42)
    )

    # 3. 初始化模型
    input_shape = tuple(config.get('input_shape', [1, 3, 32, 32]))
    model = ModelWrapper(device, input_shape=input_shape)
    
    # 6. 设置优化器 (Same settings for base model)
    # 论文 Appendix B: SGD, Momentum 0.9, WD 1e-4, LR 0.1
    # 这里的 Epoch 数通常建议 160 或 200
    finetune_epochs = 200 
    lr = 0.1
    weight_decay = 0.0001 # 1e-4
    
    optimizer = torch.optim.SGD(
        model.base.parameters(), 
        lr=lr, 
        momentum=0.9, 
        weight_decay=weight_decay
    )
    
    # 学习率调度: CIFAR-10 常用 milestones [100, 150]
    milestones = [int(finetune_epochs * 0.5), int(finetune_epochs * 0.75)]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    print(f">> Optimizer reset: LR={lr}, Epochs={finetune_epochs}, WD={weight_decay}")

    # 7. 训练循环
    best_acc = 0.0
    
    for epoch in range(finetune_epochs):
        # --- Train ---
        model.base.train()
        train_loss = 0.0
        
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            out = model.forward(x)
            loss = F.cross_entropy(out, y)
            loss.backward()
            
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()
        
        # --- Eval ---
        val_acc = evaluate(model, val_loader, device)
        current_lr = scheduler.get_last_lr()[0]
        
        print(f"Epoch {epoch+1}/{finetune_epochs} | Loss: {train_loss/len(train_loader):.4f} | Val Acc: {val_acc:.4f} | LR: {current_lr:.5f}")

        # --- Save Best ---
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.base.state_dict(), os.path.join(checkpoint_dir, 'finetuned_best.pth'))
            print(f"   >> New Best Saved! ({best_acc:.4f})")

    print(f"=== Finetuning Complete. Best Acc: {best_acc:.4f} ===")

def evaluate(model_wrapper, val_loader, device):
    model_wrapper.base.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model_wrapper.forward(x)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/full.yaml', help='path to config file')
    parser.add_argument('--checkpoint_dir', type=str, required=True, help='path to folder containing best_policy.json and student_model.pth')
    args = parser.parse_args()
    
    finetune(args.config, args.checkpoint_dir)