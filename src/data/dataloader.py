import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
from torch.utils.data import DataLoader, Subset

def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_cifar10_loaders(batch_size=128, num_workers=4, seed=42, split_for_reward=False):
    """
    split_for_reward: 
        - True (用于 Stage 1): 返回 (train_loader_45k, reward_loader_5k, test_loader)
        - False (用于 Stage 2): 返回 (train_loader_50k, test_loader)
    """
    
    # 固定随机性
    g = torch.Generator()
    g.manual_seed(seed)

    # CIFAR-10 标准均值和方差
    # mean = (0.4914, 0.4822, 0.4465)
    # std = (0.2023, 0.1994, 0.2010)
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # Test/Reward 都不需要 Augmentation
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # 加载完整训练集
    full_train_aug = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    full_train_clean = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_test
    )
    
    # 加载测试集
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, worker_init_fn=_seed_worker, generator=g
    )

    # --- 核心切分逻辑 ---
    if split_for_reward:
        # 生成随机索引
        num_train = len(full_train_aug) # 50000
        indices = torch.randperm(num_train, generator=g).tolist()
        
        train_size = 45000
        train_idx = indices[:train_size]
        reward_idx = indices[train_size:]
        
        # 创建 Subset
        # 训练集用带增强的 Dataset
        train_subset = Subset(full_train_aug, train_idx)
        # Reward集用无增强的 Dataset (Clean)
        reward_subset = Subset(full_train_clean, reward_idx)
        
        trainloader = DataLoader(
            train_subset, batch_size=batch_size, shuffle=True, 
            num_workers=num_workers, worker_init_fn=_seed_worker, generator=g
        )
        rewardloader = DataLoader(
            reward_subset, batch_size=batch_size, shuffle=False, 
            num_workers=num_workers, worker_init_fn=_seed_worker, generator=g
        )
        return trainloader, rewardloader, testloader

    else:
        # 不切分时，直接用全量带增强数据
        trainloader = DataLoader(
            full_train_aug, batch_size=batch_size, shuffle=True, 
            num_workers=num_workers, worker_init_fn=_seed_worker, generator=g
        )
        return trainloader, testloader