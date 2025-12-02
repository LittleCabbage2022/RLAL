import os
import torch

class CheckpointManager:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def save(self, epoch, student_model, agent, env_state, config, best=False):
        """
        student_model: 剪枝目标 CNN（如 MobileNet）
        agent: SAC agent（包含 policy_net / q_net）
        env_state: dict，包含当前环境状态（已剪枝层、flops、acc、动作轨迹等）
        config: 训练配置
        best: 是否保存最佳模型
        """

        ckpt = {
            "epoch": epoch,
            "student_model": student_model.state_dict(),
            "policy_net": agent.policy_net.state_dict(),
            "critic1_net": agent.critic1_net.state_dict(),
            "critic2_net": agent.critic2_net.state_dict(),
            "critic1_target": agent.critic1_target.state_dict(),
            "critic2_target": agent.critic2_target.state_dict(),
            "log_alpha": agent.log_alpha,
            "env_state": env_state,
            "config": config,
        }

        filename = "best.pth" if best else f"epoch_{epoch}.pth"
        path = os.path.join(self.save_dir, filename)
        torch.save(ckpt, path)

        print(f"[Checkpoint] Saved → {path}")

    def load(self, path, student_model, agent):
        """
        恢复 checkpoint: student CNN + SAC agent
        返回: (epoch, env_state, config)
        """

        print(f"[Checkpoint] Loading from {path} ...")
        ckpt = torch.load(path, map_location="cpu")

        # --- student model ---
        student_model.load_state_dict(ckpt["student_model"])

        # --- SAC components ---
        agent.policy_net.load_state_dict(ckpt["policy_net"])
        agent.critic1_net.load_state_dict(ckpt["critic1_net"])
        agent.critic2_net.load_state_dict(ckpt["critic2_net"])
        agent.critic1_target.load_state_dict(ckpt["critic1_target"])
        agent.critic2_target.load_state_dict(ckpt["critic2_target"])
        agent.log_alpha = ckpt["log_alpha"]

        print("[Checkpoint] Model & Agent restored.")

        # --- return extra info ---
        return ckpt["epoch"], ckpt["env_state"], ckpt["config"]

