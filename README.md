## Reproducing *Jointly Training and Pruning CNNs via Learnable Agent Guidance and Alignment*
- baseline acc：93.50%
- pruned acc：93.56%
### quick start
- nohup run.sh param_config > main_process.log 2>&1 &
- nohup python -u -m src.run_train --config configs/xxxx.yaml >train.log 2>&1 &
- nohup python -u -m src.finetune --config configs/xxxx.yaml --checkpoint_dir checkpoints/param_config >finetune.log 2>&1 &
