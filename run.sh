#!/bin/bash
# nohup run.sh config > main_process.log 2>&1 &

# 从命令行参数获取实验名
if [ $# -eq 0 ]; then
    echo "Usage: $0 <EXP_NAME>"
    echo "Example: $0 param0_8-4-10000"
    exit 1
fi

EXP_NAME="$1"

# ================= 配置区域 =================
CONFIG_FILE="configs/${EXP_NAME}.yaml"
CHECKPOINT_DIR="checkpoints/${EXP_NAME}"

# 日志文件名
LOG_TRAIN="train${EXP_NAME}.log"
LOG_FINETUNE="finetune${EXP_NAME}.log"
# ===========================================

echo "=========================================="
echo "Task: $EXP_NAME"
echo "Start Time: $(date)"
echo "=========================================="

# ------------------------------------------
# 第一步：运行 Search (src.run_train)
# ------------------------------------------
echo ">> [Step 1/2] Starting Joint Training..."
# 注意：这里不用 nohup，因为整个脚本会被 nohup 调用
python -u -m src.run_train \
    --config $CONFIG_FILE \
    > $LOG_TRAIN 2>&1

# 检查上一步是否成功 (返回码为 0 表示成功)
if [ $? -ne 0 ]; then
    echo "❌ Error: Step 1 (Search) failed. Check $LOG_TRAIN for details."
    exit 1
fi
echo "✅ Step 1 Finished."

# ------------------------------------------
# 第二步：运行 Finetune (src.finetune)
# ------------------------------------------
echo ">> [Step 2/2] Starting Finetune (No KD)..."
python -u -m src.finetune \
    --config $CONFIG_FILE \
    --checkpoint_dir $CHECKPOINT_DIR \
    > $LOG_FINETUNE 2>&1

if [ $? -ne 0 ]; then
    echo "❌ Error: Step 2 (Finetune) failed. Check $LOG_FINETUNE for details."
    exit 1
fi

echo "=========================================="
echo "🎉 All Done! End Time: $(date)"
echo "Final Log: $LOG_FINETUNE"
echo "=========================================="
