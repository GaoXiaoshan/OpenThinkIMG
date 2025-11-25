#!/bin/bash
# DeepSpeed配置验证脚本

echo "🔍 检查DeepSpeed环境..."
echo ""

# 1. 检查DeepSpeed是否安装
echo "1️⃣ 检查DeepSpeed安装:"
if python -c "import deepspeed" 2>/dev/null; then
    VERSION=$(python -c "import deepspeed; print(deepspeed.__version__)")
    echo "   ✅ DeepSpeed已安装 (版本: $VERSION)"
else
    echo "   ❌ DeepSpeed未安装"
    echo "   请运行: pip install deepspeed"
    exit 1
fi
echo ""

# 2. 检查配置文件
echo "2️⃣ 检查配置文件:"
for config in deepspeed_grpo_config.json deepspeed_grpo_aggressive.json deepspeed_grpo_fast.json; do
    if [ -f "$config" ]; then
        echo "   ✅ $config"
    else
        echo "   ❌ $config 未找到"
    fi
done
echo ""

# 3. 检查GPU
echo "3️⃣ 检查GPU状态:"
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    echo "   ✅ 检测到 $GPU_COUNT 张GPU"
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader | \
    while IFS=, read -r idx name total used free; do
        echo "      GPU $idx: $name"
        echo "         总内存: $total"
        echo "         已用: $used"
        echo "         可用: $free"
    done
else
    echo "   ❌ nvidia-smi 未找到"
fi
echo ""

# 4. 检查CPU内存
echo "4️⃣ 检查CPU内存:"
TOTAL_MEM=$(free -h | grep Mem | awk '{print $2}')
AVAIL_MEM=$(free -h | grep Mem | awk '{print $7}')
echo "   总内存: $TOTAL_MEM"
echo "   可用内存: $AVAIL_MEM"
AVAIL_GB=$(free -g | grep Mem | awk '{print $7}')
if [ "$AVAIL_GB" -lt 50 ]; then
    echo "   ⚠️  可用内存较少，建议使用 deepspeed_grpo_fast.json"
else
    echo "   ✅ 内存充足，可以使用CPU Offload"
fi
echo ""

# 5. 检查NCCL
echo "5️⃣ 检查NCCL配置:"
if [ -z "$TORCH_NCCL_TIMEOUT" ]; then
    echo "   ⚠️  TORCH_NCCL_TIMEOUT 未设置"
    echo "   建议添加: export TORCH_NCCL_TIMEOUT=600"
else
    echo "   ✅ TORCH_NCCL_TIMEOUT=$TORCH_NCCL_TIMEOUT"
fi
echo ""

# 6. 推荐配置
echo "📋 推荐配置建议:"
echo ""
AVAIL_GPU_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
if [ "$AVAIL_GPU_MEM" -gt 50000 ]; then
    echo "   🟢 GPU内存充足 (>50GB空闲)"
    echo "   推荐: deepspeed_grpo_fast.json"
elif [ "$AVAIL_GPU_MEM" -gt 30000 ]; then
    echo "   🟡 GPU内存中等 (30-50GB空闲)"
    echo "   推荐: deepspeed_grpo_config.json ⭐"
else
    echo "   🔴 GPU内存紧张 (<30GB空闲)"
    echo "   推荐: deepspeed_grpo_aggressive.json"
fi
echo ""

echo "✅ 检查完成！"
echo ""
echo "📝 快速启动命令:"
echo "   torchrun --nproc_per_node=3 \\"
echo "       r1_v/open_r1/tool_grpo.py \\"
echo "       --deepspeed ./deepspeed_grpo_config.json \\"
echo "       [其他参数...]"
