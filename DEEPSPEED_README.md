# DeepSpeed配置说明 - GRPO训练

## 📁 配置文件列表

我为你的项目创建了3个DeepSpeed配置文件，根据不同的需求选择：

### 1. `deepspeed_grpo_config.json` ⭐⭐⭐⭐⭐ **推荐**

**适用场景**：平衡内存和速度
- ✅ ZeRO Stage 2
- ✅ Optimizer CPU Offload
- ✅ 最佳性价比

**内存节省**：25-35 GB per GPU
**训练速度**：基准的 85-90%
**推荐指数**：⭐⭐⭐⭐⭐

---

### 2. `deepspeed_grpo_aggressive.json` ⭐⭐⭐⭐

**适用场景**：极致节省内存（OOM严重时）
- ✅ ZeRO Stage 2
- ✅ Optimizer CPU Offload
- ✅ **Parameter CPU Offload**（额外）
- ✅ 更小的bucket size

**内存节省**：35-45 GB per GPU
**训练速度**：基准的 70-80%
**推荐指数**：⭐⭐⭐⭐

---

### 3. `deepspeed_grpo_fast.json` ⭐⭐⭐

**适用场景**：速度优先（内存勉强够用时）
- ✅ ZeRO Stage 2
- ❌ 无CPU Offload
- ✅ 更大的bucket size

**内存节省**：15-20 GB per GPU
**训练速度**：基准的 95-98%
**推荐指数**：⭐⭐⭐

---

## 🚀 使用方法

### 步骤1：安装DeepSpeed

```bash
pip install deepspeed
```

### 步骤2：选择配置文件

根据你的内存情况选择：

```bash
# 如果当前OOM严重（推荐）
CONFIG_FILE="deepspeed_grpo_config.json"

# 如果还是OOM
CONFIG_FILE="deepspeed_grpo_aggressive.json"

# 如果内存够用，追求速度
CONFIG_FILE="deepspeed_grpo_fast.json"
```

### 步骤3：运行训练

在你的运行命令中添加 `--deepspeed` 参数：

```bash
cd /root/work/filestorage/gaoshan/projects/OpenThinkIMG
export VLLM_HOST_IP=$(hostname -I | awk '{print $1}')
export VLLM_RPC_PORT=29550
export PYTHONPATH=$(pwd):$PYTHONPATH
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_TIMEOUT=600  
export TRANSFORMERS_NO_TF=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

LOG_FILE="rl_qwen25_$(date +%Y%m%d_%H%M%S).log"

WANDB_DISABLED=true CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --nproc_per_node=3 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=29503 \
    r1_v/open_r1/tool_grpo.py --use_vllm True \
    --output_dir ./models/RL_1112 \
    --model_name_or_path /root/work/filestorage/gaoshan/models/Qwen2_5-VL-7B-Instruct \
    --dataset_name /root/work/filestorage/gaoshan/dataset/OpenThinkIMG/OpenThinkIMG-Chart-RL-14501/openthinkIMG_chart_RL.json \
    --max_prompt_length 4096 \
    --max_completion_length 2048 \
    --temperature 1.0 \
    --seed 42 \
    --learning_rate 1e-6 \
    --num_generations 3 \
    --lr_scheduler_type constant \
    --vllm_gpu_memory_utilization 0.5 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --logging_steps 1 \
    --bf16 true \
    --report_to none \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --deepspeed ./deepspeed_grpo_config.json \    # ← 添加这行！
    --max_pixels 200000 \
    --num_train_epochs 1 \
    --run_name qwen2vl_rl_v1112_1529 \
    --save_steps 100 \
    --save_only_model true \
    --controller_addr http://localhost:20001 \
    --vllm_device cuda:3 \
    --use_tool true \
2>&1 | while IFS= read -r line; do
    printf '%s\n' "$line"
    printf '%s\n' "$line" >> "$LOG_FILE"
done
```

---

## 📊 配置详解

### ZeRO Stage 2 的工作原理

```
正常训练（每张卡）：
├─ 模型参数: 14 GB (完整)
├─ Optimizer: 28 GB (完整)
└─ Gradients: 14 GB (完整)
总计: 56 GB

ZeRO Stage 2（每张卡）：
├─ 模型参数: 14 GB (完整) ← 保持完整，VLLM可用
├─ Optimizer: 28 GB ÷ 3 = 9.3 GB ← 分片！
└─ Gradients: 14 GB ÷ 3 = 4.7 GB ← 分片！
总计: 28 GB (节省28 GB)
```

### CPU Offload 的工作原理

```
Optimizer CPU Offload:
GPU: 模型参数 + Gradients (18.7 GB)
CPU: Optimizer states (28 GB) ← 在CPU内存中

训练时：
1. Forward/Backward在GPU
2. Optimizer步骤在CPU
3. 参数更新传回GPU
```

---

## ⚠️ 注意事项

### 1. **不要同时使用 8bit optimizer**

```bash
# ❌ 错误：
--deepspeed ./deepspeed_grpo_config.json \
--optim adamw_bnb_8bit \

# ✅ 正确：
--deepspeed ./deepspeed_grpo_config.json \
# 不需要指定optim，DeepSpeed会管理
```

### 2. **确保VLLM使用独立GPU**

```bash
# ✅ 正确配置：
CUDA_VISIBLE_DEVICES=0,1,2,3 \  # 4张卡
torchrun --nproc_per_node=3 \    # 3张训练
--vllm_device cuda:3 \           # 第4张用VLLM
```

### 3. **CPU内存要求**

使用CPU Offload时，确保CPU内存充足：
- **推荐配置**：至少 100GB CPU RAM
- **激进配置**：至少 150GB CPU RAM

检查CPU内存：
```bash
free -h
```

### 4. **兼容性验证**

首次运行时，检查DeepSpeed是否正确初始化：

```bash
# 日志中应该看到：
[INFO] DeepSpeed info: version=...
[INFO] Using ZeRO stage 2
[INFO] Offloading optimizer to CPU
```

---

## 🔧 高级调优

### 如果还是OOM

1. **进一步减小bucket size**（在配置中修改）：
```json
"allgather_bucket_size": 1e8,  // 从2e8改为1e8
"reduce_bucket_size": 1e8,
```

2. **同时减小训练参数**：
```bash
--max_completion_length 1536 \
--max_pixels 150000 \
```

### 如果速度太慢

1. **使用fast配置**：
```bash
--deepspeed ./deepspeed_grpo_fast.json \
```

2. **调整bucket size**（在配置中）：
```json
"allgather_bucket_size": 1e9,  // 增大
"reduce_bucket_size": 1e9,
```

---

## 📈 预期效果

### 使用推荐配置后

| 指标 | 之前 | 之后 | 改善 |
|------|------|------|------|
| **GPU内存** | 76 GB | 40-45 GB | ✅ -30 GB |
| **训练速度** | 100% | 85-90% | ⚠️ -10% |
| **OOM风险** | 高 | 低 | ✅ |

---

## 🆘 故障排除

### 错误1：`ImportError: No module named 'deepspeed'`

```bash
pip install deepspeed
```

### 错误2：`NCCL timeout`

增加超时时间：
```bash
export TORCH_NCCL_TIMEOUT=1800  # 30分钟
```

### 错误3：CPU内存不足

使用非offload配置：
```bash
--deepspeed ./deepspeed_grpo_fast.json \
```

### 错误4：训练变慢太多

使用less aggressive配置：
```bash
--deepspeed ./deepspeed_grpo_fast.json \
```

---

## 💡 最佳实践

1. **首次训练**：使用 `deepspeed_grpo_config.json`
2. **监控GPU内存**：`watch -n 1 nvidia-smi`
3. **如果OOM**：切换到 `deepspeed_grpo_aggressive.json`
4. **如果速度慢**：切换到 `deepspeed_grpo_fast.json`
5. **保存checkpoint**：DeepSpeed会自动处理分片的checkpoint

---

## 📞 需要帮助？

如果遇到问题：
1. 查看日志中的DeepSpeed初始化信息
2. 运行 `nvidia-smi` 监控内存
3. 检查CPU内存使用 `free -h`
4. 尝试不同的配置文件

---

**祝训练成功！** 🎉
