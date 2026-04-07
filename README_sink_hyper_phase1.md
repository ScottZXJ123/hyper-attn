# Sink-Aware HyperAttention (Phase 1)

This phase validates sink-aware decomposition for **single-layer offline attention** using cached Q/K/V tensors.

## Scope
- Data source: cached tensors from Pythia-160M, layer 8, head 3
- Sequence lengths: 512, 1024, 2048
- Methods compared:
  1. Exact softmax attention
  2. Vanilla HyperAttention
  3. Sink-Aware HyperAttention (`sink_size = 32`)

## What is implemented
- `models/attention/sink_hyper_attn.py`
  - `SinkAwareHyperAttention`
  - Input shape: `[batch, heads, seq_len, dim]`
  - Exact prefix term over sink keys/values (`:sink_size`)
  - HyperAttention tail term over `[sink_size:]` with `causal=True, return_lse=True`
  - Proper log-space merge via `add_self_attentions`

- `benchmark_sink_single_attention.py`
  - Loads cached `.pt` tensors
  - Runs exact / hyper / sink-hyper
  - Measures:
    - Relative Frobenius error of output
    - `lse_mae = mean(abs(lse_hat - lse_exact))`
    - Forward runtime (ms)
  - Writes:
    - `results/sink_hyper_phase1/results.csv`
    - `results/sink_hyper_phase1/summary.json`

## Fixed HyperAttention parameters
- `lsh_num_projs = 7`
- `block_size = 256`
- `sample_size = 256`
- `min_seq_len = 0`
- `causal = True`

## Reproducibility and timing policy
- Models run in eval mode and are executed under `torch.inference_mode()`.
- Correctness pass is seed-fixed per sample (`seed + sample_index` or `seed + sample_id` when numeric).
- Runtime is measured separately from correctness using median timing with:
  - `warmup = 2`
  - `repeats = 5`
- Input tensors are explicitly cast by config (`dtype`, default `float32`).

## Run
```bash
python benchmark_sink_single_attention.py --config configs/sink_hyper_phase1.yaml
```

### Suggested execution order
1. **Sanity check**: `seq_lens=[512]`, `max_samples=2`, `dtype=float32`
2. **Full correctness**: `seq_lens=[512,1024,2048]`, all samples, `dtype=float32`
3. **Runtime-focused pass**: keep warmup/repeats and switch dtype as needed (for example, `bfloat16` on GPU)

## Data format
Accepted cached input formats in `data_dir`:
1. `q.pt`, `k.pt`, `v.pt`
   - each tensor can be rank-4 (`[B,H,N,D]`) or rank-5 (`[S,B,H,N,D]` for S samples)
2. `*.pt` files where each file is a dict with keys: `q`, `k`, `v`

Only samples with sequence length in `[512, 1024, 2048]` are evaluated.

---

## End-to-end guide (Vast.ai GPU instance)

下面给你一套从**租机器 → 配环境 → 跑实验 → 看结果**的完整流程（针对你截图这种 L40S + CUDA 12.x 的机器）。

### 0) 租机建议
- GPU: `L40S` (48GB) 足够本阶段单层离线实验
- 系统: Ubuntu 22.04
- Python: 3.10 或 3.11
- 磁盘: 建议 >= 50GB（代码 + 数据 + 结果）
- 网络: 保证可以访问 GitHub / PyPI

> 说明：本项目 Phase 1 只做离线单层 attention，不需要 patch 全模型。

### 1) 登录机器并准备系统依赖
```bash
sudo apt-get update
sudo apt-get install -y git wget tmux htop
```

可选：建议先开一个 `tmux`，避免 SSH 断连中断实验：
```bash
tmux new -s sink_hyper
```

### 2) 拉取代码
```bash
git clone <your-fork-or-repo-url> hyper-attn
cd hyper-attn
git checkout feat/sink-hyper-phase1
```

### 3) 创建 Python 环境
推荐 `venv`：
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

### 4) 安装核心依赖
> 你机器显示 Max CUDA 12.7，通常可直接使用 PyTorch 官方 cu124 wheel（由驱动前向兼容）。

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install pyyaml einops tqdm
```

HyperAttention 代码依赖 triton。若你希望与仓库原说明一致可尝试：
```bash
pip install triton==2.0.0.dev20221202 --no-deps
```

### 5) 环境自检
```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY
```

若 `cuda available: False`，优先检查：
- 是否租的是 GPU 机型
- 宿主机驱动是否正常
- 是否安装了 CPU 版 torch（需重装 cu124 版）

### 6) 准备离线缓存 Q/K/V 数据
默认配置读取目录：
```text
data/pythia160m_layer8_head3_qkv
```

你可以用两种格式之一：

1. 三文件格式
```text
data/pythia160m_layer8_head3_qkv/
  q.pt
  k.pt
  v.pt
```
- 每个文件 tensor 形状可为：
  - `[B,H,N,D]`（单样本）
  - `[S,B,H,N,D]`（多样本）

2. 多样本文件格式
```text
data/pythia160m_layer8_head3_qkv/
  sample_0.pt
  sample_1.pt
  ...
```
- 每个 `.pt` 是 dict，至少包含 `q/k/v`

并确保 tensor shape 为 `[batch, heads, seq_len, dim]`，且 `seq_len ∈ {512, 1024, 2048}`。

### 7) 按阶段运行实验（推荐流程）

#### Step 1: 小规模 sanity check
目标：先验证流程和趋势是否正常（不直接全量跑）。

复制一份配置：
```bash
cp configs/sink_hyper_phase1.yaml configs/sink_hyper_phase1_sanity.yaml
```

编辑 `configs/sink_hyper_phase1_sanity.yaml`，建议：
```yaml
seq_lens: [512]
dtype: float32
max_samples: 2
seed: 1234
warmup: 2
repeats: 5
```

运行：
```bash
python benchmark_sink_single_attention.py --config configs/sink_hyper_phase1_sanity.yaml
```

重点看：
- 三种方法是否都可运行
- `sink_hyper` 的 `rel_frob_error` 是否低于 `hyper`
- `sink_hyper` 的 `lse_mae` 是否更低或至少可比

#### Step 2: Full Phase 1 correctness
在 sanity check 通过后，运行全量 correctness：
```bash
python benchmark_sink_single_attention.py --config configs/sink_hyper_phase1.yaml
```

建议保持：
- `dtype: float32`
- `seq_lens: [512, 1024, 2048]`
- 全部样本

#### Step 3: Runtime benchmark
确认 correctness 趋势后再做 timing：
- 保持 warmup/repeats（当前默认 `2/5`）
- GPU 环境下可改 `dtype: bfloat16` 做吞吐评估

例如创建运行时配置：
```bash
cp configs/sink_hyper_phase1.yaml configs/sink_hyper_phase1_runtime.yaml
# 把 dtype 改成 bfloat16，然后运行
python benchmark_sink_single_attention.py --config configs/sink_hyper_phase1_runtime.yaml
```

### 8) 输出结果位置
默认输出目录：
```text
results/sink_hyper_phase1/
  results.csv
  summary.json
```

### 9) 快速查看结果
```bash
python - <<'PY'
import pandas as pd
df = pd.read_csv("results/sink_hyper_phase1/results.csv")
print(df.groupby(["seq_len","method"])[["rel_frob_error","lse_mae","runtime_ms"]].mean())
PY
```

### 10) 常见问题排查
- **`ModuleNotFoundError: torch`**  
  说明 PyTorch 未安装到当前环境；确认已 `source .venv/bin/activate` 后重新安装。

- **运行很慢 / OOM**  
  先用 sanity 配置（`seq_len=512`, `max_samples=2`）检查，再逐步放开。

- **结果不稳定**  
  本脚本已做每样本 seed 固定；请确认没有改动 `seed` 逻辑，并记录使用的配置文件。
