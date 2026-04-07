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

This is a complete workflow from **renting a machine -> setting up environment -> running experiments -> checking outputs** (targeting an L40S + CUDA 12.x setup like your screenshot).

### 0) Recommended instance
- GPU: `L40S` (48GB) is sufficient for this phase
- OS: Ubuntu 22.04
- Python: 3.10 or 3.11
- Disk: >= 50GB recommended (code + data + outputs)
- Network: allow access to GitHub / PyPI

> Phase 1 is **offline single-layer attention only**. No full-LLM patching is required.

### 1) Prepare system packages
```bash
sudo apt-get update
sudo apt-get install -y git wget tmux htop
```

Optional but recommended: start a `tmux` session to avoid interruption if SSH disconnects.
```bash
tmux new -s sink_hyper
```

### 2) Clone repository
```bash
git clone <your-fork-or-repo-url> hyper-attn
cd hyper-attn
git checkout feat/sink-hyper-phase1
```

### 3) Create Python environment
Using `venv`:
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

### 4) Install dependencies
Your machine shows CUDA 12.7 support. In practice, PyTorch `cu124` wheels usually work due to driver forward compatibility.

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install pyyaml einops tqdm
```

HyperAttention also depends on triton. To follow this repository's original recommendation:
```bash
pip install triton==2.0.0.dev20221202 --no-deps
```

### 5) Environment sanity check
```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY
```

If `cuda available: False`, check:
- you rented a GPU instance (not CPU-only),
- host driver is healthy,
- you did not install CPU-only torch by mistake.

### 6) Prepare cached offline Q/K/V tensors
Default data path:
```text
data/pythia160m_layer8_head3_qkv
```

Two supported formats:

1. Three-file format
```text
data/pythia160m_layer8_head3_qkv/
  q.pt
  k.pt
  v.pt
```
- each tensor can be:
  - `[B,H,N,D]` (single sample)
  - `[S,B,H,N,D]` (multiple samples)

2. Per-sample file format
```text
data/pythia160m_layer8_head3_qkv/
  sample_0.pt
  sample_1.pt
  ...
```
- each `.pt` is a dict containing at least `q/k/v`

Tensors must be in `[batch, heads, seq_len, dim]`, with `seq_len ∈ {512, 1024, 2048}`.

### 6.1) Important note about Pythia model download
This benchmark **does not automatically download Pythia-160M** and does not run full model inference.
It consumes **already-exported cached Q/K/V tensors** only.

If you do not yet have cached tensors, generate them in a separate preprocessing step (outside this phase) using Pythia-160M and save them into `data/pythia160m_layer8_head3_qkv`.

### 7) Run experiments in stages (recommended)

#### Step 1: Small sanity check
Goal: validate pipeline and trend first (before full run).

Copy config:
```bash
cp configs/sink_hyper_phase1.yaml configs/sink_hyper_phase1_sanity.yaml
```

Edit `configs/sink_hyper_phase1_sanity.yaml`:
```yaml
seq_lens: [512]
dtype: float32
max_samples: 2
seed: 1234
warmup: 2
repeats: 5
```

Run:
```bash
python benchmark_sink_single_attention.py --config configs/sink_hyper_phase1_sanity.yaml
```

Check:
- all three methods run correctly,
- `sink_hyper` has lower `rel_frob_error` than `hyper`,
- `sink_hyper` has lower or comparable `lse_mae`.

#### Step 2: Full Phase 1 correctness
After sanity passes, run full correctness:
```bash
python benchmark_sink_single_attention.py --config configs/sink_hyper_phase1.yaml
```

Recommended settings:
- `dtype: float32`
- `seq_lens: [512, 1024, 2048]`
- all prepared samples

#### Step 3: Runtime benchmark
Run timing only after correctness trend is confirmed:
- keep warmup/repeats (`2/5` by default),
- optionally switch `dtype: bfloat16` on GPU for throughput-focused timing.

Example runtime config flow:
```bash
cp configs/sink_hyper_phase1.yaml configs/sink_hyper_phase1_runtime.yaml
# change dtype to bfloat16, then run:
python benchmark_sink_single_attention.py --config configs/sink_hyper_phase1_runtime.yaml
```

### 8) Output paths
Default output directory:
```text
results/sink_hyper_phase1/
  results.csv
  summary.json
```

### 9) Quick result inspection
```bash
python - <<'PY'
import pandas as pd
df = pd.read_csv("results/sink_hyper_phase1/results.csv")
print(df.groupby(["seq_len","method"])[["rel_frob_error","lse_mae","runtime_ms"]].mean())
PY
```

### 10) Troubleshooting
- **`ModuleNotFoundError: torch`**  
  PyTorch is not installed in the active environment. Re-activate `.venv` and reinstall.

- **Slow run / OOM**  
  Start from sanity config (`seq_len=512`, `max_samples=2`) and scale up gradually.

- **Unstable metrics**  
  Per-sample seeds are fixed in this benchmark. Verify seed logic and keep config files versioned.
