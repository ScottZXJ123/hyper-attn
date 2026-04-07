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

## Run
```bash
python benchmark_sink_single_attention.py --config configs/sink_hyper_phase1.yaml
```

## Data format
Accepted cached input formats in `data_dir`:
1. `q.pt`, `k.pt`, `v.pt`
   - each tensor can be rank-4 (`[B,H,N,D]`) or rank-5 (`[S,B,H,N,D]` for S samples)
2. `*.pt` files where each file is a dict with keys: `q`, `k`, `v`

Only samples with sequence length in `[512, 1024, 2048]` are evaluated.
