import argparse
import csv
import json
import time
from pathlib import Path

import torch
import yaml

from models.attention.hyper_attn import HyperAttention
from models.attention.sink_hyper_attn import SinkAwareHyperAttention
from models.attention.utils import exact_attention


def parse_args():
    parser = argparse.ArgumentParser(description="Phase-1 sink-aware HyperAttention single-layer benchmark")
    parser.add_argument("--config", type=str, default="configs/sink_hyper_phase1.yaml")
    return parser.parse_args()


def _load_tensor(path: Path):
    x = torch.load(path, map_location="cpu")
    if isinstance(x, dict) and "tensor" in x:
        return x["tensor"]
    return x


def load_samples(data_dir: Path):
    q_file = data_dir / "q.pt"
    k_file = data_dir / "k.pt"
    v_file = data_dir / "v.pt"

    samples = []
    if q_file.exists() and k_file.exists() and v_file.exists():
        q_all = _load_tensor(q_file)
        k_all = _load_tensor(k_file)
        v_all = _load_tensor(v_file)

        if q_all.ndim == 4:
            samples.append({"sample_id": "0", "q": q_all, "k": k_all, "v": v_all})
        elif q_all.ndim == 5:
            for i in range(q_all.shape[0]):
                samples.append({"sample_id": str(i), "q": q_all[i], "k": k_all[i], "v": v_all[i]})
        else:
            raise ValueError(f"Unsupported q.pt rank: {q_all.ndim}")
        return samples

    for pt_file in sorted(data_dir.glob("*.pt")):
        payload = torch.load(pt_file, map_location="cpu")
        if not (isinstance(payload, dict) and {"q", "k", "v"}.issubset(payload.keys())):
            continue
        samples.append(
            {
                "sample_id": pt_file.stem,
                "q": payload["q"],
                "k": payload["k"],
                "v": payload["v"],
            }
        )
    if not samples:
        raise FileNotFoundError(
            f"No samples found in {data_dir}. Expected q.pt/k.pt/v.pt or *.pt dictionaries with q/k/v keys."
        )
    return samples


def ensure_bhnd(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 4:
        raise ValueError(f"Expected rank-4 tensor [batch, heads, seq_len, dim], got {x.shape}")
    return x


def time_forward(fn):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt_ms = (time.perf_counter() - t0) * 1000.0
    return out, dt_ms


def rel_fro_error(pred, exact):
    return (pred - exact).norm().div(exact.norm().clamp_min(1e-12)).item()


def aggregate(rows):
    summary = {}
    for seq_len in sorted({r["seq_len"] for r in rows}):
        summary[str(seq_len)] = {}
        for method in sorted({r["method"] for r in rows}):
            vals = [r for r in rows if r["seq_len"] == seq_len and r["method"] == method]
            if not vals:
                continue
            def m(k):
                t = torch.tensor([v[k] for v in vals], dtype=torch.float64)
                return {"mean": t.mean().item(), "std": t.std(unbiased=False).item(), "count": int(t.numel())}

            summary[str(seq_len)][method] = {
                "rel_frob_error": m("rel_frob_error"),
                "lse_mae": m("lse_mae"),
                "runtime_ms": m("runtime_ms"),
            }
    return summary


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_dir = Path(cfg["data_dir"])
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    seq_lens = set(cfg["seq_lens"])
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    hyper = HyperAttention(
        input_dim=cfg["input_dim"],
        lsh_num_projs=7,
        block_size=256,
        sample_size=256,
        min_seq_len=0,
        cuda=(device.type == "cuda"),
    ).to(device)

    sink_hyper = SinkAwareHyperAttention(
        input_dim=cfg["input_dim"],
        sink_size=32,
        lsh_num_projs=7,
        block_size=256,
        sample_size=256,
        min_seq_len=0,
        cuda=(device.type == "cuda"),
    ).to(device)

    rows = []
    for sample in load_samples(data_dir):
        q = ensure_bhnd(sample["q"]).to(device)
        k = ensure_bhnd(sample["k"]).to(device)
        v = ensure_bhnd(sample["v"]).to(device)

        seq_len = int(q.shape[2])
        if seq_len not in seq_lens:
            continue

        if k.shape[2] != seq_len or v.shape[2] != seq_len:
            raise ValueError(f"q/k/v seq_len mismatch for sample {sample['sample_id']}: {q.shape}, {k.shape}, {v.shape}")

        (o_exact, lse_exact), t_exact = time_forward(lambda: exact_attention(q, k, v, softmax_scale=q.shape[-1] ** -0.5, causal=True))
        rows.append(
            {
                "sample_id": sample["sample_id"],
                "seq_len": seq_len,
                "method": "exact",
                "rel_frob_error": 0.0,
                "lse_mae": 0.0,
                "runtime_ms": t_exact,
            }
        )

        (o_hyper, lse_hyper), t_hyper = time_forward(lambda: hyper(q, k, v, causal=True, return_lse=True))
        rows.append(
            {
                "sample_id": sample["sample_id"],
                "seq_len": seq_len,
                "method": "hyper",
                "rel_frob_error": rel_fro_error(o_hyper, o_exact),
                "lse_mae": (lse_hyper - lse_exact).abs().mean().item(),
                "runtime_ms": t_hyper,
            }
        )

        (o_sink, lse_sink), t_sink = time_forward(lambda: sink_hyper(q, k, v, causal=True, return_lse=True))
        rows.append(
            {
                "sample_id": sample["sample_id"],
                "seq_len": seq_len,
                "method": "sink_hyper",
                "rel_frob_error": rel_fro_error(o_sink, o_exact),
                "lse_mae": (lse_sink - lse_exact).abs().mean().item(),
                "runtime_ms": t_sink,
            }
        )

    csv_path = output_dir / "results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "seq_len", "method", "rel_frob_error", "lse_mae", "runtime_ms"])
        writer.writeheader()
        writer.writerows(rows)

    summary = aggregate(rows)
    json_path = output_dir / "summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
