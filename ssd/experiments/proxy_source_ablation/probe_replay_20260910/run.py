"""Run the full-distribution replay probe; validate artifacts, not only exit codes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["alpaca", "c4", "gsm", "humaneval"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 123])
    ap.add_argument("--numseqs", type=int, default=32)
    ap.add_argument("--output-len", type=int, default=256)
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "out")
    ap.add_argument("--gpus", default="2,3,4,5,6")
    ap.add_argument("--port", type=int, default=19400)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[3]
    args.out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=args.gpus,
               SSD_PROFILE="0", SSD_PROFILE_DUET="0", SSD_PROFILE_DUET_DETAIL="0",
               SSD_TREE_EXEC="0", SSD_TREE_ARENA="0", SSD_TREE_PROXY_GRAPH="0",
               SSD_TREE_EXEC_WARMUP="0", SSD_DUET_EXIT_REPLICA="1",
               SSD_DUET_PROBE_LAYERS="56,79", SSD_DUET_PROBE_KIND="distribution",
               SSD_DUET_PROBE_STRIDE="8")
    port = args.port
    for seed in args.seeds:
        for ds in args.datasets:
            tag = f"{ds}_seed{seed}"
            manifest = args.out.resolve() / f"{tag}.json"
            done = args.out.resolve() / f"{tag}.complete.json"
            log = args.out.resolve() / f"{tag}.log"
            if done.exists():
                previous = json.loads(done.read_text())
                if previous["numseqs"] != args.numseqs or previous["output_len"] != args.output_len:
                    raise RuntimeError(f"incompatible completed run: {done}")
                print(f"[skip] {tag}", flush=True)
                continue
            if manifest.exists():
                raise RuntimeError(f"unfinished output exists; choose a fresh --out: {manifest}")
            flags = [] if ds == "gsm" else [f"--{ds}"]
            command = [str(root / ".venv/bin/python"), "-O", str(root / "bench/bench.py"),
                       "--llama", "--size", "70", "--gpus", "5",
                       "--model_path", "/home/chokwans99/awq_calibrated/layerskip_llama2_70b",
                       "--draft_path", "/data/chokwans99/models/TinyLlama-1.1B-Chat-v1.0",
                       "--quant_awq", "--quant_awq_artifact", "/home/chokwans99/awq_artifacts/layerskip70b_awq_tp4",
                       *flags, "--numseqs", str(args.numseqs), "--output_len", str(args.output_len),
                       "--b", "1", "--temp", "1.0", "--seed", str(seed),
                       "--async", "--spec", "--duet", "--duet_exit_layer", "56",
                       "--duet_phase1_k", "8", "--duet_phase2_k", "4",
                       "--duet_draft_fan_out", "3", "--duet_p2_budget", "15",
                       "--duet_p1_tree_policy", "off", "--duet_p2_tree_policy", "off",
                       "--duet_only_proxy"]
            env.update(SSD_DUET_PROBE_OUT=str(manifest), SSD_DIST_PORT=str(port))
            print(f"[run] {tag} prompts={args.numseqs} output={args.output_len} port={port}", flush=True)
            start = time.time()
            with log.open("w") as f:
                result = subprocess.run(command, cwd=root, env=env, stdout=f, stderr=subprocess.STDOUT)
            text = log.read_text(errors="replace")
            if (result.returncode or not manifest.exists() or "Final Decode Throughput:" not in text
                    or "max|probe-engine|=0.000e+00 OK" not in text
                    or "Traceback (most recent call last):" in text or "PROBE_FLUSH_FAILED" in text):
                raise RuntimeError(f"run failed validation: {tag}; inspect {log}")
            meta = json.loads(manifest.read_text())
            import numpy as np
            ids, steps = set(), set()
            nrows = 0
            for chunk in meta["chunks"]:
                with np.load(manifest.parent / chunk["file"]) as z:
                    ids.update(z["seq_id"].tolist())
                    steps.update(z["step"].tolist())
                    nrows += len(z["step"])
                    if any(not np.isfinite(z[k]).all() for k in ("p_E", "p_T", "p_D")):
                        raise RuntimeError(f"nonfinite probability: {tag}")
            if len(ids) != args.numseqs or len(steps) != meta["n_samples"] or nrows != meta["n_rows"]:
                raise RuntimeError(f"incomplete row coverage: {tag}")
            for sid, count in meta["seq_steps"].items():
                sampled = []
                for chunk in meta["chunks"]:
                    with np.load(manifest.parent / chunk["file"]) as z:
                        sampled.extend(z["local_step"][(z["seq_id"] == int(sid)) & (z["position"] == 0)].tolist())
                if sampled != list(range(0, count, meta["stride"])):
                    raise RuntimeError(f"sampling coverage mismatch: {tag}/{sid}")
            record = dict(dataset=ds, seed=seed, numseqs=args.numseqs, output_len=args.output_len,
                          elapsed_s=time.time()-start, command=command,
                          n_steps=meta["n_steps"], n_samples=meta["n_samples"], n_rows=nrows, manifest=manifest.name,
                          validation="exit, throughput marker, tap exact match, all prompts, all sampled steps, finite float32 probabilities")
            done.write_text(json.dumps(record, indent=2))
            print(f"[done] {tag}: {meta['n_steps']} steps, {nrows} rows, {record['elapsed_s']:.1f}s", flush=True)
            port += 1


if __name__ == "__main__":
    main()
