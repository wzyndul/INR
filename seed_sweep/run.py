import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from siren_inr_gan.train import train as train_siren, set_seed as set_seed_s
from sinet_inr_gan.train import train as train_sinet, set_seed as set_seed_n


def build_args(network, params, seed, data_path, output_dir):
    width = int(params["width"])
    layer_widths = [width] * 3 + [1]
    common = dict(
        data_path=data_path,
        hidden_dim=width,
        num_layers=4,
        layer_widths=layer_widths,
        n_iterations=int(params["n_iterations"]),
        n_critic=int(params.get("n_critic", 0)),
        lr_G=float(params["lr_G"]),
        lr_D=float(params.get("lr_D", 0.0)),
        lambda_adv=float(params.get("lambda_adv", 0.0)),
        grad_clip=float(params["grad_clip"]),
        log_every=500,
        early_stop_patience=5000,
        output_dir=output_dir,
        experiment_name=None,
        seed=seed,
        save_model=False,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    if network == "siren":
        return SimpleNamespace(**common, omega_0=30.0, omega_hidden=30.0)
    return SimpleNamespace(
        **common,
        sorting_group_size=int(params["sorting_group_size"]),
        fourier_scale=float(params["fourier_scale"]),
        fourier_mapping_size=int(params["fourier_mapping_size"]),
        lambda_eikonal=float(params["lambda_eikonal"]),
        lambda_laplace=float(params["lambda_laplace"]),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks_file", required=True)
    ap.add_argument("--task_id", type=int, required=True)
    ap.add_argument("--data_path", default=os.path.join(ROOT, "data_preprocessed.nc"))
    ap.add_argument("--out_root", required=True)
    args = ap.parse_args()

    with open(args.tasks_file) as f:
        task = json.load(f)["tasks"][args.task_id]

    network = task["network"]
    cfg_idx = task["config_idx"]
    seed = task["seed"]

    output_dir = os.path.join(args.out_root, network, f"p{cfg_idx:02d}", f"seed{seed:02d}")
    os.makedirs(output_dir, exist_ok=True)
    result_path = os.path.join(output_dir, "result.json")

    if os.path.exists(result_path):
        return

    train_args = build_args(network, task["params"], seed, args.data_path, output_dir)
    if network == "siren":
        set_seed_s(seed)
        train_fn = train_siren
    else:
        set_seed_n(seed)
        train_fn = train_sinet

    t0 = time.time()
    G, _, _, _, _, _, meta, log, experiment = train_fn(train_args)
    wall = time.time() - t0

    n_params = sum(p.numel() for p in G.parameters())

    out = {
        "task_id": args.task_id,
        "network": network,
        "config_idx": cfg_idx,
        "seed": seed,
        "source_trial": task["source_trial"],
        "psnr": float(log["best_psnr"]),
        "mse_celsius": float(log["best_mse"]),
        "rmse_celsius": float(log["best_rmse"]),
        "mae_celsius": float(log["best_mae"]),
        "max_error_celsius": float(log["best_max_err"]),
        "smape_percent": float(log["best_smape"]),
        "compression_factor": meta["n_valid"] / n_params,
        "compression_time_seconds": float(log["compression_time_seconds"]),
        "decompression_time_seconds": float(log["decompression_time_seconds"]),
        "best_iter": int(log["best_iter"]),
        "stopped_early": bool(log.get("stopped_early", False)),
        "wall_seconds": wall,
        "generator_params": n_params,
        "n_valid": int(meta["n_valid"]),
    }
    with open(result_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"#{args.task_id} {network} p{cfg_idx} s{seed}: "
          f"PSNR={out['psnr']:.3f} CF={out['compression_factor']:.2f} "
          f"wall={wall:.0f}s")

    if experiment:
        experiment.end()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
