import argparse
import os
import sys
import time
from types import SimpleNamespace

import torch

import optuna
from optuna.samplers import NSGAIISampler

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

TAG = "gan"

from siren_inr_gan.train import train as train_siren, set_seed as set_seed_s
from sinet_inr_gan.train import train as train_sinet, set_seed as set_seed_n


WIDTH_GRID = [16, 32, 48, 64, 80, 96]


def make_siren_args(trial: optuna.Trial, data_path: str, output_root: str) -> SimpleNamespace:
    depth = 3
    width = trial.suggest_categorical("width", WIDTH_GRID)
    n_critic = trial.suggest_int("n_critic", 1, 3)
    lr_G = trial.suggest_float("lr_G", 1e-5, 5e-4, log=True)
    lr_D = trial.suggest_float("lr_D", 1e-6, 5e-4, log=True)
    lambda_adv = trial.suggest_float("lambda_adv", 1e-4, 0.5, log=True)
    grad_clip = trial.suggest_float("grad_clip", 0.1, 5.0)
    n_iterations = trial.suggest_int("n_iterations", 15000, 30000, step=5000)

    layer_widths = [width] * depth + [1]

    return SimpleNamespace(
        data_path=data_path,
        hidden_dim=width,
        num_layers=depth + 1,
        layer_widths=layer_widths,
        omega_0=30.0,
        omega_hidden=30.0,
        n_iterations=n_iterations,
        n_critic=n_critic,
        lr_G=lr_G,
        lr_D=lr_D,
        lambda_adv=lambda_adv,
        grad_clip=grad_clip,
        log_every=500,
        early_stop_patience=5000,
        output_dir=os.path.join(output_root, f"trial_{trial.number:04d}"),
        experiment_name=None,
        seed=0,
        save_model=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )


def make_sinet_args(trial: optuna.Trial, data_path: str, output_root: str) -> SimpleNamespace:
    depth = 3
    width = trial.suggest_categorical("width", WIDTH_GRID)
    sorting_group_size = trial.suggest_categorical("sorting_group_size", [1, 2, 4, 8, 16])
    fourier_mapping_size = trial.suggest_int("fourier_mapping_size", 8, 64, step=8)
    fourier_scale = trial.suggest_float("fourier_scale", 0.5, 5.0)
    lambda_eikonal = trial.suggest_float("lambda_eikonal", 1e-5, 1e-1, log=True)
    lambda_laplace = trial.suggest_float("lambda_laplace", 1e-5, 1e-1, log=True)
    n_critic = trial.suggest_int("n_critic", 1, 3)
    lr_G = trial.suggest_float("lr_G", 1e-5, 5e-4, log=True)
    lr_D = trial.suggest_float("lr_D", 1e-6, 5e-4, log=True)
    lambda_adv = trial.suggest_float("lambda_adv", 1e-4, 0.5, log=True)
    grad_clip = trial.suggest_float("grad_clip", 0.1, 5.0)
    n_iterations = trial.suggest_int("n_iterations", 15000, 30000, step=5000)

    layer_widths = [width] * depth + [1]

    return SimpleNamespace(
        data_path=data_path,
        hidden_dim=width,
        num_layers=depth + 1,
        layer_widths=layer_widths,
        sorting_group_size=sorting_group_size,
        fourier_scale=fourier_scale,
        fourier_mapping_size=fourier_mapping_size,
        lambda_eikonal=lambda_eikonal,
        lambda_laplace=lambda_laplace,
        n_iterations=n_iterations,
        n_critic=n_critic,
        lr_G=lr_G,
        lr_D=lr_D,
        lambda_adv=lambda_adv,
        grad_clip=grad_clip,
        log_every=500,
        early_stop_patience=5000,
        output_dir=os.path.join(output_root, f"trial_{trial.number:04d}"),
        experiment_name=None,
        seed=0,
        save_model=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )


def run_trial(trial: optuna.Trial, network: str, data_path: str, output_root: str,
              smoke_test: bool = False):
    if network == "siren":
        args = make_siren_args(trial, data_path, output_root)
        set_seed_s(args.seed)
        train_fn = train_siren
    elif network == "sinet":
        args = make_sinet_args(trial, data_path, output_root)
        set_seed_n(args.seed)
        train_fn = train_sinet
    else:
        raise ValueError(network)

    if smoke_test:
        args.n_iterations = 200
        args.log_every = 50

    os.makedirs(args.output_dir, exist_ok=True)
    t0 = time.time()

    G, D, coords_norm, temps_norm, real_grid, nan_mask, meta, log, experiment = train_fn(args)

    gen_path = os.path.join(args.output_dir, "generator_weights.pt")
    torch.save(G.state_dict(), gen_path)

    n_valid = meta["n_valid"]
    n_params = sum(p.numel() for p in G.parameters())
    BITS_PER_VALUE = 32  # zarowno temperatura, jak i wagi sieci sa zapisane jako float32

    # compression_factor = (liczba_punktow * bity_na_dana) / (liczba_parametrow * bity_na_parametr)
    compression_factor = (n_valid * BITS_PER_VALUE) / (n_params * BITS_PER_VALUE)

    # wynik koncowy = najlepszy checkpoint (po PSNR), nie ostatnia iteracja
    final_psnr = log.get("best_psnr", float("nan"))
    final_mse = log.get("best_mse", float("nan"))
    final_rmse = log.get("best_rmse", float("nan"))
    final_mae = log.get("best_mae", float("nan"))
    final_max_err = log.get("best_max_err", float("nan"))
    final_smape = log.get("best_smape", float("nan"))
    ct = log.get("compression_time_seconds", float("nan"))
    dt = log.get("decompression_time_seconds", float("nan"))

    trial.set_user_attr("final_mse_celsius", final_mse)
    trial.set_user_attr("final_rmse_celsius", final_rmse)
    trial.set_user_attr("final_mae_celsius", final_mae)
    trial.set_user_attr("final_max_error_celsius", final_max_err)
    trial.set_user_attr("final_smape_percent", final_smape)
    trial.set_user_attr("generator_params", n_params)
    trial.set_user_attr("n_valid", n_valid)
    trial.set_user_attr("file_size_bytes", os.path.getsize(gen_path))
    trial.set_user_attr("compression_time_seconds", ct)
    trial.set_user_attr("decompression_time_seconds", dt)
    trial.set_user_attr("best_iter", log.get("best_iter"))
    trial.set_user_attr("stopped_early", log.get("stopped_early", False))
    trial.set_user_attr("wall_seconds", time.time() - t0)

    if experiment:
        experiment.log_metrics(
            {
                "final_psnr_db": final_psnr,
                "final_mse_celsius": final_mse,
                "final_rmse_celsius": final_rmse,
                "final_mae_celsius": final_mae,
                "final_max_error_celsius": final_max_err,
                "final_smape_percent": final_smape,
                "compression_factor": compression_factor,
                "compression_time_seconds": ct,
                "decompression_time_seconds": dt,
                "trial_number": trial.number,
            }
        )
        experiment.add_tag(f"optuna_{network}")
        experiment.end()

    del G, D, coords_norm, temps_norm, real_grid
    torch.cuda.empty_cache()

    return final_psnr, compression_factor


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--network", choices=["siren", "sinet"], required=True)
    p.add_argument("--n_trials", type=int, default=200)
    p.add_argument("--data_path", default=os.path.join(ROOT, "data_preprocessed.nc"))
    p.add_argument("--output_root", default=None)
    p.add_argument("--storage", default=None)
    p.add_argument("--study_name", default=None)
    p.add_argument(
        "--population_size",
        type=int,
        default=24,
        help="NSGA-II population size — should be << n_trials",
    )
    p.add_argument(
        "--seed", type=int, default=0, help="Sampler seed for reproducibility"
    )
    p.add_argument(
        "--smoke_test",
        action="store_true",
        help="Force n_iterations=200 for fast end-to-end pipeline test",
    )
    return p.parse_args()


def main():
    args = get_args()
    output_root = args.output_root or os.path.join(HERE, f"outputs_{TAG}", args.network)
    storage = args.storage or f"sqlite:///{os.path.join(ROOT, f'optuna_{args.network}_{TAG}.db')}"
    study_name = args.study_name or f"{args.network}_compression_pareto_{TAG}"

    os.makedirs(output_root, exist_ok=True)

    sampler = NSGAIISampler(population_size=args.population_size, seed=args.seed)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        directions=["maximize", "maximize"],
        sampler=sampler,
        load_if_exists=True,
    )

    print(f"[{args.network}] storage={storage} study={study_name}")
    print(f"[{args.network}] existing trials: {len(study.trials)}, target: {args.n_trials}")

    study.optimize(
        lambda t: run_trial(t, args.network, args.data_path, output_root, args.smoke_test),
        n_trials=args.n_trials,
        gc_after_trial=True,
        catch=(RuntimeError, ValueError),
    )

    print(f"\n[{args.network}] Pareto front:")
    for t in study.best_trials:
        print(
            f"  trial {t.number:>4}  PSNR={t.values[0]:.3f}dB  comp_ratio={t.values[1]:.4f}  params={t.user_attrs.get('generator_params', '?')}  rmse={t.user_attrs.get('final_rmse_celsius', '?')}"
        )


if __name__ == "__main__":
    main()
