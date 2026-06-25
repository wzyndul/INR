import argparse
import os
import math
import time
from datetime import datetime
from dotenv import load_dotenv
from comet_ml import Experiment

import xarray as xr
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

load_dotenv()


def set_seed(seed: int = 0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_experiment_name(args) -> str:
    arch_str = (
        "_".join(map(str, args.layer_widths))
        if args.layer_widths
        else f"h{args.hidden_dim}_l{args.num_layers}"
    )
    gan_str = (
        f"nc{args.n_critic}_lam{args.lambda_adv}"
        if args.n_critic > 0
        else "mse_only"
    )
    return f"sinet_fsize{args.fourier_mapping_size}_{arch_str}_{gan_str}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def get_args():
    parser = argparse.ArgumentParser(description="SiNET INR-GAN for E-OBS compression")
    parser.add_argument("--data_path", default="data_preprocessed.nc")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--layer_widths", type=int, nargs="+", default=None)
    parser.add_argument("--sorting_group_size", type=int, default=16)
    parser.add_argument("--fourier_scale", type=float, default=1.5)
    parser.add_argument("--fourier_mapping_size", type=int, default=64)
    parser.add_argument("--lambda_eikonal", type=float, default=0.0)
    parser.add_argument("--lambda_laplace", type=float, default=0.0)
    parser.add_argument("--n_iterations", type=int, default=5)
    parser.add_argument("--n_critic", type=int, default=5)
    parser.add_argument("--lr_G", type=float, default=1e-4)
    parser.add_argument("--lr_D", type=float, default=2.5e-5)
    parser.add_argument("--lambda_adv", type=float, default=0.001)
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=1.0,
        help="Gradient clipping max norm for generator (0 = disabled)",
    )
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=5000,
        help="Stop training if PSNR has not improved for this many iterations "
        "(0 = disabled). The best checkpoint is restored at the end.",
    )
    parser.add_argument("--output_dir", default="outputs/sinet/")
    parser.add_argument(
        "--experiment_name",
        default=None,
        help="Custom experiment name for Comet ML (auto-generated if not specified)",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--save_model",
        action="store_true",
        default=False,
        help="Save generator weights and metadata after training",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def load_eobs(data_path: str, device: str):
    ds = xr.open_dataset(data_path)
    lat_valid = ds["lat"].values.astype(np.float32)
    lon_valid = ds["lon"].values.astype(np.float32)
    temp_valid = ds["tg"].values.astype(np.float32)
    nan_mask = ds["nan_mask"].values.astype(bool)
    lat_vals = ds["lat_vals"].values
    lon_vals = ds["lon_vals"].values
    ds.close()
    H, W = nan_mask.shape

    lat_min, lat_max = float(lat_vals.min()), float(lat_vals.max())
    lon_min, lon_max = float(lon_vals.min()), float(lon_vals.max())
    lat_norm = 2.0 * (lat_valid - lat_min) / (lat_max - lat_min) - 1.0
    lon_norm = 2.0 * (lon_valid - lon_min) / (lon_max - lon_min) - 1.0

    temp_min = float(temp_valid.min())
    temp_max = float(temp_valid.max())
    temp_norm = 2.0 * (temp_valid - temp_min) / (temp_max - temp_min) - 1.0

    coords_norm = torch.from_numpy(np.stack([lat_norm, lon_norm], axis=1)).to(device)
    temps_norm = torch.from_numpy(temp_norm[:, None]).to(device)

    real_2d = np.zeros((H, W), dtype=np.float32)
    real_2d[nan_mask] = temp_norm
    real_grid = torch.from_numpy(real_2d[None, None]).to(device)

    meta = dict(
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
        temp_min=temp_min,
        temp_max=temp_max,
        H=H,
        W=W,
        nan_mask=nan_mask,
        lat_vals=lat_vals,
        lon_vals=lon_vals,
        varname="tg",
        n_valid=int(nan_mask.sum()),
    )
    return coords_norm, temps_norm, real_grid, nan_mask, meta


class FourierFeatures(nn.Module):
    def __init__(
        self, input_dim: int, mapping_size: int, scale: float, trainable: bool = False
    ):
        super().__init__()
        if trainable:
            self.log_scale = nn.Parameter(torch.log(torch.tensor(scale)))
            B = torch.randn((input_dim, mapping_size))
            self.B = nn.Parameter(B)
        else:
            self.log_scale = torch.log(torch.tensor(scale))
            B = torch.randn((input_dim, mapping_size))
            self.register_buffer("B", B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = 2 * np.pi * x @ (self.B * torch.exp(self.log_scale))
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class GroupSortActivation(nn.Module):
    def __init__(self, group_size: int = 16):
        super().__init__()
        self.group_size = group_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, dim = x.shape
        if dim % self.group_size != 0:
            raise ValueError(
                f"Input dim {dim} must be divisible by group_size {self.group_size}"
            )
        x_reshaped = x.view(batch_size, -1, self.group_size)
        x_sorted, _ = x_reshaped.sort(dim=-1)
        return x_sorted.view(batch_size, dim)


class SiNET(nn.Module):
    def __init__(
        self,
        layer_widths,
        sorting_group_size: int = 16,
        scale: float = 1.5,
        fourier_mapping_size: int = 64,
    ):
        super().__init__()
        assert len(layer_widths) >= 2, "Need at least 2 layers"
        self.layer_widths = layer_widths

        self.fourier_features = FourierFeatures(
            input_dim=2,
            mapping_size=fourier_mapping_size,
            scale=scale,
            trainable=False,
        )
        fourier_out_dim = fourier_mapping_size * 2

        mlps = []
        in_dim = fourier_out_dim
        for i, out_dim in enumerate(layer_widths):
            if i == len(layer_widths) - 1:  # Output layer
                mlps.append(nn.Linear(in_dim, out_dim, bias=True))
            else:
                mlps.append(nn.Linear(in_dim, out_dim, bias=True))
                mlps.append(GroupSortActivation(sorting_group_size))
            in_dim = out_dim

        self.net = nn.Sequential(*mlps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fourier_features = self.fourier_features(x)
        return self.net(fourier_features)


class PatchGANDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, stride=1, padding=1),
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 1, 4, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def compute_sdf_losses(coords, pred, target, lambda_eikonal, lambda_laplace):
    pred = pred.squeeze()
    target = target.squeeze()
    mse = F.mse_loss(pred, target)
    total = mse

    eikonal_val = torch.tensor(0.0, device=pred.device)
    laplace_val = torch.tensor(0.0, device=pred.device)

    if lambda_eikonal > 0 or lambda_laplace > 0:
        grad_outputs = torch.ones_like(pred)
        grad = torch.autograd.grad(
            outputs=pred,
            inputs=[coords],
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
        )[0]

    if lambda_eikonal > 0:
        grad_norm = grad.norm(dim=-1)
        eikonal_val = torch.abs(grad_norm - 1.0).mean()
        total = total + lambda_eikonal * eikonal_val

    if lambda_laplace > 0:
        grad2 = torch.autograd.grad(
            outputs=grad.sum(),
            inputs=[coords],
            create_graph=True,
            retain_graph=True,
        )[0]
        laplace_val = (grad2**2).norm(dim=-1).mean()
        total = total + lambda_laplace * laplace_val

    return total, mse, eikonal_val, laplace_val


def build_fake_grid(G, coords_norm, nan_mask, H, W, device, chunk_size=50_000):
    N = coords_norm.shape[0]
    preds = []
    for start in range(0, N, chunk_size):
        preds.append(G(coords_norm[start : start + chunk_size]))
    pred_flat = torch.cat(preds, dim=0)

    grid = torch.zeros(H, W, device=device)
    mask_tensor = torch.from_numpy(nan_mask).to(device)
    grid[mask_tensor] = pred_flat.squeeze(1)
    return grid.unsqueeze(0).unsqueeze(0)


def compute_metrics(pred_norm, target_norm, temp_min, temp_max):
    with torch.no_grad():
        mse_norm = F.mse_loss(pred_norm, target_norm).item()
        psnr = 10 * math.log10(4.0 / mse_norm) if mse_norm > 0 else float("inf")

        scale = (temp_max - temp_min) / 2.0
        pred_c = pred_norm.squeeze() * scale + (temp_max + temp_min) / 2.0
        target_c = target_norm.squeeze() * scale + (temp_max + temp_min) / 2.0
        diff = (pred_c - target_c).abs()
        mse_c = float(diff.pow(2).mean())
        rmse = float(math.sqrt(mse_c))
        mae = float(diff.mean())
        max_err = float(diff.max())
        denom = pred_c.abs() + target_c.abs() + 1e-8
        smape = float((200.0 * diff / denom).mean())
    return {
        "psnr": psnr,
        "mse_celsius": mse_c,
        "rmse_celsius": rmse,
        "mae_celsius": mae,
        "max_error_celsius": max_err,
        "smape": smape,
    }


def train(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device

    experiment = None
    api_key = os.getenv("COMET_API_KEY")
    if api_key:
        experiment = Experiment(
            api_key=api_key, project_name="sinet-inr-gan", workspace="ketiovv"
        )
        exp_name = args.experiment_name or get_experiment_name(args)
        experiment.set_name(exp_name)
    else:
        print("Warning: COMET_API_KEY not found in .env file")

    coords_norm, temps_norm, real_grid, nan_mask, meta = load_eobs(
        args.data_path, device
    )
    H, W = meta["H"], meta["W"]
    temp_min, temp_max = meta["temp_min"], meta["temp_max"]

    if args.layer_widths is not None:
        layer_widths = args.layer_widths
    else:
        layer_widths = [args.hidden_dim] * (args.num_layers - 1) + [1]

    G = SiNET(
        layer_widths=layer_widths,
        sorting_group_size=args.sorting_group_size,
        scale=args.fourier_scale,
        fourier_mapping_size=args.fourier_mapping_size,
    ).to(device)

    D = PatchGANDiscriminator().to(device)
    if experiment:
        experiment.log_parameters(
            {
                "model": "SiNET INR-GAN",
                "layer_widths": layer_widths,
                "fourier_mapping_size": args.fourier_mapping_size,
                "fourier_scale": args.fourier_scale,
                "sorting_group_size": args.sorting_group_size,
                "lambda_eikonal": args.lambda_eikonal,
                "lambda_laplace": args.lambda_laplace,
                "n_iterations": args.n_iterations,
                "n_critic": args.n_critic,
                "lr_G": args.lr_G,
                "lr_D": args.lr_D,
                "lambda_adv": args.lambda_adv,
                "seed": args.seed,
                "generator_params": sum(p.numel() for p in G.parameters()),
                "discriminator_params": sum(p.numel() for p in D.parameters()),
            }
        )

    opt_G = torch.optim.Adam(G.parameters(), lr=args.lr_G, betas=(0.0, 0.99))
    opt_D = torch.optim.Adam(D.parameters(), lr=args.lr_D, betas=(0.0, 0.99))

    log = dict(
        iter=[],
        L_mse=[],
        L_adv=[],
        L_total=[],
        L_D=[],
        psnr=[],
        mse=[],
        rmse=[],
        mae=[],
        max_err=[],
        smape=[],
        compression_time_seconds=None,
        decompression_time_seconds=None,
        best_iter=None,
        best_psnr=float("nan"),
        best_mse=float("nan"),
        best_rmse=float("nan"),
        best_mae=float("nan"),
        best_max_err=float("nan"),
        best_smape=float("nan"),
        stopped_early=False,
    )

    best_psnr = -float("inf")
    best_iter = 0
    best_state = None

    loss_D = torch.tensor(0.0, device=device)
    loss_adv = torch.tensor(0.0, device=device)

    use_gan = args.n_critic > 0

    # CT - czas kompresji (calego treningu)
    if device == "cuda":
        torch.cuda.synchronize()
    t_train_start = time.perf_counter()

    for iteration in range(1, args.n_iterations + 1):
        lam = args.lambda_adv if use_gan else 0.0

        if use_gan:
            for _ in range(args.n_critic):
                with torch.no_grad():
                    fake_grid = build_fake_grid(G, coords_norm, nan_mask, H, W, device)

                d_real = D(real_grid)
                d_fake = D(fake_grid)

                loss_D = 0.5 * F.mse_loss(
                    d_real, torch.ones_like(d_real)
                ) + 0.5 * F.mse_loss(d_fake, torch.zeros_like(d_fake))

                opt_D.zero_grad()
                loss_D.backward()
                opt_D.step()

        mask_tensor = torch.from_numpy(nan_mask).to(device)
        target_valid = temps_norm

        if args.lambda_eikonal > 0 or args.lambda_laplace > 0:
            coords_grad = coords_norm.detach().requires_grad_(True)
            fake_grid = build_fake_grid(G, coords_grad, nan_mask, H, W, device)
            pred_valid = fake_grid.squeeze()[mask_tensor].unsqueeze(1)
            loss_sdf, loss_mse, eik_val, lap_val = compute_sdf_losses(
                coords_grad,
                pred_valid,
                target_valid,
                args.lambda_eikonal,
                args.lambda_laplace,
            )
        else:
            fake_grid = build_fake_grid(G, coords_norm, nan_mask, H, W, device)
            pred_valid = fake_grid.squeeze()[mask_tensor].unsqueeze(1)
            loss_mse = F.mse_loss(pred_valid.squeeze(), target_valid.squeeze())
            loss_sdf = loss_mse

        if use_gan and lam > 0:
            d_fake_g = D(fake_grid)
            loss_adv = 0.5 * F.mse_loss(d_fake_g, torch.ones_like(d_fake_g))
        else:
            loss_adv = torch.tensor(0.0, device=device)

        loss_G = loss_sdf + lam * loss_adv

        opt_G.zero_grad()
        loss_G.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=args.grad_clip)
        opt_G.step()

        if iteration % args.log_every == 0 or iteration == 1:
            m = compute_metrics(pred_valid, target_valid, temp_min, temp_max)

            log["iter"].append(iteration)
            log["L_mse"].append(loss_mse.item())
            log["L_adv"].append(loss_adv.item())
            log["L_total"].append(loss_G.item())
            log["L_D"].append(loss_D.item())
            log["psnr"].append(m["psnr"])
            log["mse"].append(m["mse_celsius"])
            log["rmse"].append(m["rmse_celsius"])
            log["mae"].append(m["mae_celsius"])
            log["max_err"].append(m["max_error_celsius"])
            log["smape"].append(m["smape"])

            if m["psnr"] > best_psnr:
                best_psnr = m["psnr"]
                best_iter = iteration
                best_state = {
                    k: v.detach().cpu().clone() for k, v in G.state_dict().items()
                }
                log["best_iter"] = best_iter
                log["best_psnr"] = m["psnr"]
                log["best_mse"] = m["mse_celsius"]
                log["best_rmse"] = m["rmse_celsius"]
                log["best_mae"] = m["mae_celsius"]
                log["best_max_err"] = m["max_error_celsius"]
                log["best_smape"] = m["smape"]

            if experiment:
                experiment.log_metrics(
                    {
                        "loss_mse": loss_mse.item(),
                        "loss_adv": loss_adv.item(),
                        "loss_total": loss_G.item(),
                        "loss_D": loss_D.item(),
                        "lambda_adv": lam,
                        "psnr_db": m["psnr"],
                        "mse_celsius": m["mse_celsius"],
                        "rmse_celsius": m["rmse_celsius"],
                        "mae_celsius": m["mae_celsius"],
                        "max_error_celsius": m["max_error_celsius"],
                        "smape_percent": m["smape"],
                    },
                    step=iteration,
                )

            if (
                args.early_stop_patience > 0
                and iteration - best_iter >= args.early_stop_patience
            ):
                log["stopped_early"] = True
                print(
                    f"Early stopping at iteration {iteration}: PSNR has not "
                    f"improved for {iteration - best_iter} iterations "
                    f"(best={best_psnr:.4f} dB @ iter {best_iter})"
                )
                break

    # CT - koniec
    if device == "cuda":
        torch.cuda.synchronize()
    log["compression_time_seconds"] = time.perf_counter() - t_train_start

    # przywroc najlepszy checkpoint (po PSNR) jako wynik koncowy
    if best_state is not None:
        G.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        print(
            f"Restored best checkpoint: PSNR={best_psnr:.4f} dB @ iter {best_iter}"
        )

    # DT - czas dekompresji (pojedynczy forward pass G na pelnych coords, usredniony po 10 powtorzeniach)
    G.eval()
    with torch.no_grad():
        # warm-up
        _ = build_fake_grid(G, coords_norm, nan_mask, H, W, device)
        if device == "cuda":
            torch.cuda.synchronize()
        t_dec_start = time.perf_counter()
        N_RUNS = 10
        for _ in range(N_RUNS):
            _ = build_fake_grid(G, coords_norm, nan_mask, H, W, device)
        if device == "cuda":
            torch.cuda.synchronize()
        log["decompression_time_seconds"] = (time.perf_counter() - t_dec_start) / N_RUNS

    if experiment:
        experiment.log_metrics({
            "compression_time_seconds": log["compression_time_seconds"],
            "decompression_time_seconds": log["decompression_time_seconds"],
        })

    return G, D, coords_norm, temps_norm, real_grid, nan_mask, meta, log, experiment


def save_outputs(
    G, coords_norm, temps_norm, real_grid, nan_mask, meta, log, args, experiment=None
):
    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device
    H, W = meta["H"], meta["W"]
    temp_min, temp_max = meta["temp_min"], meta["temp_max"]

    gen_path = os.path.join(args.output_dir, "generator_weights.pt")
    meta_path = os.path.join(args.output_dir, "metadata.pt")

    if args.save_model:
        torch.save(G.state_dict(), gen_path)
        torch.save(
            {
                "lat_min": meta["lat_min"],
                "lat_max": meta["lat_max"],
                "lon_min": meta["lon_min"],
                "lon_max": meta["lon_max"],
                "temp_min": temp_min,
                "temp_max": temp_max,
                "H": H,
                "W": W,
                "nan_mask": nan_mask,
                "lat_vals": meta["lat_vals"],
                "lon_vals": meta["lon_vals"],
                "varname": meta["varname"],
                "layer_widths": G.layer_widths,
                "fourier_scale": args.fourier_scale,
                "fourier_mapping_size": args.fourier_mapping_size,
                "sorting_group_size": args.sorting_group_size,
            },
            meta_path,
        )

    G.eval()
    with torch.no_grad():
        fake_grid = build_fake_grid(G, coords_norm, nan_mask, H, W, device)

    mask_tensor = torch.from_numpy(nan_mask).to(device)
    pred_valid = fake_grid.squeeze()[mask_tensor].unsqueeze(1)
    m = compute_metrics(pred_valid, temps_norm, temp_min, temp_max)

    if experiment:
        n_params = sum(p.numel() for p in G.parameters())
        bits_per_value = 32
        compression_factor = (meta["n_valid"] * bits_per_value) / (n_params * bits_per_value)
        experiment.log_metrics(
            {
                "final_psnr_db": m["psnr"],
                "final_mse_celsius": m["mse_celsius"],
                "final_rmse_celsius": m["rmse_celsius"],
                "final_mae_celsius": m["mae_celsius"],
                "final_max_error_celsius": m["max_error_celsius"],
                "final_smape_percent": m["smape"],
                "compression_factor": compression_factor,
            }
        )

    def denorm(t_norm):
        return (t_norm + 1.0) * 0.5 * (temp_max - temp_min) + temp_min

    real_np = denorm(real_grid.squeeze().cpu().numpy())
    fake_np = denorm(fake_grid.squeeze().cpu().numpy())
    real_np[~nan_mask] = np.nan
    fake_np[~nan_mask] = np.nan
    err_np = np.abs(real_np - fake_np)
    err_np[~nan_mask] = np.nan

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    cmap_temp = "RdYlBu_r"
    cmap_err = "hot_r"
    vmin, vmax = np.nanmin(real_np), np.nanmax(real_np)
    emax = np.nanmax(err_np)

    im0 = axes[0].imshow(real_np, origin="lower", cmap=cmap_temp, vmin=vmin, vmax=vmax)
    axes[0].set_title("Original (E-OBS)")
    plt.colorbar(im0, ax=axes[0], label="°C")

    im1 = axes[1].imshow(fake_np, origin="lower", cmap=cmap_temp, vmin=vmin, vmax=vmax)
    axes[1].set_title("Reconstructed (SiNET)")
    plt.colorbar(im1, ax=axes[1], label="°C")

    im2 = axes[2].imshow(err_np, origin="lower", cmap=cmap_err, vmin=0, vmax=emax)
    axes[2].set_title("Absolute Error")
    plt.colorbar(im2, ax=axes[2], label="°C")

    metrics_text = f"PSNR: {psnr:.2f} dB\nRMSE: {rmse:.4f} °C\nMAE: {mae:.4f} °C\nMaxErr: {max_err:.4f} °C"
    fig.text(
        0.5,
        -0.02,
        metrics_text,
        ha="center",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    fig.suptitle(f"SiNET INR-GAN", fontsize=11)
    plt.tight_layout()
    cmp_path = os.path.join(args.output_dir, "comparison.png")
    plt.savefig(cmp_path, dpi=150, bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)

    axes[0, 0].plot(log["iter"], log["L_mse"], color="steelblue", marker="o")
    axes[0, 0].set_ylabel("L_MSE")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_title("Loss: MSE")

    axes[0, 1].plot(log["iter"], log["L_adv"], color="coral", marker="o")
    axes[0, 1].set_ylabel("L_Adv")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_title("Loss: Adversarial")

    axes[1, 0].plot(log["iter"], log["psnr"], color="darkorange", marker="s")
    axes[1, 0].set_ylabel("PSNR (dB)")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_title("Peak Signal-to-Noise Ratio")

    axes[1, 1].plot(log["iter"], log["rmse"], color="forestgreen", marker="s")
    axes[1, 1].set_ylabel("RMSE (°C)")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_title("Root Mean Squared Error")

    axes[2, 0].plot(log["iter"], log["mae"], color="purple", marker="^")
    axes[2, 0].set_ylabel("MAE (°C)")
    axes[2, 0].set_xlabel("Iteration")
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].set_title("Mean Absolute Error")

    axes[2, 1].plot(log["iter"], log["max_err"], color="red", marker="^")
    axes[2, 1].set_ylabel("MaxErr (°C)")
    axes[2, 1].set_xlabel("Iteration")
    axes[2, 1].grid(True, alpha=0.3)
    axes[2, 1].set_title("Maximum Error")

    fig.suptitle(f"Training Metrics")
    plt.tight_layout()
    crv_path = os.path.join(args.output_dir, "training_curves.png")
    plt.savefig(crv_path, dpi=150, bbox_inches="tight")
    plt.close()

    G.train()


def main():
    args = get_args()
    set_seed(args.seed)
    G, D, coords_norm, temps_norm, real_grid, nan_mask, meta, log, experiment = train(
        args
    )
    save_outputs(
        G, coords_norm, temps_norm, real_grid, nan_mask, meta, log, args, experiment
    )


if __name__ == "__main__":
    main()
