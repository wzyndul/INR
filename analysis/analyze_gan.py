import os
import csv
import numpy as np
import optuna
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PLOTS = os.path.join(HERE, "plots_gan")
os.makedirs(PLOTS, exist_ok=True)

N_VALID = 120_810
COLORS = {"siren": "#d95f02", "sinet": "#1b9e77"}
MARKERS = {"siren": "o", "sinet": "s"}
LABELS = {"siren": "SIREN", "sinet": "SiNET"}

HP_RANGES = {
    "width": (16, 96),
    "n_critic": (1, 3),
    "lr_G": (1e-5, 5e-4),
    "lr_D": (1e-6, 5e-4),
    "lambda_adv": (1e-4, 0.5),
    "grad_clip": (0.1, 5.0),
    "n_iterations": (15000, 30000),
    "sorting_group_size": (1, 16),
    "fourier_mapping_size": (8, 64),
    "fourier_scale": (0.5, 5.0),
    "lambda_eikonal": (1e-5, 1e-1),
    "lambda_laplace": (1e-5, 1e-1),
}
def load(net):
    return optuna.load_study(
        study_name=f"{net}_compression_pareto_gan",
        storage=f"sqlite:///{os.path.join(ROOT, f'optuna_{net}_gan.db')}",
    )


def trials_to_arrays(study, net):
    comp = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    arr = {
        "trial": np.array([t.number for t in comp]),
        "psnr": np.array([t.values[0] for t in comp]),
        "params": np.array([t.user_attrs.get("generator_params", 1) for t in comp]),
        "rmse": np.array([t.user_attrs.get("final_rmse_celsius", np.nan) for t in comp]),
        "mae": np.array([t.user_attrs.get("final_mae_celsius", np.nan) for t in comp]),
        "wall": np.array([t.user_attrs.get("wall_seconds", np.nan) for t in comp]),
        "lambda_adv": np.array([t.params.get("lambda_adv", np.nan) for t in comp]),
        "width": np.array([t.params.get("width", 0) for t in comp]),
        "n_critic": np.array([t.params.get("n_critic", 0) for t in comp]),
    }
    arr["cf"] = N_VALID / arr["params"]
    pareto = study.best_trials
    p_psnr = np.array([t.values[0] for t in pareto])
    p_params = np.array([t.user_attrs.get("generator_params", 1) for t in pareto])
    p_cf = N_VALID / p_params
    p_rmse = np.array([t.user_attrs.get("final_rmse_celsius", np.nan) for t in pareto])
    p_mae = np.array([t.user_attrs.get("final_mae_celsius", np.nan) for t in pareto])
    p_trial = np.array([t.number for t in pareto])
    order = np.argsort(p_cf)
    pareto_dict = dict(
        trial=p_trial[order], psnr=p_psnr[order], params=p_params[order],
        cf=p_cf[order], rmse=p_rmse[order], mae=p_mae[order],
        full=[pareto[i] for i in order],
    )
    return arr, pareto_dict


def _save(fig, fname):
    fig.savefig(os.path.join(PLOTS, fname), dpi=170, bbox_inches="tight")
    plt.close(fig)


def pareto_main(data):
    fig, ax = plt.subplots(figsize=(8, 5))
    for net in ("sinet", "siren"):
        a, p = data[net]
        ax.scatter(a["cf"], a["psnr"], alpha=0.15, s=14, color=COLORS[net])
        ax.plot(p["cf"], p["psnr"], "-", color=COLORS[net], linewidth=1.8)
        ax.scatter(
            p["cf"], p["psnr"], s=70, marker=MARKERS[net],
            color=COLORS[net], edgecolor="black", linewidth=0.6,
            label=LABELS[net],
            zorder=5,
        )
    ax.set_xlabel("Wielkość kompresji", fontsize=11)
    ax.set_ylabel("PSNR [dB]", fontsize=11)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"×{int(round(v))}"))
    ax.set_xlim(left=0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=10)
    fig.tight_layout()
    _save(fig, "pareto_main_gan.png")


def rate_distortion(data):
    fig, ax = plt.subplots(figsize=(8, 5))
    for net in ("sinet", "siren"):
        a, p = data[net]
        ax.scatter(a["params"], a["psnr"], alpha=0.15, s=14, color=COLORS[net])
        ax.plot(p["params"], p["psnr"], "-", color=COLORS[net], linewidth=1.8)
        ax.scatter(
            p["params"], p["psnr"], s=70, marker=MARKERS[net],
            color=COLORS[net], edgecolor="black", linewidth=0.6,
            label=f"{LABELS[net]}", zorder=5,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Liczba parametrów generatora", fontsize=11)
    ax.set_ylabel("PSNR [dB]", fontsize=11)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right", fontsize=10)
    fig.tight_layout()
    _save(fig, "rate_distortion_gan.png")


def lambda_sensitivity(data):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, net in zip(axes, ("siren", "sinet")):
        a, _ = data[net]
        bins = np.logspace(-4, np.log10(0.5), 6)
        groups, ticklabels = [], []
        for i in range(len(bins) - 1):
            mask = (a["lambda_adv"] >= bins[i]) & (a["lambda_adv"] < bins[i + 1])
            if mask.sum() > 0:
                groups.append(a["psnr"][mask])
                ticklabels.append(f"{bins[i]:.0e}\n-\n{bins[i+1]:.0e}")
        bp = ax.boxplot(groups, showmeans=True, widths=0.6, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor(COLORS[net])
            patch.set_alpha(0.6)
        ax.set_xticklabels(ticklabels, fontsize=8)
        ax.set_xlabel(r"Waga członu adwersarialnego $\lambda_{adv}$", fontsize=10)
        ax.set_title(f"{LABELS[net]}", fontsize=12)
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel("PSNR [dB]", fontsize=11)
    fig.tight_layout()
    _save(fig, "lambda_sensitivity_gan.png")


def lambda_vs_cf(data):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, net in zip(axes, ("siren", "sinet")):
        a, _ = data[net]
        bins = np.logspace(-4, np.log10(0.5), 6)
        groups, ticklabels = [], []
        for i in range(len(bins) - 1):
            mask = (a["lambda_adv"] >= bins[i]) & (a["lambda_adv"] < bins[i + 1])
            if mask.sum() > 0:
                groups.append(a["cf"][mask])
                ticklabels.append(f"{bins[i]:.0e}\n-\n{bins[i+1]:.0e}")
        bp = ax.boxplot(groups, showmeans=True, widths=0.6, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor(COLORS[net])
            patch.set_alpha(0.6)
        ax.set_xticklabels(ticklabels, fontsize=8)
        ax.set_xlabel(r"Waga członu adwersarialnego $\lambda_{adv}$", fontsize=10)
        ax.set_title(f"{LABELS[net]}", fontsize=12)
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"×{int(round(v))}"))
    axes[0].set_ylabel("Wielkość kompresji", fontsize=11)
    fig.tight_layout()
    _save(fig, "lambda_cf_gan.png")


def hp_importance(studies):
    targets = [
        ("psnr", lambda t: t.values[0], "PSNR"),
        (
            "cf",
            lambda t: N_VALID / t.user_attrs.get("generator_params", 1),
            "wielkość kompresji",
        ),
    ]
    for tag, target_fn, target_label in targets:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for ax, net in zip(axes, ("siren", "sinet")):
            try:
                imp = optuna.importance.get_param_importances(
                    studies[net], target=target_fn
                )
                names = list(imp.keys())
                vals = list(imp.values())
                order = np.argsort(vals)
                y = np.array(names)[order]
                v = np.array(vals)[order]
                ax.barh(y, v, color=COLORS[net], alpha=0.75, edgecolor="black", linewidth=0.5)
                ax.set_xlabel("Waga ważności", fontsize=10)
                ax.set_title(f"{LABELS[net]}", fontsize=12)
                ax.grid(True, axis="x", alpha=0.3)
            except Exception as e:
                ax.text(0.5, 0.5, f"Brak danych: {e}", ha="center", va="center",
                        transform=ax.transAxes)
        fig.tight_layout()
        fname = "hp_importance_gan.png" if tag == "psnr" else "hp_importance_cf_gan.png"
        _save(fig, fname)


def convergence_efficiency(data):
    fig, ax = plt.subplots(figsize=(8, 5))
    for net in ("siren", "sinet"):
        a, _ = data[net]
        wall_min = a["wall"] / 60.0
        ax.scatter(wall_min, a["psnr"], alpha=0.3, s=18, color=COLORS[net], label=f"{LABELS[net]}")
    ax.set_xlabel("Czas treningu [min]", fontsize=11)
    ax.set_ylabel("PSNR [dB]", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=10)
    fig.tight_layout()
    _save(fig, "convergence_efficiency_gan.png")


def width_cf(data):
    """Przy d=3 stalym - barplot mediany CR vs szerokosc."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, net in zip(axes, ("siren", "sinet")):
        a, _ = data[net]
        widths = sorted(set(int(w) for w in a["width"] if w > 0))
        meds = []
        for w in widths:
            m = a["width"] == w
            meds.append(np.median(a["cf"][m]))
        meds = np.array(meds)
        ax.bar(range(len(widths)), meds, color=COLORS[net], alpha=0.8,
               edgecolor=COLORS[net], linewidth=0.5)
        for i, m in enumerate(meds):
            ax.text(i, m, f"×{m:.0f}", ha="center", va="bottom", fontsize=10)
        ax.set_xticks(range(len(widths)))
        ax.set_xticklabels(widths)
        ax.set_xlabel("Szerokość warstwy $w$", fontsize=10)
        ax.set_title(f"{LABELS[net]}", fontsize=12)
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"×{int(round(v))}"))
    axes[0].set_ylabel("Mediana wielkości kompresji", fontsize=11)
    fig.tight_layout()
    _save(fig, "width_cf_gan.png")


def pareto_hp_table(data):
    siren_hps = ["width", "n_critic", "lr_G", "lr_D", "lambda_adv",
                 "grad_clip", "n_iterations"]
    sinet_only = ["sorting_group_size", "fourier_mapping_size", "fourier_scale",
                  "lambda_eikonal", "lambda_laplace"]

    def collect(net, hp_list):
        _, p = data[net]
        out = {}
        for hp in hp_list:
            vals = [t.params[hp] for t in p["full"] if hp in t.params]
            if vals:
                out[hp] = (min(vals), max(vals))
        return out

    siren_d = collect("siren", siren_hps)
    sinet_d = collect("sinet", siren_hps + sinet_only)

    path_csv = os.path.join(PLOTS, "pareto_hp_table_gan.csv")
    with open(path_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "hp", "siren_min", "siren_max", "sinet_min", "sinet_max",
            "search_min", "search_max",
        ])
        for hp in siren_hps:
            lo, hi = HP_RANGES[hp]
            s = siren_d.get(hp, (None, None))
            n = sinet_d.get(hp, (None, None))
            w.writerow([hp, s[0], s[1], n[0], n[1], lo, hi])
        for hp in sinet_only:
            lo, hi = HP_RANGES[hp]
            n = sinet_d.get(hp, (None, None))
            w.writerow([hp, "", "", n[0], n[1], lo, hi])
    print(f"Zapisano CSV: {path_csv}")


def write_pareto_csv(data):
    path = os.path.join(PLOTS, "pareto_table_gan.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "network", "trial", "params", "cf_x", "psnr_db", "rmse_C", "mae_C",
            "width", "n_critic", "lr_G", "lr_D", "lambda_adv", "grad_clip",
            "n_iterations", "sorting_group_size", "fourier_mapping_size",
            "fourier_scale", "lambda_eikonal", "lambda_laplace",
        ])
        for net in ("siren", "sinet"):
            _, p = data[net]
            for t in p["full"]:
                params = t.params
                w.writerow([
                    net, t.number, t.user_attrs["generator_params"],
                    f"{N_VALID/t.user_attrs['generator_params']:.2f}",
                    f"{t.values[0]:.3f}",
                    f"{t.user_attrs.get('final_rmse_celsius',float('nan')):.4f}",
                    f"{t.user_attrs.get('final_mae_celsius',float('nan')):.4f}",
                    params.get("width", ""), params.get("n_critic", ""),
                    f"{params.get('lr_G', float('nan')):.2e}",
                    f"{params.get('lr_D', float('nan')):.2e}",
                    f"{params.get('lambda_adv', float('nan')):.2e}",
                    f"{params.get('grad_clip', float('nan')):.3f}",
                    params.get("n_iterations", ""),
                    params.get("sorting_group_size", ""),
                    params.get("fourier_mapping_size", ""),
                    f"{params.get('fourier_scale', '')}",
                    f"{params.get('lambda_eikonal', '')}",
                    f"{params.get('lambda_laplace', '')}",
                ])
    print(f"Zapisano CSV: {path}")


def print_summary(data):
    print("\n=== Podsumowanie liczbowe (gan) ===")
    for net in ("siren", "sinet"):
        a, p = data[net]
        print(f"\n{LABELS[net]}:")
        print(f"  ukończone próby:    {len(a['psnr'])}")
        print(f"  PSNR (min/med/max): {a['psnr'].min():.2f} / {np.median(a['psnr']):.2f} / {a['psnr'].max():.2f} dB")
        print(f"  CF   (min/med/max): {a['cf'].min():.1f}× / {np.median(a['cf']):.1f}× / {a['cf'].max():.1f}×")
        print(f"  RMSE (min/med/max): {a['rmse'].min():.3f} / {np.median(a['rmse']):.3f} / {a['rmse'].max():.3f} °C")
        print(f"  punktów Pareto:     {len(p['cf'])}")
        if len(p["cf"]):
            i = np.argmax(p["psnr"])
            print(f"  najlepsze PSNR:     {p['psnr'][i]:.2f} dB @ {p['cf'][i]:.1f}×  (trial #{p['trial'][i]}, {p['params'][i]} params)")
            i = np.argmax(p["cf"])
            print(f"  najsilniejsza komp: {p['cf'][i]:.1f}× @ PSNR={p['psnr'][i]:.2f} dB  (trial #{p['trial'][i]}, {p['params'][i]} params)")


def main():
    studies = {net: load(net) for net in ("siren", "sinet")}
    data = {net: trials_to_arrays(studies[net], net) for net in ("siren", "sinet")}
    print_summary(data)
    pareto_main(data)
    rate_distortion(data)
    lambda_sensitivity(data)
    lambda_vs_cf(data)
    hp_importance(studies)
    convergence_efficiency(data)
    width_cf(data)
    pareto_hp_table(data)
    write_pareto_csv(data)
    print(f"\nWszystkie wykresy zapisane w {PLOTS}/")


if __name__ == "__main__":
    main()
