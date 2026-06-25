import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT  = os.path.join(HERE, "pareto_combined.png")


def load(csv_path):
    rows = {"siren": [], "sinet": []}
    with open(csv_path) as f:
        r = csv.DictReader(f)
        for row in r:
            rows[row["network"]].append((float(row["cf_x"]),
                                         float(row["psnr_db"])))
    for net in rows:
        rows[net].sort(key=lambda p: p[0])
    return rows


def main():
    gan = load(os.path.join(HERE, "plots_gan", "pareto_table_gan.csv"))
    inr = load(os.path.join(HERE, "plots_no_gan", "pareto_table_no_gan.csv"))

    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    style = dict(linewidth=1.6, markersize=7, markeredgecolor="black",
                 markeredgewidth=0.5)

    siren_gan = gan["siren"]
    siren_inr = inr["siren"]
    sinet_gan = gan["sinet"]
    sinet_inr = inr["sinet"]

    ax.plot([p[0] for p in siren_inr], [p[1] for p in siren_inr],
            "o-",  color="#1f4ea8", label="SIREN INR",     **style)
    ax.plot([p[0] for p in siren_gan], [p[1] for p in siren_gan],
            "s--", color="#e67e22", label="SIREN INR-GAN", **style)
    ax.plot([p[0] for p in sinet_inr], [p[1] for p in sinet_inr],
            "o-",  color="#8e44ad", label="SiNET INR",     **style)
    ax.plot([p[0] for p in sinet_gan], [p[1] for p in sinet_gan],
            "s--", color="#16a085", label="SiNET INR-GAN", **style)

    ax.set_xlim(0, 200)
    ax.xaxis.set_major_locator(MultipleLocator(25))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"×{int(v)}"))
    ax.set_xlabel("Wielkość kompresji", fontsize=11)
    ax.set_ylabel("PSNR [dB]", fontsize=11)
    ax.grid(True, which="both", alpha=0.3, linestyle=":")
    ax.legend(loc="lower left", fontsize=10, framealpha=0.9)
    ax.tick_params(labelsize=10)

    plt.tight_layout()
    plt.savefig(OUT, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Zapisano: {OUT}")


if __name__ == "__main__":
    main()
