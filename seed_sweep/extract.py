import argparse
import json
import os

import optuna

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="gan", choices=["gan", "no_gan"])
    ap.add_argument("--n_seeds", type=int, default=50)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = args.out or os.path.join(HERE, f"tasks_{args.tag}.json")

    tasks = []
    task_id = 0
    for net in ("siren", "sinet"):
        study = optuna.load_study(
            study_name=f"{net}_compression_pareto_{args.tag}",
            storage=f"sqlite:///{os.path.join(ROOT, f'optuna_{net}_{args.tag}.db')}",
        )
        pareto = sorted(study.best_trials, key=lambda t: t.values[1])
        seen = set()
        unique = []
        for t in pareto:
            key = tuple(t.values)
            if key not in seen:
                seen.add(key)
                unique.append(t)
        for cfg_idx, t in enumerate(unique):
            for seed in range(args.n_seeds):
                tasks.append({
                    "task_id": task_id,
                    "network": net,
                    "config_idx": cfg_idx,
                    "seed": seed,
                    "source_trial": t.number,
                    "params": dict(t.params),
                })
                task_id += 1

    with open(out, "w") as f:
        json.dump({"n_seeds": args.n_seeds, "tag": args.tag, "tasks": tasks}, f, indent=2)

    print(f"{len(tasks)} zadan -> {out}")
    print(f"--array=0-{len(tasks) - 1}")


if __name__ == "__main__":
    main()
