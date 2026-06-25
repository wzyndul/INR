# INR / INR-GAN compression of E-OBS climate fields

Code for the master thesis on lossy compression of climate temperature fields
with implicit neural representations. Two generator architectures (SIREN and
SiNET) compared in two training variants (with PatchGAN discriminator and
without).

## Layout

```
new/
├── data_preprocessed.nc            input field (E-OBS, single timestep)
├── requirements.txt
├── .env                            COMET_API_KEY (optional)
├── README.md
│
├── siren_inr_gan/train.py          SIREN generator + training loop
├── sinet_inr_gan/train.py          SiNET generator + training loop
│
├── hyperopt/                       NSGA-II hyperparameter search
│   ├── bayes_opt_gan.py
│   ├── bayes_opt_no_gan.py
│   ├── run_bayes_opt_gan.sh
│   ├── run_bayes_opt_no_gan.sh
│   └── outputs_{tag}/{network}/    per-trial training outputs
│
├── seed_sweep/                     50-seed reproducibility check
│   ├── extract.py                  Pareto front -> task list
│   ├── run.py                      single (config, seed) job
│   ├── aggregate.py                per-config mean / std / range
│   ├── run_gan.sh
│   ├── run_no_gan.sh
│   ├── aggregate_gan.sh
│   ├── aggregate_no_gan.sh
│   ├── tasks_{tag}.json            task list (built by extract.py)
│   ├── outputs_{tag}/              per-task result.json files
│   ├── runs_{tag}.csv              flat record per (config, seed)
│   └── summary_{tag}.csv           per-config aggregate
│
├── analysis/                       plots & tables
│   ├── analyze_gan.py
│   ├── analyze_no_gan.py
│   ├── pareto_combined.py
│   ├── plots_gan/                  per-study figures, CSV
│   └── plots_no_gan/
│
└── optuna_{network}_{tag}.db       Optuna study databases at repo root
```

`{tag}` is `gan` or `no_gan`. `{network}` is `siren` or `sinet`.

Paths are anchored to the repo root via `ROOT = parent(HERE)` in every script,
so calls work regardless of working directory. SLURM wrappers compute the
repo root from their own location too.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Place `data_preprocessed.nc` in the repo root.

## Workflow

### 1. Hyperparameter optimization (NSGA-II)

```bash
sbatch hyperopt/run_bayes_opt_gan.sh    siren     # SIREN INR-GAN
sbatch hyperopt/run_bayes_opt_gan.sh    sinet     # SiNET INR-GAN
sbatch hyperopt/run_bayes_opt_no_gan.sh siren     # SIREN INR
sbatch hyperopt/run_bayes_opt_no_gan.sh sinet     # SiNET INR
```

100 trials each by default, `n_iterations` up to 30 000. Writes:

- `optuna_{network}_{tag}.db`           Optuna study (at repo root)
- `hyperopt/outputs_{tag}/{network}/`   per-trial generator weights

When finished, generate plots and tables:

```bash
python analysis/analyze_gan.py
python analysis/analyze_no_gan.py
python analysis/pareto_combined.py
```

Outputs land in `analysis/plots_{tag}/` and the combined Pareto plot in `analysis/pareto_combined.png`.

### 2. Consistency check (50-seed sweep)

```bash
python seed_sweep/extract.py --tag gan    --n_seeds 50
python seed_sweep/extract.py --tag no_gan --n_seeds 50
```

Each call prints the SLURM array range. Update `--array=0-N%K` in the matching
`seed_sweep/run_*.sh` if the printed `N` differs from the default.

```bash
JID=$(sbatch seed_sweep/run_gan.sh    | awk '{print $4}')
sbatch --dependency=afterany:$JID seed_sweep/aggregate_gan.sh

JID=$(sbatch seed_sweep/run_no_gan.sh | awk '{print $4}')
sbatch --dependency=afterany:$JID seed_sweep/aggregate_no_gan.sh
```

Aggregation writes `seed_sweep/summary_{tag}.csv` and `seed_sweep/runs_{tag}.csv`.

## Storage layout

| variant | tag      | study name                            | DB                            |
| ------- | -------- | ------------------------------------- | ----------------------------- |
| INR-GAN | `gan`    | `{network}_compression_pareto_gan`    | `optuna_{network}_gan.db`     |
| INR     | `no_gan` | `{network}_compression_pareto_no_gan` | `optuna_{network}_no_gan.db`  |
