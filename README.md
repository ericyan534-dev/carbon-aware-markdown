# Saving Food Is Not Saving Carbon

Code for the paper *Saving Food Is Not Saving Carbon: Why Profit-Maximizing Discounts Beat Zero-Waste Targets* (IEEE SusTech 2027). It estimates the discount elasticity of fresh food on FreshRetailNet-50K, values every discounted sale by the production it displaces, and solves optimal markdown policies for profit, carbon priced at the social cost of carbon, and waste, reproducing every table, figure and number in the paper.

## Setup

Python 3.13:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
```

## Data

FreshRetailNet-50K (CC BY 4.0) is on Hugging Face at [Dingdong-Inc/FreshRetailNet-50K](https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K). Place the train split at `data/raw/train.parquet`:

```bash
mkdir -p data/raw
curl -L -o data/raw/train.parquet \
  https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K/resolve/main/data/train.parquet
```

SHA-256: `6706832db892bbae4969c19d87e07975d2543d2ba7d7d4756360654785de5a3d`. Only the scripts marked "data" below read it.

## Reproducing the paper

Run from the repository root. Results go to `outputs/`; figures go to `outputs/figures/`. Times are wall-clock on a 14-core laptop.

| Paper result | Command | Output | Time |
|---|---|---|---|
| Sec. II-C: IV elasticity 1.94 (s.e. 0.017), FE 1.51, category estimates of Fig. 2 (data) | `python scripts/estimate_elasticity.py` | `elasticity/` | 10 s |
| Sec. II-C: first-stage F = 18,832 (data) | `python scripts/first_stage_f.py` | `elasticity/first_stage_f.json` | 4 s |
| Sec. II-C: city×day 1.95, subcategory×day 1.96, leave-city-out 2.17 (data) | `python scripts/iv_robustness.py` | `elasticity/iv_robustness.json` | 25 s |
| Sec. II-C: ten flexible demand curves (five shapes × two instruments) match the displaced share within 0.03 at 30% and 50% off (data) | `python scripts/demand_shape.py` | `demand_shape/summary.json` | 80 s |
| Table II; Figs. 1c, 3; Sec. III-B, V-B, V-D ranges (60–121%, 99.4–100%, 57.7–75.6%, $1,177–23,006/t, 21.8–26.2%) | `python scripts/run_experiments.py` | `main/headlines.json`, `main/goals.csv`, `main/pareto.csv` | 12 s |
| Sec. III-B: displacement adds 8–20 pp to the static-rule overstatement on top of freshness and household consumption | `python scripts/robustness_switching.py` | `switching/ablation_by_case.csv` | 6 s |
| Sec. IV-B, V-C: ε* = 1.25, profit-max. policy starts discounting at ε = 1.27, ε*_θ = 1.10–1.23, model curves of Fig. 4b | `python scripts/threshold_curves.py` | `threshold/theory_meta.json` | 3 s |
| Sec. IV-C, V-C: 2,080 states, 145,600 comparisons, θ̄ 0.07–0.54, Hill fit | `python scripts/alignment_compute.py && python scripts/alignment_analyze.py` | `alignment/summary.json` | 15 s |
| Sec. V-C: store-half prediction of profiles below 95% of the best value, AUC 0.97 (mean of 20 splits) (data) | `python scripts/threshold_prediction.py` | `threshold/prediction_summary.json` | 80 s |
| Sec. V-D: monotone in stock, 120 cases and 10,000 randomized solves | `python scripts/stock_monotonicity.py` | `monotonicity/summary.json` | 5 s |
| Sec. V-D: two-level rule 99.4–99.7% of optimal profit | `python scripts/two_level_rule.py` | `rule/summary.json` | 5 s |
| Sec. III-B, V-D, V-E: 4,121 profiles (rule ≥98.4%, median 99.99%; ≥98.5% at SCC; profit-max. median 99.9%, chain 98.4%; overstatement median 89%; ~3 ms per product) | `python scripts/profiles_calibrate.py` (data), then `python scripts/profiles_run.py all`, `python scripts/profiles_run.py bench_serial`, `python scripts/profiles_summarize.py` | `profiles/headlines.json` | 1 min |
| Sec. V-E: batch size 1.25–3× and SCC $120–340/t | `python scripts/sensitivity_batch_scc.py` | `sensitivity/sensitivity_summary.json` | 20 s |
| Sec. V-E: elasticity uncertainty (500 draws) | `python scripts/elasticity_uncertainty.py` | `elasticity_uncertainty/headline_bands.csv` | 10 s |
| Sec. V-E: 1,260 household cases | `python scripts/household_grid.py` | `household/` | 25 s |
| Sec. V-E: strategic waiting and re-spending | `python scripts/strategic_waiting.py && python scripts/respending.py` | `consumer/` | 20 s |
| Sec. V-E: 36 switching-share pairs ≥95.7%, split decay rates ≥99.2% | `python scripts/robustness_switching.py` | `switching/switching.json` | 6 s |
| Sec. V-E: Sobol analysis, 6,144 evaluations | `python scripts/run_sobol.py` | `sobol/` | 40 s |
| Figs. 1–5 (`teaser`, `elasticity_col`, `goals2`, `why4`, `policy_maps`) | `python scripts/make_paper_figures.py` | `figures/` | 7 s |

`make_paper_figures.py` uses the outputs of `run_experiments.py` (Figs. 1 and 3), `estimate_elasticity.py` (Fig. 2) and `threshold_curves.py`, the profile scripts and the alignment scripts (Fig. 4). `threshold_prediction.py` uses the outputs of `profiles_calibrate.py` and `threshold_curves.py`.

## Layout

- `tblca/`: model library (demand and displacement, markdown dynamic program, fast solver, threshold rules, elasticity estimation, flexible demand curves, alignment index)
- `scripts/`: one script per result above
- `tests/`: unit tests

## Citation

```bibtex
@inproceedings{carbonmarkdown2027,
  title     = {Saving Food Is Not Saving Carbon: Why Profit-Maximizing Discounts Beat Zero-Waste Targets},
  author    = {Author One and Author Two and Author Three},
  booktitle = {IEEE Conference on Technologies for Sustainability (SusTech)},
  year      = {2027}
}
```

## License

Code: MIT (see `LICENSE`). Data: FreshRetailNet-50K, CC BY 4.0.
