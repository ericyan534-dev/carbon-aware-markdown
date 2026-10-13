# The Profit-Maximizing Discount Is Nearly Optimal at the Social Cost of Carbon

Code for the paper *The Profit-Maximizing Discount Is Nearly Optimal at the Social Cost of Carbon: Carbon-Aware Markdown Pricing of Near-Expiry Food* (IEEE SusTech 2027). It estimates the discount elasticity of fresh food on FreshRetailNet-50K, values every discounted sale by the production it displaces, and solves optimal markdown policies for profit, carbon priced at the social cost of carbon, and waste, reproducing every table, figure and number in the paper.

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
| Sec. III: IV elasticity 1.94 (s.e. 0.017), FE 1.51, category estimates (data) | `python scripts/estimate_elasticity.py` | `elasticity/` | 10 s |
| Sec. III: first-stage F = 18,832 (data) | `python scripts/first_stage_f.py` | `elasticity/first_stage_f.json` | 4 s |
| Sec. III: city×day 1.95, subcategory×day 1.96, leave-city-out 2.17 (data) | `python scripts/iv_robustness.py` | `elasticity/iv_robustness.json` | 25 s |
| Table II; Sec. V-B, V-D, V-E ranges (99.4–100%, 57.7–75.6%, $1,177–23,006/t, 21.8–26.2%, 60–121%) | `python scripts/run_experiments.py` | `main/headlines.json` | 4 s |
| Sec. IV-C, V-C: θ̄ 0.07–0.54, 2,080 states, 145,600 comparisons, Hill fit | `python scripts/alignment_compute.py && python scripts/alignment_analyze.py` | `alignment/summary.json` | 15 s |
| Sec. IV-D: monotone in stock, 120 cases and 10,000 randomized solves | `python scripts/stock_monotonicity.py` | `monotonicity/summary.json` | 5 s |
| Sec. V-D: two-level rule 99.4–99.7% of optimal profit | `python scripts/two_level_rule.py` | `rule/summary.json` | 5 s |
| Sec. V-D, V-E, V-F: 4,121 profiles (rule ≥98.4%, median 99.99%; ≥98.5% at SCC; profit-max median 99.9%, chain 98.4%; overstatement median 89%; ~3 ms per product) | `python scripts/profiles_calibrate.py` (data), then `python scripts/profiles_run.py all`, `python scripts/profiles_run.py bench_serial`, `python scripts/profiles_summarize.py` | `profiles/headlines.json` | 1 min |
| Sec. V-F: batch size 1.25–3× and SCC $120–340/t | `python scripts/sensitivity_batch_scc.py` | `sensitivity/sensitivity_summary.json` | 20 s |
| Sec. V-F: elasticity uncertainty (500 draws) | `python scripts/elasticity_uncertainty.py` | `elasticity_uncertainty/headline_bands.csv` | 10 s |
| Sec. V-F: 1,260 household cases | `python scripts/household_grid.py` | `household/` | 25 s |
| Sec. V-F: strategic waiting and re-spending | `python scripts/strategic_waiting.py && python scripts/respending.py` | `consumer/` | 20 s |
| Sec. V-F: 36 switching-share pairs ≥95.7%, split decay rates ≥99.2% | `python scripts/robustness_switching.py` | `switching/switching.json` | 6 s |
| Sec. V-F: Sobol analysis, 6,144 evaluations | `python scripts/run_sobol.py` | `sobol/` | 40 s |
| Figs. 1–5 (`teaser2`, `elasticity_col`, `frontier`, `why`, `policy_maps`) | `python scripts/make_paper_figures.py` | `figures/` | 10 s |

`make_paper_figures.py` uses the outputs of `estimate_elasticity.py` (Fig. 2), `run_experiments.py` (Fig. 3) and the alignment scripts (Fig. 4); it also reports the onset of profitable discounting (ε = 1.27).

## Layout

- `tblca/`: model library (demand and displacement, markdown dynamic program, fast solver, threshold rules, elasticity estimation, alignment index)
- `scripts/`: one script per result above
- `tests/`: unit tests

## Citation

```bibtex
@inproceedings{carbonmarkdown2027,
  title     = {The Profit-Maximizing Discount Is Nearly Optimal at the Social Cost of Carbon: Carbon-Aware Markdown Pricing of Near-Expiry Food},
  author    = {Author One and Author Two and Author Three},
  booktitle = {IEEE Conference on Technologies for Sustainability (SusTech)},
  year      = {2027}
}
```

## License

Code: MIT (see `LICENSE`). Data: FreshRetailNet-50K, CC BY 4.0.
