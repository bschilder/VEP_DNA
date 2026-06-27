# VEP_DNA demo

A small, self-contained demo of the pVEP **downstream analysis** layer: the
joint-effect Ridge (surrogate) model and epistasis testing described in
[`docs/methods_epistasis.md`](../docs/methods_epistasis.md) and the manuscript
Methods.

> ### ⚠️ Hardware
> A **CUDA-capable NVIDIA GPU is the intended hardware** for pVEP. The full
> pipeline (DNA sequence-model inference with SpliceAI, Flashzoi/Borzoi, etc.;
> see the main [README](../README.md) "Demo" and "Getting started" sections)
> should be run on a GPU — we **strongly recommend** it for any real use.
> This CPU-only demo exists *solely* so that a reviewer can verify the analysis
> code end-to-end on a normal desktop without a GPU or model weights. It is not
> representative of production runtimes.

## What it does

Runs the surrogate model on a tiny **bundled, simulated** dataset
(`data/Xwt.csv`, `data/y_vep.csv` — 120 haplotypes × 6 WT variants × 4 sites):

1. Fits a multi-target Ridge model relating wild-type (WT) variant presence to
   per-site VEP scores (`wtvariants_to_vep_linear_model`).
2. Computes the joint-effect interaction table and per-pair epistasis statistics
   (F-test, additive vs. interaction models).
3. Runs cross-model epistasis testing (`test_epistasis_across_models`).

## Requirements

- Python 3.10–3.12
- `pandas`, `numpy`, `scikit-learn`, `scipy` (all in the base `conda/conda.yml`
  environment; **no GPU, no model weights, no network access**)

## Run it

From the repository root:

```bash
python demo/run_demo.py
```

## Expected output

- **Runtime:** < 5 seconds on a normal desktop CPU.
- **Console:** the loaded data shapes, a progress bar, the head of the
  joint-effect interaction table, and an epistasis summary line.
- **Files:** `demo/output/interaction_df.csv` (24 rows: every WT-variant × site
  pair, with signed/absolute interaction strength, position distance, and
  epistasis statistics) and `demo/output/epistasis_df.csv`.

Reference copies are committed in [`expected_output/`](expected_output/); the
run is deterministic (`random_state=42`), so your `output/` files should match
them. On this simulated additive dataset the surrogate recovers the planted
structure and **no pairs are flagged epistatic at p < 0.05** — that is the
expected result here; the demo verifies the code path, not a specific biological
finding.

## Next step

To run the actual VEP models on real data (GPU required), follow the
[main README](../README.md): create a model environment from `conda/`, then use
`notebooks/vep_dna_pipeline.ipynb`.
