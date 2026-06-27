#!/usr/bin/env python3
"""
VEP_DNA demo — joint-effect linear (surrogate) model + epistasis testing.

This is a small, self-contained demo of the *downstream analysis* layer of the
pVEP framework (the joint-effect Ridge model and F-test epistasis testing
described in docs/methods_epistasis.md and the manuscript Methods). It runs on a
tiny bundled, simulated dataset and requires only CPU (pandas / numpy / scikit-
learn / scipy) — no GPU, no model weights, no downloads.

NOTE ON HARDWARE: A CUDA-capable NVIDIA GPU is the *intended* hardware for the
full pVEP pipeline (DNA sequence-model inference with SpliceAI / Flashzoi, etc.;
see the "Demo" / "Getting started" sections of the README). This CPU-only demo
exercises the analysis that consumes those VEP scores so reviewers can verify
the code end-to-end on a normal desktop; for real work, use a GPU.

Run from the repository root:

    python demo/run_demo.py

Expected output: an interaction/epistasis table printed to stdout and written to
demo/output/, matching the reference files in demo/expected_output/.
"""
import os
import sys

import numpy as np
import pandas as pd

# Ensure repo root is importable (so `import src.*` works) when run from anywhere
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.analysis.attribution import (
    wtvariants_to_vep_linear_model,
    test_epistasis_across_models,
)

DEMO_DIR = os.path.join(REPO_ROOT, "demo")
DATA_DIR = os.path.join(DEMO_DIR, "data")
OUT_DIR = os.path.join(DEMO_DIR, "output")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # --- Load the small bundled, simulated dataset --------------------------
    Xwt = pd.read_csv(os.path.join(DATA_DIR, "Xwt.csv"), index_col=0)
    y_vep = pd.read_csv(os.path.join(DATA_DIR, "y_vep.csv"), index_col=0)
    print(f"Loaded Xwt {Xwt.shape} (haplotypes x WT variants) "
          f"and y_vep {y_vep.shape} (haplotypes x sites)")

    # --- Fit the joint-effect Ridge model with within-model epistasis -------
    result = wtvariants_to_vep_linear_model(
        Xwt=Xwt,
        y_vep=y_vep,
        model_type="ridge",
        alpha=1.0,
        random_state=42,
        test_epistasis=True,
        epistasis_pvalue_threshold=0.05,
        verbose=True,
    )
    interaction_df = result["interaction_df"].sort_values(
        ["wt_variant", "site"]
    ).reset_index(drop=True)
    epi = result.get("epistasis_results", {}) or {}

    # --- Cross-model epistasis (one model per site) -------------------------
    models_dict = {
        site: wtvariants_to_vep_linear_model(
            Xwt=Xwt, y_vep=y_vep[[site]], model_type="ridge", alpha=1.0,
            random_state=42, test_epistasis=False, verbose=False,
        )
        for site in y_vep.columns
    }
    cross = test_epistasis_across_models(
        models_dict=models_dict, Xwt_full=Xwt, y_vep_full=y_vep,
        model_type="ridge", epistasis_alpha=1.0,
        epistasis_pvalue_threshold=0.05, random_state=42, verbose=False,
    )
    epistasis_df = cross["epistasis_df"].sort_values(
        ["wt_variant", "site"]
    ).reset_index(drop=True)

    # --- Report + persist ---------------------------------------------------
    print("\n=== Joint-effect interaction table (head) ===")
    print(interaction_df.head(8).to_string(index=False))
    print(f"\nWithin-model epistasis: tested={epi.get('n_tested')} "
          f"epistatic={epi.get('n_epistatic')} "
          f"rate={epi.get('epistasis_rate')}")

    interaction_df.to_csv(os.path.join(OUT_DIR, "interaction_df.csv"), index=False)
    epistasis_df.to_csv(os.path.join(OUT_DIR, "epistasis_df.csv"), index=False)
    print(f"\nWrote demo/output/interaction_df.csv ({len(interaction_df)} rows) "
          f"and demo/output/epistasis_df.csv ({len(epistasis_df)} rows)")
    print("Compare against demo/expected_output/ to confirm a successful run.")


if __name__ == "__main__":
    main()
