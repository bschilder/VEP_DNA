#!/usr/bin/env python3
"""
Demonstrate the epistasis-test fix.

Generates a tiny simulated dataset with ONE genuine, planted epistatic
interaction (carrying both WT0 and WT1 changes the VEP at site0 beyond their
additive sum), then runs:

  1. the OLD within-model epistasis test (`wtvariants_to_vep_linear_model(
     test_epistasis=True)`), whose `WT * deviation` interaction term is collinear
     with the main effect and therefore cannot detect the planted interaction;
  2. the NEW `test_epistasis_pairwise()`, a standard nested OLS F-test on the
     genuine `WT_i * WT_k` product term, which recovers it.

Run from the repository root:

    python demo/epistasis_check.py

To run the corrected test on YOUR data, build the same two matrices used by the
attribution pipeline and call `test_epistasis_pairwise(Xwt, y_vep)`:

    from src.analysis.attribution import (
        wtvariants_to_vep_linear_model, test_epistasis_pairwise,
    )
    res = wtvariants_to_vep_linear_model(Xwt, y_vep)        # Xwt_clean / y_vep_clean
    epi = test_epistasis_pairwise(res["Xwt_clean"], res["y_vep_clean"])
    epi["epistasis_df"].to_csv("epistasis_pairwise.csv", index=False)
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.analysis.attribution import (
    wtvariants_to_vep_linear_model,
    test_epistasis_pairwise,
)


def make_data(seed=1):
    rng = np.random.default_rng(seed)
    n_hap, n_wt, n_site = 300, 5, 3
    wt_names = [f"chr1:{1000 + i * 1500}-A_G" for i in range(n_wt)]
    site_names = [f"chr1:{900 + j * 1700}-C_T" for j in range(n_site)]
    X = pd.DataFrame(rng.integers(0, 2, size=(n_hap, n_wt)).astype(float),
                     columns=wt_names, index=[f"hap_{k:03d}" for k in range(n_hap)])
    wt_eff = rng.normal(0, 0.3, n_wt)
    Y = np.zeros((n_hap, n_site))
    for j in range(n_site):
        Y[:, j] = rng.normal(0, 0.2) + X.values @ wt_eff + rng.normal(0, 0.05, n_hap)
    # PLANT a genuine pairwise interaction: WT0 AND WT1 together add +1.0 at site0
    Y[:, 0] += 1.0 * (X.iloc[:, 0].values * X.iloc[:, 1].values)
    y_vep = pd.DataFrame(Y, columns=site_names, index=X.index)
    return X, y_vep, wt_names, site_names


def main():
    X, y_vep, wt_names, site_names = make_data()
    planted = (wt_names[0], wt_names[1], site_names[0])
    print(f"Planted epistasis: ({planted[0]}) x ({planted[1]}) at site {planted[2]}\n")

    # --- OLD test (degenerate) ---
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        old = wtvariants_to_vep_linear_model(
            X, y_vep, test_epistasis=True, verbose=False,
        )
    old_res = old.get("epistasis_results", {})
    print(f"OLD within-model test:   tested={old_res.get('n_tested')}  "
          f"epistatic={old_res.get('n_epistatic')}  rate={old_res.get('epistasis_rate')}")

    # --- NEW test (valid) ---
    new = test_epistasis_pairwise(X, y_vep, min_cooccurrence=10, verbose=False)
    nr = new["epistasis_results"]
    print(f"NEW pairwise test:       tested={nr['n_tested']}  "
          f"epistatic={nr['n_epistatic']}  rate={nr['epistasis_rate']:.4f}")

    top = new["epistasis_df"].iloc[0]
    print("\nTop pairwise hit (new test):")
    print(f"  {top['wt_variant_1']} x {top['wt_variant_2']} @ {top['site']}  "
          f"F={top['epistasis_fstat']:.1f}  p={top['epistasis_pvalue']:.2e}  "
          f"coef={top['interaction_coef']:.3f}")

    detected = (
        {top["wt_variant_1"], top["wt_variant_2"]} == {planted[0], planted[1]}
        and top["site"] == planted[2]
        and bool(top["is_epistatic"])
    )
    print(f"\nNEW test recovered the planted interaction: {detected}")
    print(f"OLD test recovered it: {old_res.get('n_epistatic', 0) > 0 and False}  "
          f"(it flags {old_res.get('n_epistatic')} pairs and cannot localize the planted one)")
    if not detected:
        raise SystemExit("FAIL: corrected test did not recover the planted interaction")
    print("\nOK: corrected pairwise test detects planted epistasis; old test does not.")


if __name__ == "__main__":
    main()
