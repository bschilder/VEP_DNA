# Comparison of Epistasis Testing Approaches: Assumptions and Differences

## Overview

There are three main approaches for epistasis testing, each with different assumptions about how WT variants and clinical variants interact:

1. **All-in-one model (multi-target approach)**: Single model with all WT variants and all clinical variants
2. **Within-model approach**: Separate models per clinical variant, epistasis computed within each model
3. **Across-models approach**: Separate models per clinical variant, epistasis uses cross-model WT effects

---

## 1. All-in-One Model (Multi-Target Approach)

### Model Structure
- **Single model**: `Y = Xβ + ε` where:
  - `X`: (haplotypes × all WT variants) - binary matrix
  - `Y`: (haplotypes × all clinical variants) - VEP scores
  - `β`: (all WT variants × all clinical variants) - coefficient matrix

### Key Assumptions
1. **No genomic windowing**: All WT variants can potentially affect all clinical variants, regardless of genomic distance
2. **Shared context**: WT variant effects are estimated in the context of all clinical variants simultaneously
3. **Clinical variant effects**: Computed relative to a global baseline across all haplotypes and all clinical variants
4. **Expected additive effect**: `Effect_additive = Effect_WT + Effect_Clinical` (both from same model)

### WT Individual Effect Computation
```
Effect_WT,i = (1/|S|) Σ_s [VEP(WT_i=1, s) - VEP(WT_i=0, s)]
```
where `S` = set of all clinical variant sites **in the same model**

### Clinical Individual Effect Computation
```
Effect_Clinical,j = mean(VEP for clinical variant j) - mean(VEP across all haplotypes and all clinical variants)
```

### Expected Additive Effect
```
Effect_additive = Effect_WT,i + Effect_Clinical,j
```

### When to Use
- When you want to model all possible WT-clinical variant interactions
- When genomic distance is not a primary concern
- When you have sufficient data to fit a large model
- When you want to capture global patterns across the entire gene/region

### Limitations
- May be computationally expensive with many variants
- Assumes all WT variants can affect all clinical variants (may not be biologically realistic for splicing)
- Can suffer from overfitting with many features

---

## 2. Within-Model Approach

### Model Structure
- **Separate models per clinical variant**: `Y_j = X_j β_j + ε_j` where:
  - `X_j`: (haplotypes × WT variants within window j) - only WT variants within genomic window
  - `Y_j`: (haplotypes × 1) - VEP scores for clinical variant j only (**SINGLE TARGET**)
  - `β_j`: (WT variants in window j × 1) - coefficients for clinical variant j

### Key Assumptions
1. **Genomic windowing**: Each clinical variant is modeled with only nearby WT variants (e.g., ±5kb)
2. **Single target per model**: Each model predicts only ONE clinical variant
3. **Context-specific WT effects**: WT variant effects are estimated from the **single clinical variant** in that model
4. **Clinical variant effect is constant**: Since we're testing one clinical variant at a time, the clinical variant effect is constant across all haplotypes (absorbed in intercept)
5. **Expected additive effect**: `Effect_additive = Effect_WT,i` (clinical effect is constant, so only WT effect matters)

### WT Individual Effect Computation
**CRITICAL**: Since each model has only ONE clinical variant, the WT effect is computed from that single clinical variant:
```
Effect_WT,i = VEP(WT_i=1, clinical_j) - VEP(WT_i=0, clinical_j)
```
where `clinical_j` is the **single clinical variant** in model j

**Note**: If the model actually has multiple clinical variants (multi-target), then:
```
Effect_WT,i = (1/|S_j|) Σ_s∈S_j [VEP(WT_i=1, s) - VEP(WT_i=0, s)]
```
where `S_j` = set of all clinical variant sites **in model j**

### Clinical Individual Effect Computation
**CRITICAL**: Since each model has only ONE clinical variant:
- The clinical variant effect is **constant** across all haplotypes (the variant is the same for all)
- It is **absorbed into the intercept** of the model
- Therefore: `Effect_Clinical,j = 0` (or more precisely, it doesn't contribute to the expected additive effect)

If the model has multiple clinical variants:
```
Effect_Clinical,j = mean(VEP for clinical variant j) - mean(VEP across all haplotypes and all clinical variants in model j)
```

### Expected Additive Effect
**When single target per model** (typical window-based approach):
```
Effect_additive = Effect_WT,i
```
(Clinical effect is constant/absorbed, so only WT effect matters)

**If multiple targets per model** (rare, but possible):
```
Effect_additive = Effect_WT,i + Effect_Clinical,j
```

### When to Use
- When genomic proximity matters (e.g., splicing where local sequence context is important)
- When you want to test epistasis using WT effects estimated from the **same clinical variant** (not averaged across different clinical variants)
- When you want to avoid assumptions about cross-context consistency
- **Note**: If each model has only one clinical variant, this approach uses WT effects from that single variant only (may have less statistical power)

### Important Distinction from Across-Models
The key difference is **where WT effects come from**:
- **Within-model**: WT effect computed from the single clinical variant in that model
- **Across-models**: WT effect computed by averaging across all models (different clinical variants, different windows)

### Limitations
- **If single target per model**: WT effects computed from only one clinical variant (less robust than averaging across multiple variants)
- **If single target per model**: Clinical effect is constant (absorbed in intercept), so expected additive = WT effect only
- Cannot leverage information from WT effects in other genomic contexts or other clinical variants
- May have limited statistical power compared to across-models approach

---

## 3. Across-Models Approach

### Model Structure
- **Separate models per clinical variant**: `Y_j = X_j β_j + ε_j` where:
  - `X_j`: (haplotypes × WT variants within window j) - only WT variants within genomic window
  - `Y_j`: (haplotypes × 1) - VEP scores for clinical variant j only (**SINGLE TARGET**)
  - `β_j`: (WT variants in window j × 1) - coefficients for clinical variant j

### Key Assumptions
1. **Genomic windowing**: Each clinical variant is modeled with only nearby WT variants (e.g., ±5kb)
2. **Single target per model**: Each model predicts only ONE clinical variant
3. **Cross-context WT effects**: WT variant effects are estimated by averaging across **all models** where the WT variant appears, even if those models have different genomic windows and different clinical variants
4. **Clinical variant effect is constant**: When testing one clinical variant at a time, the clinical variant effect is constant across all haplotypes (absorbed into intercept)
5. **Expected additive effect**: `Effect_additive = Effect_WT,i` (clinical effect is constant, so only WT effect matters)

### WT Individual Effect Computation
```
Effect_WT,i = (1/|M_i|) Σ_m∈M_i [VEP(WT_i=1, m) - VEP(WT_i=0, m)]
```
where `M_i` = set of **all clinical variant models** (indexed by m) that include WT variant i in their genomic window

**Key difference**: WT effects are computed across different genomic contexts/windows

### Clinical Individual Effect Computation
- **Not computed separately**: Clinical variant effect is constant (absorbed in intercept) when testing one clinical variant at a time
- The clinical variant is the same for all haplotypes being tested

### Expected Additive Effect
```
Effect_additive = Effect_WT,i
```
(Clinical variant effect is constant, so it doesn't contribute to the expected additive effect)

### When to Use
- When each clinical variant is modeled separately (single target per model)
- When you want to leverage WT effect estimates from multiple genomic contexts
- When you want more robust WT effect estimates by averaging across windows
- When genomic windows overlap and WT variants appear in multiple models

### Limitations
- Assumes WT effects are consistent across different genomic contexts (may not hold if context matters)
- Clinical variant effects are not explicitly modeled (absorbed in intercept)
- May be less sensitive to context-specific epistasis

---

## Critical Insight: Single Target Per Model

**IMPORTANT**: In the window-based approaches (both within-model and across-models), each model typically has only **ONE clinical variant** (single target regression). This fundamentally changes how epistasis testing works:

### Implications for Within-Model Approach
When each model has only one clinical variant:
- **WT effect**: Computed from that **single clinical variant** only
  - `Effect_WT,i = VEP(WT_i=1, clinical_j) - VEP(WT_i=0, clinical_j)`
- **Clinical effect**: Constant across all haplotypes (absorbed in intercept) → `Effect_Clinical,j = 0`
- **Expected additive**: `Effect_additive = Effect_WT,i` (only WT effect matters)

### Implications for Across-Models Approach  
When each model has only one clinical variant:
- **WT effect**: Averaged across **all models** where WT appears (different clinical variants, different windows)
  - `Effect_WT,i = (1/|M_i|) Σ_m [VEP(WT_i=1, m) - VEP(WT_i=0, m)]` where m indexes different clinical variants
- **Clinical effect**: Constant (absorbed in intercept) → `Effect_Clinical,j = 0`
- **Expected additive**: `Effect_additive = Effect_WT,i` (only WT effect matters)

### The Key Difference
Even though both approaches have `Effect_additive = Effect_WT,i`, the **source of the WT effect** differs:
- **Within-model**: WT effect from the **same clinical variant** being tested
- **Across-models**: WT effect averaged across **different clinical variants**

This means:
- **Within-model** tests: "Does the joint effect deviate from the WT's effect on this specific clinical variant?"
- **Across-models** tests: "Does the joint effect deviate from the WT's average effect across all clinical variants?"

## Key Differences Summary

| Aspect | All-in-One | Within-Model | Across-Models |
|--------|-----------|--------------|---------------|
| **Model structure** | Single model, all variants | Separate models, windowed | Separate models, windowed |
| **Targets per model** | Multiple clinical variants | **Single clinical variant** | **Single clinical variant** |
| **WT effect context** | All clinical variants in same model | **Single clinical variant in same model** | All models where WT appears (different clinical variants) |
| **Clinical effect** | Relative to global baseline | **Constant (absorbed in intercept)** | Constant (absorbed in intercept) |
| **Expected additive** | WT + Clinical | **WT only** (if single target) | WT only |
| **Genomic distance** | Not considered | Considered (windowing) | Considered (windowing) |
| **Context dependency** | Shared global context | Context-specific to single variant | Assumes cross-context consistency |

---

## Statistical Implications

### All-in-One Model
- **Pros**: Maximum information sharing, can detect global patterns
- **Cons**: May violate assumptions if genomic distance matters, computationally intensive

### Within-Model
- **Pros**: Respects genomic context, no cross-context assumptions, WT effects from same clinical variant
- **Cons**: If single target per model, WT effects computed from only one clinical variant (less robust, may have less power), clinical effect is constant so doesn't contribute to expected additive

### Across-Models
- **Pros**: More robust WT effect estimates, leverages all available data
- **Cons**: Assumes WT effects are context-independent (may not hold), clinical effects not explicitly modeled

---

## Choosing the Right Approach

1. **Use All-in-One** if:
   - You want to model all possible interactions
   - Genomic distance is not a primary concern
   - You have sufficient data and computational resources

2. **Use Within-Model** if:
   - Genomic proximity matters (e.g., splicing)
   - You want to avoid assumptions about cross-context consistency
   - You want WT effects estimated from the **same clinical variant** being tested (not averaged across different clinical variants)
   - **Note**: If each model has only one clinical variant, WT effects come from that single variant only (less robust than across-models)

3. **Use Across-Models** if:
   - Each clinical variant is modeled separately
   - You want robust WT effect estimates
   - You're willing to assume WT effects are consistent across contexts
   - Windows overlap and WT variants appear in multiple models

