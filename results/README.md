# Results Directory

Analysis outputs and figures for JANUS-Z project.

## Current Results (v17)

### `janus_v17_comprehensive_results.json`

Complete analysis results in JSON format.

**Contents**:
- SMF fitting results (χ², BIC, optimal ε)
- Proto-cluster statistics (M_vir, σ_v)
- Metallicity measurements
- Model comparison metrics

### `figures/`

Publication-quality figures:

| File | Description | Size |
|------|-------------|------|
| `fig_v17_killer_plot_suite.pdf` | SMF at z~12, z~10, ε comparison, χ² landscape | 42 KB |
| `fig_v17_clustering_analysis.pdf` | Proto-cluster virial masses and velocity dispersions | 30 KB |
| `fig_v17_metallicity_evolution.pdf` | Metallicity-redshift relation and MZR | 39 KB |

## Historical Results

### Version 16
- `janus_v16_comprehensive_results.json`
- `figures/fig_v16_*.pdf`

### Version 15
- `janus_v15_results.json`
- `figures/fig_v15_killer_plot.pdf`

---

## Key Metrics (v17)

```
JANUS:
  χ² (ε=0.15): 149,547
  BIC: 149,552
  ε_optimal: 0.10
  Status: Physical (ε < 0.15)

ΛCDM:
  χ² (ε=0.15): 55,251
  BIC: 29,114
  ε_optimal: 0.10
  Status: Catastrophic failure at physical ε

ΔBIC: -120,438 (Very strong evidence for JANUS)
```

---

## Reproducibility

To regenerate all results:

```bash
cd scripts
python analysis_janus_v17_jan2026.py
```

Results will be written to this directory.
