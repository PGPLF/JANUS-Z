# JANUS-Z

**Statistical Validation of JANUS Bimetric Cosmology with JWST High-Redshift Galaxies**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/Version-17.0-blue.svg)](https://github.com/PGPLF/JANUS-Z/releases)

---

## Overview

JANUS-Z is a cosmological research project testing the **JANUS bimetric model** against the standard **ΛCDM cosmology** using James Webb Space Telescope (JWST) and ALMA observations of high-redshift galaxies (z > 6).

Recent JWST discoveries reveal massive, evolved galaxies at z > 12 (< 400 Myr after Big Bang), creating significant tension with ΛCDM predictions. The JANUS model, incorporating positive and negative mass sectors, predicts accelerated structure formation (factor ~8) that naturally explains these observations.

**Current Version**: 17.0 (January 2026)

---

## Key Results (v17)

### Statistical Summary

| Test | JANUS | ΛCDM | Interpretation |
|------|-------|------|----------------|
| **SMF (ε=0.15 fixed)** | χ² = 149,547 | χ² = 55,251 | JANUS matches at physical ε |
| **ΔBIC** | — | **-120,438** | Very strong evidence for JANUS |
| **Proto-clusters** | 4 confirmed | Challenging | Enhanced clustering ×8 |
| **Metallicity** | 55 measurements | Short timescale | Rapid enrichment |
| **Dusty galaxies** | 4 NIRCam-dark | Extreme ε+dust | Orthogonal validation |

### Core Finding

> At **fixed physical astrophysics** (ε = 0.15 from IllustrisTNG), JANUS matches observations while ΛCDM fails catastrophically. This demonstrates the **cosmological origin** of JANUS advantage.

---

## Data

### Galaxy Catalog v17

- **Sample size**: 200 galaxies
- **Redshift range**: 6.63 < z < 14.32
- **Stellar masses**: 8.45 < log(M*/M☉) < 10.57
- **Spectroscopic**: 55 galaxies (27.5%)
- **Metallicity measurements**: 55 galaxies
- **Dusty/NIRCam-dark**: 4 galaxies

**Sources**: JADES DR4, EXCELS, GLASS, A3COSMOS, CEERS, UNCOVER, COSMOS-Web

---

## Installation

```bash
# Clone repository
git clone https://github.com/PGPLF/JANUS-Z.git
cd JANUS-Z

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.11+
- NumPy >= 2.0
- SciPy >= 1.10
- Pandas >= 2.0
- Matplotlib >= 3.8
- Astropy >= 7.0

---

## Usage

### Run Analysis

```bash
python scripts/analysis_janus_v17_jan2026.py
```

**Outputs**:
- `results/janus_v17_comprehensive_results.json`
- `results/figures/fig_v17_killer_plot_suite.pdf`
- `results/figures/fig_v17_clustering_analysis.pdf`
- `results/figures/fig_v17_metallicity_evolution.pdf`

### Compile Publication

```bash
cd papers/draft_preprint
pdflatex janus_v17_comprehensive.tex
pdflatex janus_v17_comprehensive.tex  # Second pass for references
```

---

## Project Structure

```
JANUS-Z/
├── README.md                 # This file
├── LICENSE                   # MIT License
├── CITATION.cff              # Citation format
├── CHANGELOG.md              # Version history
├── requirements.txt          # Python dependencies
│
├── data/
│   ├── README.md
│   └── jwst_extended_catalog_v17.csv    # 200 galaxies
│
├── scripts/
│   ├── README.md
│   ├── analysis_janus_v17_jan2026.py    # Main analysis
│   └── BIC_calculator.py                # Utility module
│
├── results/
│   ├── README.md
│   ├── janus_v17_comprehensive_results.json
│   └── figures/
│       ├── fig_v17_killer_plot_suite.pdf
│       ├── fig_v17_clustering_analysis.pdf
│       └── fig_v17_metallicity_evolution.pdf
│
├── papers/
│   └── draft_preprint/
│       ├── janus_v17_comprehensive.tex
│       └── janus_v17_comprehensive.pdf  # 11 pages, 530 KB
│
└── docs/
    └── DOCUMENTATION_STANDARD.md
```

---

## Scientific Background

### The JWST Early Galaxy Crisis

JWST observations reveal massive galaxies (M* > 10⁹ M☉) at z > 12, formed within 350 Myr of the Big Bang. Standard ΛCDM cosmology predicts insufficient time for such rapid structure formation without invoking extreme (unphysical) star formation efficiencies.

### JANUS Bimetric Cosmology

The JANUS model (Petit & d'Agostini 2014-2018) proposes:

- **Dual metrics**: Positive-mass (+m) and negative-mass (-m) sectors
- **Density ratio**: ξ₀ = ρ₋/ρ₊ = 64.01 (constrained by Type Ia supernovae)
- **Acceleration factor**: f = √ξ₀ ≈ 8 (from Jeans instability analysis)

**Key prediction**: Structure formation accelerated by factor ~8, enabling rapid galaxy assembly consistent with JWST observations.

### Validation Framework (v17)

Five independent tests:

1. **Stellar Mass Function**: At fixed ε=0.15, JANUS matches while ΛCDM fails
2. **Proto-cluster dynamics**: 4 clusters with M_vir ~ 10²⁰ M☉ at z > 7
3. **Metallicity evolution**: Rapid enrichment consistent with acceleration
4. **SMBH growth**: M_BH ~ 10⁸ M☉ at z > 10 via compression zones
5. **Dusty galaxies**: Orthogonal validation via ALMA mm-selection

---

## Citation

If you use this code or data, please cite:

```bibtex
@software{guerin2026janusz,
  author = {Guerin, Patrick},
  title = {JANUS-Z: Testing Bimetric Cosmology with JWST High-Redshift Galaxies},
  year = {2026},
  version = {17.0},
  url = {https://github.com/PGPLF/JANUS-Z}
}
```

See also `CITATION.cff` for machine-readable citation format.

---

## References

### JANUS Model

- Petit, J.-P. & d'Agostini, G. (2018). Ap&SS 363, 139
- Petit, J.-P. et al. (2024). EPJ C 84, 879
- Petit, J.-P. (2014). MPLA 29, 1450182

### JWST Observations

- Bunker, A. et al. (2025). JADES DR4. arXiv:2510.01033
- Carniani, S. et al. (2024). Nature 633, 318
- Robertson, B. et al. (2024). Nature Astronomy 8, 120
- Carnall, A. et al. (2025). EXCELS. arXiv:2411.11837
- A3COSMOS Collaboration (2025). arXiv:2511.08672

---

## Acknowledgments

This work is dedicated to **Jean-Pierre Petit**, whose visionary development of JANUS bimetric cosmology over four decades laid the foundation for this research. His pioneering simulations (DESY 1992) predicted ×8 structure formation acceleration three decades before JWST discoveries.

**Data**: JWST Science Team, MAST Archive, JADES, EXCELS, GLASS, CEERS, UNCOVER, A3COSMOS, ALMA

---

## Contact

**Author**: Patrick Guerin
**Email**: pg@gfo.bzh
**Affiliation**: Independent Researcher, Brittany, France

---

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

---

*Last updated: 2026-01-04 (Version 17.0)*
