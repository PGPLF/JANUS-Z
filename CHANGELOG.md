# Changelog

All notable changes to the JANUS-Z project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [17.0] - 2026-01-04

### Added
- Extended catalog: 200 galaxies at 6.63 < z < 14.32
- **Dusty galaxies test**: 4 ALMA-selected sources (orthogonal to UV selection bias)
- Metallicity evolution analysis with "impossible" ultra-metal-poor galaxy at z=12.15
- 5-pillar validation framework:
  1. Stellar Mass Function with fixed astrophysics
  2. Proto-cluster dynamics (4 clusters)
  3. Metallicity evolution
  4. SMBH growth constraints
  5. Dusty/obscured galaxies
- New data sources: EXCELS (2025), GLASS confirmations (Jan 2026)

### Changed
- Extended redshift range: 6.63 < z < 14.32 (was 6.50 < z < 14.32)
- Improved BIC calculation: Delta_BIC = -120,438 (very strong evidence)
- Updated figures: killer_plot_suite, clustering_analysis, metallicity_evolution

### Fixed
- Table 1 column alignment corrected from v16

## [16.0] - 2026-01-04

### Added
- Extended catalog: 150 galaxies from JADES DR4, EXCELS, GLASS, A3COSMOS
- Proto-cluster virial mass analysis (4 clusters at z ~ 7-10)
- Comprehensive 4-test validation framework
- Metallicity-redshift evolution analysis
- SMBH growth test with GHZ9

### Changed
- Switched from 108 to 150 galaxy sample
- Added velocity dispersion measurements (sigma_v ~ 180 km/s)
- Professional twocolumn format with booktabs tables

## [15.2] - 2026-01-04

### Fixed
- **Bibliographie**: Correction des références arXiv placeholder
  - JADES DR4: `arXiv:2510.01033` (Curtis-Lake et al. 2025)
  - Shen 2025: `arXiv:2509.19427` (Early Dark Energy simulations)
  - Castellano → Morishita 2023: `arXiv:2211.09097` (GLASS protocluster z=7.88)

### Changed
- Renamed files: `janus_v15.1_robust_statistics.*` → `janus_v15.2_robust_statistics.*`

## [15.1] - 2026-01-04

### Fixed
- **Figure 1 (Killer Plot)**: Replaced placeholder text "(to be generated)" with actual figure `fig_v15_killer_plot.pdf`
- Integrated SMF at z=12 visualization showing JANUS vs ΛCDM comparison

### Changed
- Renamed files: `janus_v15_robust_statistics.*` → `janus_v15.1_robust_statistics.*`

## [15.0] - 2026-01-04

### Added
- Robust statistical framework with proper BIC calculation
- Bootstrap validation for model comparison
- BIC_calculator.py utility module
- Publication-ready figure generation
- Complete author metadata and data availability statement

### Changed
- Consolidated v11-v14 developments into stable release
- Standardized LaTeX format for journal submission
- Improved chi-squared calculation with Poisson statistics

## [13.0] - 2026-01-03

### Added
- Final consolidated publication combining v3-v12 developments
- Complete acknowledgments section for Jean-Pierre Petit
- Comprehensive bibliography

### Changed
- Unified theoretical framework document

## [12.0] - 2026-01-03

### Added
- **Theoretical correction**: Rigorous Jeans derivation f_accel = sqrt(xi_0)
- Removed ad-hoc chi parameter from v3-v11
- Complete theoretical coherence: SNIa -> Jeans -> Galaxy formation

### Changed
- Acceleration factor: 8.063 (ad-hoc) -> 8.001 (derived)
- Zero free parameters beyond xi_0 from SNIa

## [11.0] - 2026-01-03

### Added
- **Degeneracy breaking**: Independent constraints from hydrodynamic simulations
- THESAN-zoom, IllustrisTNG, FIRE-3 epsilon constraints
- 5-scenario comparison (unconstrained, THESAN, IllustrisTNG, FIRE-3, Bayesian)

### Changed
- First statistical preference for JANUS over LCDM: Delta_chi2 = +29.7 (5.4 sigma)

## [10.0] - 2026-01-03

### Added
- Detailed bin-by-bin comparison tables
- 5 redshift bins x 5 mass bins analysis
- Residual analysis per bin

## [9.0] - 2026-01-03

### Added
- Population statistics approach (vs individual extremes)
- 108 galaxy catalog from multiple JWST surveys
- Stellar Mass Function methodology with Sheth-Tormen HMF

### Changed
- Methodology shift: M_max individuals -> SMF population

## [8.0] - 2026-01-03

### Added
- Cross-validation SNIa + JWST
- xi_0 = 64.01 fixed from SNIa (Petit & d'Agostini 2018)
- Predictive test on JWST (not fit)

## [3.0] - 2026-01-03

### Added
- Full bimetric equations: f_accel = sqrt(1 + chi * xi)
- Coupling parameter chi in [0,1]
- Rigorous perturbation theory derivation

## [2.0] - 2026-01-03

### Fixed
- **Critical correction**: Removed invented "alpha" parameter from v1
- Implemented correct JANUS physics: rho_-/rho_+ = 64 (DESY 1992)

## [1.0] - 2026-01-03

### Added
- Initial analysis framework
- 16 extreme JWST galaxies at z > 10
- Basic chi-squared comparison JANUS vs LCDM

---

## Acknowledgments

This project builds upon the theoretical foundations established by:
- **Jean-Pierre Petit** - JANUS bimetric cosmology (1970s-present)
- **Gilles d'Agostini** - SNIa constraints on xi_0 (2014-2018)

## References

- Petit, J.-P. & d'Agostini, G. (2014). Astrophysics and Space Science.
- Petit, J.-P. & d'Agostini, G. (2018). Astrophysics and Space Science.
- Carniani, S. et al. (2024). JADES DR4.
- Robertson, B. et al. (2024). Nature Astronomy.
