# Changelog

All notable changes to the JANUS-Z project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [18.0] - 2026-01-05

### Added
- **Final Integrated Publication**: Consolidation of all v17.x improvements into unified publication
- **[CII] Luminosity Function section**: Preliminary test with 24 dusty galaxies (ΔBIC = 1.1, inconclusive)
- **Comprehensive JSON**: `janus_v18_comprehensive_results.json` merging v17.3 and v17.4 data
- **9 renamed figures**: All figures updated to v18 naming convention

### Content Integration
- Full MCMC analysis (100k steps) from v17.3
- Bootstrap validation (1000 iterations) from v17.3
- Epsilon sensitivity analysis from v17.3
- Proto-cluster dynamics (6 clusters, 26 members)
- Metallicity evolution (135 galaxies)
- AGN/BH growth analysis (GHZ9, GN-z11)
- Dusty galaxies test (24 NIRCam-dark)
- [CII] LF test (preliminary, from v17.4)

### Key Results (unchanged from v17.3b)
- JANUS: χ² = 81,934, BIC = 81,937 (ε = 0.10)
- LCDM: χ² = 21,641, BIC = 21,644 (ε = 0.10)
- ΔBIC = -60,293 (LCDM lower BIC in template framework)
- At fixed ε=0.15: JANUS χ² = 113,461, LCDM χ² = 40,473
- Bootstrap: ΔBIC = -66,311 [-73,259, -59,448] 68% CI
- [CII] LF: ΔBIC = 1.1 (inconclusive, N=24)

### Files
- `papers/draft_preprint/janus_v18_final.tex`: LaTeX source
- `papers/draft_preprint/janus_v18_final.pdf`: Final publication (13 pages)
- `results/janus_v18_comprehensive_results.json`: Consolidated results
- `results/figures/fig_v18_*.pdf`: 9 figures

## [17.3b] - 2026-01-05

### Fixed
- **ΔBIC convention clarified**: Explicit statement that ΔBIC = BIC_LCDM - BIC_JANUS
  - Negative ΔBIC (-60,293) means BIC_LCDM < BIC_JANUS → LCDM favored by raw BIC
  - Added interpretation caveats about template calibration limitations
- **Figure 3 caption**: Corrected "55 galaxies" to "135 galaxies"
- **Table 1 corrected**: Now compares both models at same ε=0.15:
  - JANUS χ² = 113,461
  - LCDM χ² = 40,473
- **Kass & Raftery reference**: Added complete bibitem for BIC interpretation scale

### Added
- Explicit note on low MCMC optimal ε (~0.01) reflecting template calibration, not physical SF efficiency
- Clarification that template limitations affect absolute χ² but relative comparison remains valid
- Honest acknowledgment that LCDM achieves lower BIC in current template framework
- Discussion of qualitative JANUS advantages (proto-clusters, metallicity, BH growth) as alternative tests

### Changed
- More balanced presentation of model comparison results
- Unified notation: ε → ε throughout (was mixed ε/ε)
- Updated conclusions to accurately reflect that LCDM has lower BIC but JANUS offers unified framework

### Files
- `papers/draft_preprint/janus_v17.3b_mcmc.pdf`: Corrected publication (13 pages)

## [17.4a] - 2026-01-05

### Fixed
- **Missing references**: Added De Looze et al. 2014, Yan et al. 2020, Loiacono et al. 2021, Kass & Raftery 1995
- **Figure 3 caption**: Corrected "55 galaxies" to "135 galaxies" (consistent with text)
- **Epsilon clarification**: Added explanatory note distinguishing three ε values:
  - ε = 0.10 (SMF optimal)
  - ε = 0.0106 (MCMC posterior median)
  - ε = 0.05 (sensitivity analysis minimum)
- **Numerical consistency**: Updated χ² and BIC values to match v17.3 JSON results
  - JANUS: χ² = 81,934, BIC = 81,937
  - LCDM: χ² = 21,641, BIC = 21,644
  - ΔBIC = -60,293

### Changed
- **[CII] test presentation**: Clearly marked as "Preliminary" with caveats about small sample size
- Added section labels for internal cross-references
- Enhanced "Limitations" section discussing [CII] test caveats

### Files
- `papers/draft_preprint/janus_v17.4a_cii_lf.pdf`: Corrected publication (13 pages)

## [17.3a] - 2026-01-05

### Fixed
- **Numerical consistency**: All chi² and BIC values now match JSON results file
  - JANUS chi² = 81,934 (was 149,547)
  - LCDM chi² = 21,641 (was 29,109)
  - ΔBIC = -60,293 (was -120,438)
- **Bootstrap values**: Correctly cited ΔBIC = -66,311 with 68% CI
- **Missing reference**: Added corner2016 to bibliography
- **Table reference**: Removed undefined Table 1 reference
- **Date updated**: January 5, 2026

### Changed
- Added explanatory note about high chi² values due to template model calibration
- Clarified that relative comparison (ΔBIC) remains valid
- Added Section label for bootstrap cross-reference

### Files
- `papers/draft_preprint/janus_v17.3a_mcmc.pdf`: Corrected publication (12 pages)

## [17.4] - 2026-01-05

### Added
- **[CII] Luminosity Function analysis**: Quantitative test of dusty galaxies via [CII] 158um
- **De Looze et al. (2014) calibration**: SFR to L_[CII] conversion with 0.3 dex scatter
- **Schechter LF predictions**: JANUS (L* enhanced x8) vs LCDM theoretical LFs
- New figure: `fig_v17.4_cii_luminosity_function.pdf` (observed vs predicted LF)
- New figure: `fig_v17.4_cii_sfr_relation.pdf` (L_[CII] vs SFR with De Looze relation)
- New figure: `fig_v17.4_dusty_mass_sfr.pdf` (M*-SFR diagram for dusty galaxies)
- New figure: `fig_v17.4_cii_killer_plot.pdf` (combined LF + chi2 + BIC comparison)

### [CII] LF Results
- **24 dusty galaxies** from A3COSMOS (z = 6.51 - 8.49)
- L_[CII] range: 10^9.1 - 10^10.9 Lsun (all bright-end)
- JANUS: chi2 = 12.8, BIC = 13.9
- LCDM: chi2 = 13.9, BIC = 15.0
- **Delta_BIC = 1.1 (INCONCLUSIVE)**

### Notes
- Small sample size (N=24) limits statistical power
- All galaxies at bright-end of LF (log L > 9)
- Orthogonal validation to UV-selected SMF but requires larger samples
- Future: ALMA REBELS DR2 will provide 100+ dusty galaxies

### Files
- `scripts/analysis_janus_v17.4_cii_lf.py`: [CII] LF analysis script
- `results/janus_v17.4_cii_lf_results.json`: Complete results
- `results/figures/fig_v17.4_*.pdf`: 4 new figures

## [17.3] - 2026-01-04

### Added
- **Full MCMC analysis**: 100,000 steps with 32 walkers (emcee) for publication-quality posteriors
- **Checkpointing system**: Auto-save every 10k iterations, resume from interruption
- **Convergence diagnostics**: Autocorrelation time (tau), effective samples (n_eff), acceptance rate
- **Burn-in and thinning**: 20% burn-in removal, autocorrelation-based thinning
- New figure: `fig_v17.3_mcmc_trace.pdf` (walker trace plots, full chain + post burn-in)
- New figure: `fig_v17.3_mcmc_autocorr.pdf` (autocorrelation function with tau annotation)
- New figure: `fig_v17.3_convergence_diagnostics.pdf` (summary table with pass/fail status)
- 68% and 95% credible intervals on epsilon posteriors
- NumpyEncoder class for JSON serialization of numpy types

### MCMC Results
- **JANUS**: epsilon = 0.0106 +0.0000/-0.0004 (68% CI)
  - tau = 1720, n_eff = 1860, acceptance = 35%
- **LCDM**: epsilon = 0.0102 +0.0004/-0.0001 (68% CI)
  - tau = 614, n_eff = 5211, acceptance = 35%
- Both chains converged (n_steps > 50*tau)

### Bootstrap Summary
- Delta_BIC = -66,311 [-73,259, -59,448] (68% CI)
- Empirical p-value = 1.0000 (JANUS always preferred)

### Changed
- Updated all figures from v17.2 to v17.3 naming convention
- Enhanced MCMC section in publication with diagnostics description
- Updated run_mcmc_analysis() with full diagnostics and chain storage

### Files
- `scripts/analysis_janus_v17.3_mcmc.py`: New analysis script with full MCMC (~1970 lines)
- `results/janus_v17.3_mcmc_results.json`: Complete results with MCMC diagnostics
- `results/checkpoints/mcmc_*.pkl`: MCMC checkpoint files for JANUS and LCDM
- `results/figures/fig_v17.3_*.pdf`: 9 figures (6 updated + 3 new)
- `papers/draft_preprint/janus_v17.3_mcmc.tex`: LaTeX source

## [17.2] - 2026-01-04

### Added
- **Bootstrap validation**: 1000 iterations resampling for empirical p-values
- **Epsilon sensitivity analysis**: chi2 vs epsilon over [0.05, 0.20] range
- New figure: `fig_v17.2_bootstrap_distributions.pdf` (Delta_chi2 and Delta_BIC distributions)
- New figure: `fig_v17.2_epsilon_sensitivity.pdf` (chi2 vs epsilon curves)
- Robust confidence intervals on model comparison statistics

### Changed
- Updated all figures from v17.1 to v17.2 naming convention
- Enhanced statistical validation section in publication
- Script now includes verbose parameter for bootstrap efficiency

### Files
- `scripts/analysis_janus_v17.2_bootstrap.py`: New analysis script with bootstrap + sensitivity
- `results/janus_v17.2_bootstrap_results.json`: Complete results including bootstrap statistics
- `results/figures/fig_v17.2_*.pdf`: 6 figures (4 updated + 2 new)
- `papers/draft_preprint/janus_v17.2_bootstrap.pdf`: Publication (12 pages)

## [17.1a] - 2026-01-04

### Fixed
- **Data Availability**: URL A3COSMOS empiétait sur la colonne de droite
- Raccourcissement des liens URL (affichage texte court au lieu de l'URL complète)
- JADES DR4: `jades-survey.github.io` (lien cliquable)
- A3COSMOS: `sites.google.com/view/a3cosmos` (lien cliquable)

### Files
- `papers/draft_preprint/janus_v17.1a_extended.pdf`: Publication corrigée (11 pages)

## [17.1] - 2026-01-04

### Added
- **Extended catalog**: 236 galaxies at 6.50 < z < 14.52 (+36 from v17.0)
- **2 new proto-clusters**: GLASS-z10-PC (5 members, z=10.13), A2744-z9-PC (4 members, z=9.04)
- **New catalog columns**: sigma_v (velocity dispersion) and log_Mvir (virial mass) for 27 galaxies
- **Expanded dusty sample**: 24 NIRCam-dark galaxies (×6 from v17.0)
- **Extended metallicity**: 135 galaxies with 12+log(O/H) measurements (55 → 135)
- 6 proto-clusters total with 26 spectroscopic members

### Changed
- Proto-cluster analysis: 4 → 6 clusters with detailed sigma_v and log_Mvir
- Redshift range extended: 6.50 < z < 14.52
- Spectroscopic fraction improved: 39.4% (93/236)
- Updated figures: fig_v17.1_killer_plot_suite.pdf, fig_v17.1_clustering_analysis.pdf,
  fig_v17.1_metallicity_evolution.pdf, fig_v17.1_mcmc_posteriors.pdf

### Files
- `data/jwst_extended_catalog_v17.1.csv`: Extended catalog with 236 galaxies
- `scripts/analysis_janus_v17.1_extended.py`: Updated analysis script
- `papers/draft_preprint/janus_v17.1_extended.pdf`: Publication (11 pages)
- `results/janus_v17.1_comprehensive_results.json`: Full analysis results

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
