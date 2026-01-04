# Scripts JANUS-Z

**Dernière mise à jour**: 2026-01-05 (v17.4)
**Statut**: Production

Ce dossier contient les scripts d'analyse pour la validation du modèle cosmologique JANUS bimétrique.

---

## Scripts Principaux (Version 17.x)

### ⭐ `analysis_janus_v17.4_cii_lf.py`

**Statut**: CURRENT (v17.4)
**Description**: [CII] 158um Luminosity Function analysis for dusty galaxies

**Nouveautés v17.4**:
- Conversion SFR -> L_[CII] (De Looze+2014)
- Construction LF [CII] observée
- Prédictions Schechter JANUS/LCDM
- Comparaison chi2/BIC

**Résultats clés**:
- 24 dusty galaxies A3COSMOS
- ΔBIC = 1.1 (INCONCLUS - échantillon trop petit)

**Sorties**:
- `results/janus_v17.4_cii_lf_results.json`
- `results/figures/fig_v17.4_*.pdf` (4 figures)

---

### `analysis_janus_v17.3_mcmc.py`

**Statut**: Archived (v17.3)
**Description**: Full MCMC analysis with 100,000 iterations, convergence diagnostics, and checkpointing

**Nouveautés v17.3**:
- MCMC complet (emcee, 32 walkers, 100k steps)
- Checkpointing automatique (reprise après interruption)
- Diagnostics de convergence (τ, n_eff, acceptance rate)
- Trace plots et autocorrélation
- Résultats: JANUS ε=0.0106, ΛCDM ε=0.0102

**Données d'entrée**:
- `data/jwst_extended_catalog_v17.1.csv` (236 galaxies)

**Sorties**:
- `results/janus_v17.3_mcmc_results.json`
- `results/checkpoints/mcmc_*.pkl` (checkpoints)
- `results/figures/fig_v17.3_*.pdf` (9 figures)

**Exécution**:
```bash
python3 scripts/analysis_janus_v17.3_mcmc.py
# ~30-45 min pour 100k steps (reprend automatiquement si interrompu)
```

---

### `analysis_janus_v17.2_bootstrap.py`

**Statut**: Archived (v17.2)
**Description**: Bootstrap validation (1000 iterations) + epsilon sensitivity analysis

**Résultats clés**:
- Bootstrap ΔBIC = -66,311 [-73,259, -59,448] (68% CI)
- p-value empirique = 1.0000

---

### `analysis_janus_v17.1_extended.py`

**Statut**: Archived (v17.1)
**Description**: Catalogue étendu 236 galaxies, 6 proto-clusters

---

### `analysis_janus_v17_jan2026.py`

**Statut**: Archived (v17.0)
**Description**: Analyse comprehensive avec catalogue étendu JWST+ALMA (200 galaxies)

**Fonctionnalités**:
- Stellar Mass Functions (SMF) avec Sheth-Tormen + Behroozi
- Proto-cluster dynamics (4 clusters, 16 galaxies)
- Metallicity evolution (55 galaxies avec T_e measurements)
- AGN growth (2 AGN à z>10)
- **Dusty galaxies analysis** (NEW: 4 NIRCam-dark, orthogonal validation)
- Bayesian model comparison (BIC)
- "Killer Plot" at fixed ε=0.15

**Données d'entrée**:
- `data/jwst_extended_catalog_v17.csv` (200 galaxies)

**Sorties**:
- `results/janus_v17_comprehensive_results.json`
- `results/figures/fig_v17_killer_plot_suite.pdf`
- `results/figures/fig_v17_clustering_analysis.pdf`
- `results/figures/fig_v17_metallicity_evolution.pdf`

**Exécution**:
```bash
cd /Users/patrickguerin/Desktop/JANUS-Z/scripts
python3 analysis_janus_v17_jan2026.py
```

**Résultats clés**:
- JANUS ε=0.10 (physical), χ²=149,547
- ΛCDM ε=0.15 fixed (catastrophic), χ²=55,251
- ΔBIC = -120,438 (very strong evidence for JANUS)
- 5 independent validation tests (SMF, clustering, metallicity, AGN, dusty)

---

### `analysis_janus_v16_comprehensive.py`

**Statut**: Archived (v16.0)
**Description**: Version précédente avec 150 galaxies, 4 tests

**Différence vs v17**:
- 150 galaxies (vs 200 in v17)
- 4 tests (pas de dusty galaxies)
- ΔBIC = -120,025 (vs -120,438 in v17)

---

### `analysis_janus_v15_robust.py`

**Statut**: Archived (v15.0)
**Description**: Validation statistique robuste avec 108 galaxies

**Innovations v15**:
- Bayesian BIC au lieu de "5.7σ Wilks"
- Monte Carlo empirical (p<0.001)
- "Killer Plot" concept introduced
- CMB/BAO compatibility discussion

---

## Scripts Historiques (Versions 1-14)

### v14: Comprehensive Framework
- Données 2025-2026 (JADES DR4, EXCELS)
- Tests orthogonaux (clustering, velocity dispersions, [CII])
- Prédictions falsifiables

### v12: Théorie
- Dérivation rigoureuse f_accel = √ξ₀ depuis équations de Jeans
- Correction formule v3-v11 (ad-hoc χ parameter removed)

### v11: Degeneracy Breaking
- Contraintes astrophysiques (IllustrisTNG ε<0.15)
- 5 scénarios testés
- Breakthrough: JANUS favorisé avec paramètres physiques

### v10: Binned Comparison
- Tableaux détaillés par bin (5 redshift bins)
- Publication-style tables

### v9: Population Statistics
- Extension à ~108 galaxies
- Découverte dégénérescence astrophysique

### v8: SNIa-JWST Cross-Validation
- ξ₀=64.01 fixé par SNIa (JLA 740 SNe)
- Test prédictif sur JWST avec ξ fixé

### v3: Bimetric Formulation
- Formule f_accel = √(1+χξ) dérivée des perturbations linéaires
- 6 pages publication

### v2: Correct Physics
- Paramètre ρ₋/ρ₊ (historique ρ₋/ρ₊=64 simulations DESY 1992)
- Correction v1.0 paramètre "α" inventé

### v1: Initial Analysis
- 16 galaxies extrêmes z>10
- Découverte paramètre α critique

---

## Prérequis

### Bibliothèques Python (v17)

```bash
pip install -r requirements.txt
```

**Core dependencies**:
- `numpy >= 2.0`
- `scipy >= 1.10` (uses trapezoid, not deprecated np.trapz)
- `pandas >= 2.0`
- `matplotlib >= 3.8`
- `astropy >= 7.0`

**Required for v17.3 MCMC**:
- `emcee >= 3.0` (ensemble sampler)
- `corner >= 2.2` (posterior visualization)

---

## Structure Scripts v17

```python
# Typical script structure (v17)

# 1. Load catalog
catalog = load_extended_catalog("data/jwst_extended_catalog_v17.csv")

# 2. Compute SMF
smf_janus, smf_lcdm = compute_smf(catalog, xi_0=64.01, f_accel=8.00)

# 3. Analyze proto-clusters
clustering_results = analyze_protoclusters(catalog)

# 4. Analyze metallicity
metallicity_results = analyze_metallicity_evolution(catalog)

# 5. Analyze AGN growth
agn_results = analyze_agn_growth(catalog)

# 6. Analyze dusty galaxies (NEW v17)
dusty_results = analyze_dusty_galaxies(catalog)

# 7. Bayesian comparison
bic_results = compute_bic(smf_janus, smf_lcdm)

# 8. Generate figures
plot_killer_plot_suite(smf_janus, smf_lcdm)
plot_clustering_analysis(clustering_results)
plot_metallicity_evolution(metallicity_results)

# 9. Save results
save_results_json("results/janus_v17_comprehensive_results.json")
```

---

## Développement Futur (v18+)

### Améliorations Planifiées

1. **SMF Calibration**:
   - Replace template Sheth-Tormen with GALFORM/FSPS
   - IllustrisTNG-calibrated abundance matching
   - Reduce absolute χ² to physical levels

2. **MCMC Posteriors**:
   - emcee with 10^5 samples
   - Credible intervals on ξ₀, χ
   - Corner plots for parameter degeneracies

3. **Joint Likelihood**:
   - JWST + Planck + DESI simultaneous fit
   - Full CMB/BAO constraints
   - Boltzmann code integration (CAMB/CLASS modification)

4. **N-body Simulations**:
   - GADGET modification for bimetric gravity
   - Non-linear clustering predictions
   - Mock catalogs for validation

5. **Extended Samples**:
   - ALMA REBELS DR2 (dusty galaxies)
   - JWST Cycle 3 data (500+ galaxies)
   - Velocity dispersions (NIRSpec, 50+ galaxies)

---

## Conventions de Nommage

**Scripts d'analyse**:
```
analysis_janus_vXX_description.py
```

**Résultats JSON**:
```
janus_vXX_description_results.json
janus_vXX_description_results_YYYYMMDD.json  # Dated versions
```

**Figures**:
```
fig_vXX_description.pdf
fig_VXX_description_YYYYMMDD.pdf  # Dated versions
```

---

## Contact

Pour questions techniques ou collaborations:

**Auteur**: Patrick Guerin
**Email**: pg@gfo.bzh
**GitHub**: https://github.com/PGPLF/JANUS-Z

---

**Documentation Scripts JANUS-Z**

*Dernière mise à jour: 2026-01-05 00:30 UTC (v17.3)*
