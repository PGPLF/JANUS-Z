# JANUS-Z Plans Log

Ce fichier documente tous les plans d'implementation des versions successives.
Permet la reversibilite et le suivi des travaux.

---

## Plan progressif v17 -> v18

| Version | Contenu | Statut |
|---------|---------|--------|
| v17.0 | Catalogue 200 galaxies, 4 proto-clusters, 5 piliers validation | Termine |
| v17.1 | Catalogue etendu 236 galaxies, 6 proto-clusters, sigma_v + log_Mvir | Termine |
| v17.1a | Fix URL overflow Data Availability | Termine |
| v17.2 | Bootstrap p-values + epsilon sensitivity | Termine |
| v17.3 | MCMC complet (emcee, 10^5 samples) | Termine |
| v17.4 | Test quantitatif dusty galaxies ([CII] LF) | Termine |
| v18.0 | Integration finale | Planifie |

---

## [v17.1] - 2026-01-04 - TERMINE

### Objectif
Etendre le catalogue a 236 galaxies avec 6 proto-clusters et nouvelles colonnes sigma_v/log_Mvir.

### Etapes detaillees

| # | Tache | Description | Fichiers |
|---|-------|-------------|----------|
| 1 | Creer catalogue v17.1 | Copier catalogue v18 archive (236 galaxies) | `data/jwst_extended_catalog_v17.1.csv` |
| 2 | Ajouter colonnes | sigma_v (km/s), log_Mvir pour 27 galaxies | Meme fichier |
| 3 | Mettre a jour script | Copier v17, maj paths et references v17.1 | `scripts/analysis_janus_v17.1_extended.py` |
| 4 | Generer figures | Executer script, 4 figures PDF | `results/figures/fig_v17.1_*.pdf` |
| 5 | Mettre a jour LaTeX | Copier v17, maj stats et refs figures | `papers/draft_preprint/janus_v17.1_extended.tex` |
| 6 | Compiler PDF | 2 passes pdflatex | `janus_v17.1_extended.pdf` (11 pages) |
| 7 | Mettre a jour CHANGELOG | Documenter v17.1 | `CHANGELOG.md` |
| 8 | Commit & Push | GitHub | Commit `6dbe06a` |

### Resultats cles
- 236 galaxies (vs 200 dans v17.0)
- 6 proto-clusters (vs 4): +GLASS-z10-PC, +A2744-z9-PC
- 93 spectroscopiques (39.4%)
- 135 mesures metallicite
- 24 dusty galaxies (x6)

---

## [v17.1a] - 2026-01-04 - TERMINE

### Objectif
Corriger l'URL A3COSMOS qui empietait sur la colonne de droite (bibliographie).

### Etapes detaillees

| # | Tache | Description | Fichiers |
|---|-------|-------------|----------|
| 1 | Corriger URLs | Raccourcir textes affichage liens | `janus_v17.1a_extended.tex` |
| 2 | Recompiler PDF | 2 passes pdflatex | `janus_v17.1a_extended.pdf` |
| 3 | Mettre a jour CHANGELOG | Documenter v17.1a | `CHANGELOG.md` |
| 4 | Commit & Push | GitHub | Commit `879dd5c` |

### Corrections
- JADES DR4: `jades-survey.github.io` (au lieu de l'URL complete)
- A3COSMOS: `sites.google.com/view/a3cosmos` (au lieu de l'URL complete)

---

## [v17.2] - 2026-01-04 - TERMINE

### Objectif
Renforcer la robustesse statistique avec Bootstrap p-values et analyse de sensibilite epsilon.

### Etapes detaillees

| # | Tache | Description | Fichiers |
|---|-------|-------------|----------|
| 1 | Creer script v17.2 | Copier v17.1 + fonctions bootstrap/sensitivity | `scripts/analysis_janus_v17.2_bootstrap.py` |
| 2 | Implementer Bootstrap | 1000 reechantillonnages, chi2/BIC pour chaque | Fonction `run_bootstrap_analysis()` |
| 3 | Implementer epsilon Sensitivity | Grille epsilon in [0.05, 0.20], chi2 JANUS/LCDM | Fonction `run_epsilon_sensitivity()` |
| 4 | Generer figures v17.2 | 2 nouvelles + 4 existantes maj | `results/figures/fig_v17.2_*.pdf` |
| 5 | Mettre a jour LaTeX | Section Bootstrap + Sensitivity | `papers/draft_preprint/janus_v17.2_bootstrap.tex` |
| 6 | Compiler PDF | 2 passes pdflatex | `janus_v17.2_bootstrap.pdf` |
| 7 | Mettre a jour CHANGELOG | Documenter v17.2 | `CHANGELOG.md` |
| 8 | Commit & Push | GitHub | - |

### Details techniques

#### Bootstrap Analysis
```
Algorithme:
- Pour i = 1 a N_bootstrap (1000):
    - Reechantillonner 236 galaxies avec remplacement
    - Calculer chi2_JANUS(i), chi2_LCDM(i)
    - Calculer delta_BIC(i) = BIC_LCDM - BIC_JANUS
- Resultats:
    - delta_chi2 = chi2_LCDM - chi2_JANUS : mediane +/- 68% CI
    - delta_BIC : mediane +/- 68% CI
    - p-value empirique = fraction(delta_BIC < 0)
```

#### Epsilon Sensitivity Analysis
```
Grille:
- epsilon in [0.05, 0.20] par pas de 0.01 (16 points)
- Pour chaque epsilon:
    - chi2_JANUS(epsilon) avec epsilon fixe
    - chi2_LCDM(epsilon) avec epsilon fixe
- Visualisation:
    - Courbes chi2(epsilon) pour les deux modeles
    - Zone physique epsilon < 0.15 en vert
    - Zone unphysique epsilon > 0.15 en rouge
```

### Nouvelles figures
| Figure | Contenu |
|--------|---------|
| `fig_v17.2_bootstrap_distributions.pdf` | Histogrammes delta_chi2, delta_BIC avec CI 68%/95% |
| `fig_v17.2_epsilon_sensitivity.pdf` | chi2(epsilon) JANUS vs LCDM avec zones physiques |

### Livrables attendus
- `scripts/analysis_janus_v17.2_bootstrap.py`
- `results/janus_v17.2_bootstrap_results.json`
- `results/figures/fig_v17.2_*.pdf` (6 fichiers)
- `papers/draft_preprint/janus_v17.2_bootstrap.pdf`

---

## [v17.3] - PLANIFIE

### Objectif
MCMC complet avec emcee pour posterieurs robustes sur epsilon, diagnostics de convergence, corner plots publication-ready.

### Etapes detaillees

| # | Tache | Description | Fichiers |
|---|-------|-------------|----------|
| 1 | Creer script v17.3 | Copier v17.2, augmenter MCMC | `scripts/analysis_janus_v17.3_mcmc.py` |
| 2 | Augmenter n_steps | 1000 -> 100,000 steps | Fonction `run_mcmc_analysis()` |
| 3 | Ajouter diagnostics | Autocorrelation, Gelman-Rubin, acceptance | Nouvelles fonctions |
| 4 | Burn-in + Thinning | Retirer 20% burn-in, thin par tau | Post-processing |
| 5 | Corner plots | Posterieurs 1D/2D avec CI | `plot_mcmc_corner_full()` |
| 6 | Trace plots | Visualisation mixing walkers | `plot_mcmc_traces()` |
| 7 | Generer figures | 8-9 figures v17.3 | `results/figures/fig_v17.3_*.pdf` |
| 8 | Mettre a jour LaTeX | Section MCMC etendue | `janus_v17.3_mcmc.tex` |
| 9 | Compiler PDF | 2 passes pdflatex | `janus_v17.3_mcmc.pdf` |
| 10 | Mettre a jour docs | CHANGELOG, PLANS_LOG, CONVERSATION_LOG | - |
| 11 | Commit & Push | GitHub | - |

### Configuration MCMC

```
n_walkers = 32          # Nombre de walkers (inchange)
n_steps = 100000        # Steps par walker (x100 vs v17.2)
burn_in_frac = 0.2      # Retirer 20% initial
thin_factor = auto      # Base sur autocorrelation time
```

### Diagnostics de convergence

1. **Autocorrelation time (tau)**
   - Calculer tau pour chaque parametre
   - Verifier n_effective = n_samples / tau > 1000
   - Critere: n_steps > 50 x tau

2. **Gelman-Rubin (R-hat)**
   - Comparer variance intra-chain vs inter-chain
   - Critere: R-hat < 1.1 pour tous parametres

3. **Acceptance rate**
   - Taux d'acceptation des proposals
   - Critere: 0.2 < rate < 0.5

4. **Visual diagnostics**
   - Trace plots: verifier mixing
   - Autocorrelation plots: verifier decroissance

### Nouvelles figures

| Figure | Contenu |
|--------|---------|
| `fig_v17.3_mcmc_corner.pdf` | Corner plot 2D JANUS + LCDM |
| `fig_v17.3_mcmc_trace.pdf` | Trace plots walkers |
| `fig_v17.3_mcmc_autocorr.pdf` | Autocorrelation function |
| `fig_v17.3_convergence_diagnostics.pdf` | R-hat, n_eff summary |

### Livrables attendus

- `scripts/analysis_janus_v17.3_mcmc.py`
- `results/janus_v17.3_mcmc_results.json` (avec diagnostics)
- `results/figures/fig_v17.3_*.pdf` (8-9 figures)
- `papers/draft_preprint/janus_v17.3_mcmc.pdf`

### Estimation temps execution

- MCMC JANUS (100k steps): ~15-20 min
- MCMC LCDM (100k steps): ~15-20 min
- Total script: ~45-60 min

### Risques et mitigations

| Risque | Mitigation |
|--------|------------|
| Non-convergence | Augmenter n_steps ou ajuster prior |
| Temps trop long | Option n_steps reduit (50k) |
| Memoire | Thinning agressif si necessaire |

---

## [v17.4] - EN COURS

### Objectif
Test quantitatif des dusty galaxies via [CII] 158um Luminosity Function. Validation orthogonale independante de la selection UV.

### Donnees disponibles
- 24 dusty galaxies A3COSMOS (is_dusty=1)
- Redshift: z = 6.51 - 8.49
- Masses stellaires: log(M*/Msun) = 10.1 - 10.57
- SFR: 112 - 7079 Msun/yr (tres eleves)

### Methodologie [CII] LF

#### 1. Calcul L_[CII] depuis SFR
Relation De Looze et al. (2014), calibree ALMA:
```
log(L_[CII]/Lsun) = 1.0 × log(SFR) + 7.06
```
Dispersion: sigma = 0.3 dex

#### 2. Construction LF [CII] observee
- Bins en L_[CII]: [10^8, 10^8.5, 10^9, 10^9.5, 10^10] Lsun
- Correction completude A3COSMOS (fonction z, L)
- Volumes comouvants par bin z

#### 3. Predictions theoriques
**LCDM**:
```
Phi_[CII](L) = Phi*(L/L*)^alpha × exp(-L/L*)
```
avec L* et Phi* de Yan+2020, Loiacono+2021

**JANUS**:
```
Phi_[CII]_JANUS = f_accel^3 × Phi_[CII]_LCDM
```
(facteur ^3 pour densite volumique enhanced)

#### 4. Comparaison statistique
- chi2 sur bins LF
- Delta_BIC [CII] independant
- p-value empirique

### Etapes detaillees

| # | Tache | Description | Fichiers |
|---|-------|-------------|----------|
| 1 | Creer script v17.4 | Copier v17.3, ajouter [CII] LF | `scripts/analysis_janus_v17.4_cii_lf.py` |
| 2 | Calculer L_[CII] | Relation De Looze SFR->L_[CII] | Fonction `compute_L_CII()` |
| 3 | Construire LF obs | Binning, Vmax, completude | Fonction `compute_CII_LF_observed()` |
| 4 | Predictions JANUS/LCDM | Schechter LF avec/sans f_accel | Fonction `compute_CII_LF_predicted()` |
| 5 | Chi2 + BIC | Comparaison statistique | Fonction `compare_CII_models()` |
| 6 | Generer figures | LF plot, residuels | `results/figures/fig_v17.4_*.pdf` |
| 7 | Mettre a jour LaTeX | Section [CII] LF | `janus_v17.4_cii_lf.tex` |
| 8 | Compiler PDF | 2 passes pdflatex | `janus_v17.4_cii_lf.pdf` |
| 9 | Mettre a jour docs | CHANGELOG, README | - |
| 10 | Commit & Push | GitHub | - |

### Nouvelles figures

| Figure | Contenu |
|--------|---------|
| `fig_v17.4_cii_luminosity_function.pdf` | LF [CII] obs vs JANUS vs LCDM |
| `fig_v17.4_cii_sfr_relation.pdf` | L_[CII] vs SFR avec relation De Looze |
| `fig_v17.4_dusty_mass_sfr.pdf` | Diagramme M*-SFR dusty galaxies |

### Livrables attendus

- `scripts/analysis_janus_v17.4_cii_lf.py`
- `results/janus_v17.4_cii_lf_results.json`
- `results/figures/fig_v17.4_*.pdf` (3+ figures)
- `papers/draft_preprint/janus_v17.4_cii_lf.pdf`

### References cles

- De Looze et al. 2014: SFR-L_[CII] calibration
- Yan et al. 2020: [CII] LF z~5-7
- Loiacono et al. 2021: ALPINE [CII] LF
- A3COSMOS 2025: arXiv:2511.08672

---

## [v18.0] - PLANIFIE

### Objectif
Integration finale de toutes les ameliorations v17.x.

### Contenu prevu
- Catalogue complet 236 galaxies
- 6 proto-clusters avec sigma_v, log_Mvir
- Bootstrap p-values robustes
- MCMC posterieurs complets
- Test dusty galaxies [CII] LF
- Publication finale prete pour soumission

---

## Historique des commits

| Version | Commit | Date | Description |
|---------|--------|------|-------------|
| v17.0 | Multiple | 2026-01-04 | Initial v17 |
| v17.1 | `6dbe06a` | 2026-01-04 | Extended catalog 236 galaxies |
| v17.1a | `879dd5c` | 2026-01-04 | Fix URL overflow |
| v17.2 | `6bc1dfa` | 2026-01-04 | Bootstrap + sensitivity |

---

*Derniere mise a jour: 2026-01-04*
