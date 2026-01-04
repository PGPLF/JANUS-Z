# Datasets Supernovae Type Ia - JANUS Future Work

**Date**: 2026-01-03 (original), updated 2026-01-04
**Projet**: JANUS-Z - Analyse cosmologique avec données SNIa
**Statut**: PLANNED (v18+) - Not used in current v15-v17 (JWST-focused)

---

## Note sur l'État Actuel (v17.0)

**Versions v15-v17** se concentrent exclusivement sur la validation JANUS via JWST high-z galaxies:
- v15: 108 galaxies, robust statistics
- v16: 150 galaxies, comprehensive tests
- v17: 200 galaxies, dusty galaxies orthogonal validation

**SNIa datasets** (JLA, Pantheon) **ne sont PAS utilisés** dans v15-v17 analyses.

**Référence historique**: Le paramètre ξ₀ = 64.01 utilisé dans v15-v17 provient de Petit & d'Agostini 2018 (analyse JLA), mais les données SNIa ne sont pas ré-analysées dans ce projet.

**Travail futur (v18+)**:
- Joint fit JWST + SNIa + CMB + BAO
- Contraindre simultanément ξ₀, H₀, Ω_m
- Résoudre potentielle tension ξ₀ (SNIa) vs JWST

---

## Datasets Disponibles

### 1. JLA Compilation (740 SNe Ia)

**Source**: SDSS-II/SNLS3 Joint Light-curve Analysis (Betoule et al. 2014)
**URL**: https://supernovae.in2p3.fr/sdss_snls_jla/ReadMe.html
**Date**: 2014
**Publication**: A&A 568, A22 (2014), DOI: 10.1051/0004-6361/201423413

#### Contenu
- **Nombre de SNe**: 740 (redshift 0.01 < z < 1.3)
- **Fichier principal**: `JLA/jla_likelihood_v6/data/jla_lcparams.txt`
- **Matrices de covariance**: `JLA/covmat/`

#### Composition
- Low-z samples (z < 0.1)
- SDSS-II seasons (0.05 < z < 0.4)
- SNLS 3 years (0.2 < z < 1.0)

#### Colonnes du fichier jla_lcparams.txt
```
#name    : Nom de la supernova
zcmb     : Redshift (CMB frame avec correction vitesse peculière)
zhel     : Redshift héliocentriquehttp://127.0.0.1
dz       : Incertitude redshift
mb       : Magnitude apparente SALT2
dmb      : Incertitude magnitude
x1       : Paramètre stretch SALT2
dx1      : Incertitude stretch
color    : Couleur SALT2
dcolor   : Incertitude couleur
3rdvar   : Masse hôte (log M*/M☉)
...      : Autres paramètres
```

#### Référence
Betoule, M., et al. (2014). "Improved cosmological constraints from a joint analysis of the SDSS-II and SNLS supernova samples." Astronomy & Astrophysics, 568, A22.

---

### 2. Pantheon Compilation (1048 SNe Ia)

**Source**: Pan-STARRS1 + plusieurs datasets combinés (Scolnic et al. 2018)
**URL**: https://github.com/dscolnic/Pantheon
**Date**: 2018
**Publication**: ApJ 859, 101 (2018), DOI: 10.3847/1538-4357/aab9bb

#### Contenu
- **Nombre de SNe**: 1048 (redshift 0.01 < z < 2.3)
- **Fichier principal**: `Pantheon/lcparam_full_long.txt`
- **Matrice systématiques**: `Pantheon/sys_full_long.txt`

#### Composition
- Pan-STARRS1 (279 SNe)
- SDSS (335 SNe)
- SNLS (236 SNe)
- HST (19 SNe au-delà de z=1)
- Low-z samples (148 SNe)
- Et autres

#### Colonnes du fichier lcparam_full_long.txt
```
#name    : Nom de la supernova
zcmb     : Redshift CMB frame
zhel     : Redshift héliocentrique
dz       : Incertitude redshift
mb       : Magnitude apparente
dmb      : Incertitude magnitude
x1       : Stretch parameter
dx1      : Incertitude stretch
color    : Color parameter
dcolor   : Incertitude couleur
...      : Autres paramètres
```

#### Référence
Scolnic, D.M., et al. (2018). "The Complete Light-curve Sample of Spectroscopically Confirmed SNe Ia from Pan-STARRS1 and Cosmological Constraints from the Combined Pantheon Sample." The Astrophysical Journal, 859(2), 101.

---

## Utilisation pour JANUS v5.0

### Objectifs
1. **Validation**: Reproduire l'analyse de Petit & d'Agostini 2018 avec JLA (740 SNe)
2. **Extension**: Refaire l'analyse avec Pantheon (1048 SNe, plus récent)
3. **Combinaison**: Ajustement combiné SNIa + JWST haute-z

### Scripts d'analyse
- `scripts/analysis_janus_v5_snia_jla.py` - Analyse JLA
- `scripts/analysis_janus_v5_snia_pantheon.py` - Analyse Pantheon
- `scripts/analysis_janus_v5_combined.py` - SNIa + JWST combinés

### Paramètres JANUS à contraindre
- ξ = ρ₋/ρ₊ (rapport de densité bimétrique)
- χ ∈ [0,1] (force du couplage bimétrique)
- H₀ (constante de Hubble)
- Ω_m (densité de matière)

### Résultats attendus
- Petit & d'Agostini 2018 trouvent ξ ≈ 64 avec JLA
- v4.0 JWST préfère ξ ≈ 256
- v5.0 doit résoudre cette tension!

---

## Publications Clés Petit & d'Agostini

### Article Principal (SNIa)
**d'Agostini, G. & Petit, J.-P. (2018)**
"Constraints on Janus Cosmological model from recent observations of supernovae type Ia"
- Journal: Astrophysics and Space Science, 363(7):139
- DOI: 10.1007/s10509-018-3365-3
- Dataset: JLA (740 SNe)
- Résultat: ξ ≈ 64 (optimal)

### Articles Fondateurs
1. **Petit & d'Agostini (2014)**
   "Negative mass hypothesis in cosmology and the nature of dark energy"
   - Astrophysics and Space Science 354(2):611-615
   - DOI: 10.1007/s10509-014-2106-5

2. **Petit & d'Agostini (2014)**
   "Cosmological bimetric model with interacting positive and negative masses"
   - Modern Physics Letters A 29(34):1450182

3. **Petit & d'Agostini (2015)**
   "Lagrangian derivation of the two coupled field equations in the Janus cosmological model"
   - Astrophysics and Space Science 357:67
   - DOI: 10.1007/s10509-015-2250-6

---

**Auteur**: Patrick Guerin
**Projet**: JANUS-Z
**Version**: v5.0 - SNIa + JWST combinés
