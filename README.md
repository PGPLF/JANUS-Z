# JANUS-Z - Analyse Cosmologique JANUS vs ΛCDM

**Date de création**: 2025-12-08
**Dernière mise à jour**: 2026-01-03 12:45 UTC
**Statut**: En développement actif
**Objectif**: Publication scientifique académique

---

## Vue d'ensemble

**JANUS-Z** est un projet de recherche en cosmologie observationnelle visant à **tester le modèle bi-métrique JANUS contre le modèle cosmologique standard ΛCDM** en utilisant les observations du James Webb Space Telescope (JWST) de galaxies à très haut redshift (z > 10).

### Problématique scientifique

Les observations récentes du JWST révèlent des **galaxies massives et évoluées** à des redshifts z > 12, correspondant à moins de 400 millions d'années après le Big Bang. Ces observations créent une **tension majeure** avec le modèle ΛCDM qui prédit un temps insuffisant pour former de telles structures.

### Hypothèse JANUS

Le modèle cosmologique bi-métrique **JANUS** prédit une formation accélérée des structures via des "ponts spatiaux" entre secteurs de matière positive (+m) et négative (-m), permettant une croissance gravitationnelle amplifiée d'un facteur α (typiquement α = 2-5).

**Notre hypothèse**: JANUS explique naturellement les observations JWST sans nécessiter d'ajustements ad hoc des paramètres de formation stellaire.

---

## Données observationnelles

### Catalogue JWST High-z Galaxies

**16 galaxies confirmées** avec z > 10:

- **Redshift range**: z = 10.6 - 14.32
- **Masses stellaires**: log(M*/M☉) = 8.7 - 9.8
- **Âges des populations**: 160 - 350 Myr

**Programmes JWST sources**:
- JADES (JWST Advanced Deep Extragalactic Survey)
- CEERS (Cosmic Evolution Early Release Science)
- UNCOVER
- GLASS

**Références bibliographiques**:
- Carniani et al. 2024 (JADES-GS-z14-0, z14-1)
- Robertson et al. 2023
- Harikane et al. 2024
- Bunker et al. 2023
- Castellano et al. 2024
- Et al.

---

## Méthodologie

### Modèles comparés

**1. ΛCDM (modèle standard)**:
```
M_max(z) = SFR_max × t_cosmique(z) × efficacité × f_temps
```

**2. JANUS (modèle bi-métrique)**:
```
M_max(z) = SFR_max × [α × t_cosmique(z)] × efficacité × f_temps
```
où α est le facteur d'accélération (paramètre libre)

### Analyse statistique

- **χ² réduit** pour chaque modèle
- **Comptage des tensions**: galaxies dépassant la limite théorique
- **Facteur de Bayes** (approximation BIC)
- **Distributions postérieures** (MCMC - phase 2)

---

## Structure du projet

```
JANUS-Z/
│
├── README.md                           # Ce fichier
├── PROJECT_OVERVIEW.md                 # Vue d'ensemble détaillée
│
├── data/
│   ├── catalogs/                       # Catalogues compilés
│   ├── raw/                            # Données brutes JWST (si téléchargées)
│   └── processed/                      # Données prétraitées
│
├── scripts/
│   ├── analysis_janus_comparison_v1.py # Script principal d'analyse
│   ├── src/                            # Modules Python (phase 2)
│   └── notebooks/                      # Notebooks Jupyter
│
├── results/
│   ├── figures/                        # Figures générées
│   │   └── fig_01_mass_vs_redshift_YYYYMMDD.pdf
│   ├── tables/                         # Tables de résultats
│   │   └── comparison_statistics_YYYYMMDD.txt
│   └── comparison_results_YYYYMMDD.json
│
├── analyses/                           # Rapports d'analyse détaillés
│
├── docs/
│   ├── DOCUMENTATION_STANDARD.md       # Standards de documentation
│   └── templates/                      # Templates pour documents
│
└── papers/                             # Articles en préparation
    └── draft_arxiv/                    # Brouillon article ArXiv
```

---

## Quick Start

### Installation

```bash
# Cloner le repository
git clone https://github.com/PGPLF/JANUS-Z.git
cd JANUS-Z

# Installer les dépendances Python
pip install -r requirements.txt
```

### Exécution de l'analyse rapide

```bash
cd scripts
python analysis_janus_comparison_v1.py
```

**Résultats générés**:
- `data/catalogs/jwst_highz_catalog_YYYYMMDD.csv`: Catalogue compilé
- `results/figures/fig_01_mass_vs_redshift_YYYYMMDD.pdf`: Figure principale
- `results/tables/comparison_statistics_YYYYMMDD.txt`: Statistiques
- `results/comparison_results_YYYYMMDD.json`: Résultats JSON complets

---

## Résultats préliminaires

**Observations JWST (16 galaxies z > 10)**:

| Modèle | χ² réduit | Tensions | Interprétation |
|--------|-----------|----------|----------------|
| **ΛCDM** | ~X.XX | XX/16 galaxies | Forte tension |
| **JANUS (α=3)** | ~X.XX | X/16 galaxies | Meilleur ajustement |

**Analyse bayésienne**: ΔBIC ~ XX.X → Évidence [FORTE/TRÈS FORTE] pour JANUS

> *Note: Résultats à mettre à jour après première exécution*

---

## Roadmap

### Phase 1: Analyse rapide ✅
- [x] Compilation catalogue JWST z > 10
- [x] Implémentation modèles ΛCDM et JANUS
- [x] Calculs statistiques comparatifs
- [x] Figure principale masse vs redshift
- [ ] Exécution et validation résultats

### Phase 2: Analyse détaillée 🚧
- [ ] Architecture logicielle complète
- [ ] Analyse bayésienne MCMC (emcee/dynesty)
- [ ] Tests statistiques rigoureux
- [ ] Figures publication-quality
- [ ] Sensibilité aux paramètres

### Phase 3: Publication 📝
- [ ] Rédaction article scientifique
- [ ] Peer review interne
- [ ] Soumission preprint ArXiv (astro-ph.CO)
- [ ] Soumission journal (ApJ, A&A, ou MNRAS)

### Phase 4: Extensions 🔭
- [ ] Codes de Boltzmann (prédictions CMB)
- [ ] Analyse lentilles gravitationnelles
- [ ] Mesures H(z) à différents redshifts
- [ ] Distribution vitesses dans les amas

---

## Standards de documentation

**Tous les fichiers du projet respectent un format standardisé**:

```markdown
OBJECTIF: [Description claire]
DONNÉES D'ENTRÉE: [Sources, formats]
TÂCHES: [Étapes détaillées]
DONNÉES DE SORTIE: [Résultats attendus]
DATE: [YYYY-MM-DD HH:MM UTC]
```

Voir `docs/DOCUMENTATION_STANDARD.md` pour les détails complets.

---

## Technologies utilisées

- **Python 3.11+**
- **Bibliothèques scientifiques**:
  - NumPy >= 2.0
  - SciPy >= 1.10
  - Pandas >= 2.0
  - Matplotlib >= 3.8
  - Astropy >= 7.0
- **Analyse bayésienne** (phase 2):
  - emcee (MCMC)
  - dynesty (Nested sampling)
  - corner (Visualisation posteriors)
- **Gestion de données**:
  - astroquery (accès archives MAST)

---

## Références clés

### Publications JWST

1. **Carniani et al. 2024** - "JADES: Discovery of extremely high redshift galaxies (z~14) with well-developed morphologies"
2. **Robertson et al. 2023** - "Identification and properties of intense star-forming galaxies at z>10"
3. **Harikane et al. 2024** - "A JWST/NIRSpec First Census of Broad-Line AGNs at z = 4-7"
4. **Bunker et al. 2023** - "JADES NIRSpec Initial Data Release"

### Modèle JANUS

- Documentation et articles sur le modèle bi-métrique JANUS
- Prédictions théoriques de formation des structures

---

## Contribution

Ce projet est développé dans le cadre d'une recherche académique. Pour toute question ou collaboration:

**Contact**: Dr. Patrick Guerin
**Email**: [À compléter]
**Affiliation**: [À compléter]

---

## Licence

[À définir - probablement MIT ou CC-BY pour publication académique]

---

## Citation

Si vous utilisez ce code ou ces résultats dans vos travaux, veuillez citer:

```
Guerin, P. (2026). Testing the JANUS Bimetric Model with JWST High-Redshift Galaxies.
GitHub repository: https://github.com/PGPLF/JANUS-Z
```

*(Citation à mettre à jour après publication)*

---

## Acknowledgements

- **JWST Science Team** pour les observations exceptionnelles
- **Archive MAST** pour l'accès aux données
- **Équipes JADES, CEERS, UNCOVER** pour les catalogues publiés
- **Claude Sonnet 4.5** pour l'assistance au développement

---

**Projet JANUS-Z - Pour une nouvelle cosmologie observationnelle**

*Dernière mise à jour: 2026-01-03 12:45 UTC*
