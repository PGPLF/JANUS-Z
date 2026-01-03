# Standards de Documentation - JANUS-Z

**Date**: 2026-01-03 12:35 UTC
**Version**: 1.0
**Auteur**: Claude Sonnet 4.5

---

## FORMAT STANDARD POUR TOUS LES DOCUMENTS

Chaque document du projet JANUS-Z doit respecter le format suivant pour garantir la traçabilité et la reproductibilité scientifique.

### En-tête obligatoire

```markdown
# [TITRE DU DOCUMENT]

**OBJECTIF**: [Description claire de l'objectif principal en 1-2 phrases]

**DONNÉES D'ENTRÉE**:
- Source 1: [Description, format, chemin]
- Source 2: [Description, format, chemin]
- ...

**TÂCHES**:
1. [Étape 1 avec description claire]
2. [Étape 2 avec description claire]
3. ...

**DONNÉES DE SORTIE**:
- Fichier 1: [Nom, format, chemin, description]
- Fichier 2: [Nom, format, chemin, description]
- ...

**MÉTADONNÉES**:
- Date de création: [YYYY-MM-DD HH:MM UTC]
- Dernière modification: [YYYY-MM-DD HH:MM UTC]
- Auteur: [Nom]
- Version: [X.Y]
- Statut: [Brouillon | En révision | Validé | Publié]

---

[CONTENU DU DOCUMENT]
```

---

## FORMAT POUR LES SCRIPTS PYTHON

### En-tête de fichier

```python
"""
[NOM DU SCRIPT]

OBJECTIF:
    [Description de l'objectif principal]

DONNÉES D'ENTRÉE:
    - fichier1.csv: Description (chemin: data/raw/...)
    - fichier2.txt: Description (chemin: data/...)

TÂCHES:
    1. Chargement et validation des données
    2. Prétraitement et nettoyage
    3. Calculs principaux
    4. Génération des résultats

DONNÉES DE SORTIE:
    - resultat1.csv: Description (chemin: results/...)
    - figure1.pdf: Description (chemin: results/figures/...)

DÉPENDANCES:
    - numpy >= 2.0
    - pandas >= 2.0
    - matplotlib >= 3.8
    - scipy >= 1.10

UTILISATION:
    python script_name.py [arguments]

AUTEUR: [Nom]
DATE: [YYYY-MM-DD]
VERSION: [X.Y]
"""

import numpy as np
# ... autres imports

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'param1': valeur1,
    'param2': valeur2,
    # ... configuration clairement documentée
}

# ============================================================================
# FONCTIONS
# ============================================================================

def fonction_1(param1, param2):
    """
    Description brève de la fonction.

    Args:
        param1 (type): Description
        param2 (type): Description

    Returns:
        type: Description du retour

    Exemple:
        >>> fonction_1(1.0, 2.0)
        3.0
    """
    pass

# ============================================================================
# PROGRAMME PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    # Horodatage
    from datetime import datetime
    print(f"Exécution démarrée: {datetime.utcnow().isoformat()}Z")

    # Code principal

    print(f"Exécution terminée: {datetime.utcnow().isoformat()}Z")
```

---

## FORMAT POUR LES NOTEBOOKS JUPYTER

### Première cellule (Markdown)

```markdown
# [TITRE DU NOTEBOOK]

**OBJECTIF**: [Description]

**DONNÉES D'ENTRÉE**: [Liste]

**DONNÉES DE SORTIE**: [Liste]

**DATE**: 2026-01-03

---

## Table des matières

1. [Configuration](#1-configuration)
2. [Chargement des données](#2-chargement-des-données)
3. [Analyse exploratoire](#3-analyse-exploratoire)
4. [Calculs principaux](#4-calculs-principaux)
5. [Visualisations](#5-visualisations)
6. [Conclusions](#6-conclusions)
```

### Deuxième cellule (Code - Setup)

```python
# Configuration et imports
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Horodatage
print(f"Notebook exécuté: {datetime.utcnow().isoformat()}Z")

# Configuration matplotlib
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 100
```

---

## FORMAT POUR LES FIGURES SCIENTIFIQUES

### Métadonnées intégrées

Toutes les figures doivent inclure:

```python
# Métadonnées de la figure
fig_metadata = {
    'title': 'Titre court',
    'description': 'Description complète de la figure',
    'data_source': 'Chemin vers les données sources',
    'created': '2026-01-03 12:35 UTC',
    'version': '1.0',
    'author': 'Patrick Guerin'
}

# Sauvegarder avec métadonnées
plt.savefig('figure.pdf',
            dpi=300,
            bbox_inches='tight',
            metadata=fig_metadata)
```

### Légende et annotations

- **Titre**: Descriptif, pas juste technique
- **Axes**: Labels avec unités
- **Légende**: Claire, en haut à droite ou meilleur emplacement
- **Annotations**: Si nécessaire pour clarifier
- **Source**: Citation des données en bas de figure

---

## FORMAT POUR LES RÉSULTATS

### Fichiers de résultats

Chaque fichier de résultats (CSV, JSON, HDF5) doit avoir:

1. **Un fichier README associé** avec:
   - Description du contenu
   - Format des colonnes/champs
   - Unités utilisées
   - Date de génération
   - Script qui l'a généré

2. **Métadonnées intégrées** (si format le permet):
   - Timestamp
   - Version du code
   - Paramètres utilisés

### Exemple README pour résultats

```markdown
# [NOM_FICHIER].csv

**Généré par**: scripts/analysis_main.py
**Date**: 2026-01-03 12:35 UTC
**Version code**: 1.0

## Description

Ce fichier contient les résultats de l'analyse comparative JANUS vs ΛCDM.

## Format

| Colonne | Type | Unité | Description |
|---------|------|-------|-------------|
| galaxy_id | string | - | Identifiant unique de la galaxie |
| redshift | float | - | Redshift spectroscopique |
| log_mass | float | log(M☉) | Masse stellaire |
| chi2_lcdm | float | - | χ² pour modèle ΛCDM |
| chi2_janus | float | - | χ² pour modèle JANUS |

## Notes

- Les masses sont en échelle logarithmique (base 10)
- Les erreurs sont à 1σ
- Données manquantes codées comme NaN
```

---

## NOMENCLATURE DES FICHIERS

### Scripts Python
```
[type]_[description]_[version].py

Exemples:
- analysis_janus_comparison_v1.py
- utils_cosmology_v2.py
- plot_mass_function_v1.py
```

### Notebooks
```
[numéro]_[description].ipynb

Exemples:
- 01_data_exploration.ipynb
- 02_quick_analysis.ipynb
- 03_mcmc_fitting.ipynb
```

### Figures
```
fig_[numéro]_[description]_[date].pdf

Exemples:
- fig_01_mass_vs_redshift_20260103.pdf
- fig_02_bayesian_posterior_20260103.pdf
```

### Données
```
[type]_[description]_[date].[ext]

Exemples:
- catalog_jwst_highz_20260103.csv
- results_comparison_20260103.json
```

---

## CONVENTIONS DE CODE

### Python

1. **PEP 8**: Suivre les conventions Python standard
2. **Docstrings**: Format NumPy/Google
3. **Type hints**: Utiliser quand c'est clair
4. **Commentaires**: Expliquer le "pourquoi", pas le "quoi"
5. **Constants**: EN_MAJUSCULES
6. **Functions**: snake_case
7. **Classes**: PascalCase

### Exemple

```python
from typing import Tuple
import numpy as np

# Constants
SPEED_OF_LIGHT = 299792.458  # km/s

def calculate_age_at_redshift(
    z: float,
    H0: float = 70.0,
    omega_m: float = 0.3
) -> Tuple[float, str]:
    """
    Calculate the age of the universe at given redshift.

    Uses standard ΛCDM cosmology with given parameters.

    Parameters
    ----------
    z : float
        Redshift value (must be positive)
    H0 : float, optional
        Hubble constant in km/s/Mpc (default: 70.0)
    omega_m : float, optional
        Matter density parameter (default: 0.3)

    Returns
    -------
    age : float
        Age in Myr
    unit : str
        Unit string "Myr"

    Examples
    --------
    >>> age, unit = calculate_age_at_redshift(z=12.0)
    >>> print(f"Age: {age:.1f} {unit}")
    Age: 378.5 Myr
    """
    # Implementation avec commentaires explicatifs
    pass
```

---

## TRAÇABILITÉ ET REPRODUCTIBILITÉ

### Principes

1. **Chaque résultat doit être traçable**: Pouvoir retrouver quelles données et quel code l'ont généré
2. **Horodatage systématique**: Tous les fichiers générés ont un timestamp
3. **Versioning**: Git pour le code, versioning manuel pour les données
4. **Documentation inline**: Le code doit être auto-documenté

### Checklist pour un nouveau script

- [ ] En-tête avec OBJECTIF, DONNÉES D'ENTRÉE/SORTIE
- [ ] Imports clairement organisés
- [ ] Constants définies en début de fichier
- [ ] Fonctions avec docstrings complètes
- [ ] Horodatage en début et fin d'exécution
- [ ] Logging des étapes importantes
- [ ] Sauvegarde des résultats avec métadonnées
- [ ] Génération d'un README pour les sorties

---

## GESTION DES VERSIONS

### Git commits

Format: `[TYPE] Description courte`

Types:
- `[DATA]`: Ajout/modification de données
- `[ANALYSIS]`: Nouveau script d'analyse
- `[FIX]`: Correction de bug
- `[DOC]`: Documentation uniquement
- `[REFACTOR]`: Restructuration de code
- `[FEAT]`: Nouvelle fonctionnalité

Exemple:
```
[ANALYSIS] Add JANUS vs ΛCDM comparison script

- Implements chi² calculation for both models
- Generates mass-redshift comparison plot
- Outputs results to CSV

Input: data/catalogs/jwst_highz_catalog.csv
Output: results/comparison_20260103.csv
```

---

**Ce document est la référence pour tous les standards de documentation du projet JANUS-Z.**

**Conformité**: Tous les fichiers créés après le 2026-01-03 doivent suivre ces standards.
