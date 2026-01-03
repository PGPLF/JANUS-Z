# JANUS-Z - Projet d'Analyse d'Images

**Date de création**: 8 décembre 2025

---

## Description du projet

JANUS-Z est un projet dédié à l'analyse d'images.

---

## Structure du projet

```
JANUS-Z/
│
├── data/
│   └── images/
│       ├── raw/              # Images brutes à analyser
│       ├── processed/        # Images traitées/préprocessées
│       └── annotated/        # Images avec annotations/résultats
│
├── results/                  # Résultats des analyses
│
├── analyses/                 # Rapports et analyses détaillées
│
├── scripts/                  # Scripts d'analyse et de traitement
│
├── docs/                     # Documentation du projet
│
└── README.md                 # Ce fichier
```

---

## Organisation des données

### data/images/raw/
Déposez ici vos **images brutes** à analyser.
- Formats supportés: JPG, PNG, TIFF, etc.
- Organisez par sous-dossiers si nécessaire (par date, par source, par type, etc.)

### data/images/processed/
Images après prétraitement:
- Redimensionnement
- Normalisation
- Amélioration de contraste
- Corrections diverses

### data/images/annotated/
Images avec résultats visuels:
- Annotations
- Détections
- Segmentations
- Visualisations des analyses

---

## Résultats

### results/
Stocke les résultats quantitatifs:
- Fichiers CSV avec métriques
- Fichiers JSON avec données structurées
- Tableaux de statistiques
- Graphiques et visualisations

---

## Analyses

### analyses/
Rapports d'analyse détaillés:
- Rapports au format Markdown
- Synthèses
- Comparaisons
- Interprétations

---

## Scripts

### scripts/
Scripts de traitement et d'analyse:
- Scripts Python pour le traitement d'images
- Scripts d'analyse automatisée
- Utilitaires

---

## Utilisation

1. **Déposer les images** dans `data/images/raw/`
2. **Exécuter les scripts** d'analyse depuis le dossier `scripts/`
3. **Consulter les résultats** dans `results/` et `analyses/`

---

## Notes

- Maintenir une organisation claire des fichiers
- Documenter chaque analyse dans `analyses/`
- Versionner les scripts importants
- Sauvegarder régulièrement les données brutes

---

**Projet créé le**: 8 décembre 2025
