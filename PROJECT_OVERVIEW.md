# JANUS-Z - Vue d'ensemble du projet

**Date de création**: 2025-12-08
**Dernière mise à jour**: 2026-01-03 12:30 UTC
**Statut**: En développement actif

---

## OBJECTIF PRINCIPAL

Tester le modèle cosmologique bi-métrique JANUS contre le modèle standard ΛCDM en utilisant les observations JWST de galaxies à très haut redshift (z > 10), dans le but de produire une **publication scientifique académique**.

---

## CONTEXTE SCIENTIFIQUE

### Problématique

Les observations récentes du James Webb Space Telescope (JWST) révèlent des galaxies massives et évoluées à des redshifts z > 12 (< 400 millions d'années après le Big Bang). Ces observations créent une **tension majeure** avec le modèle cosmologique standard ΛCDM qui prédit un temps insuffisant pour former de telles structures.

### Modèle JANUS

Le modèle JANUS est un modèle cosmologique bi-métrique qui prédit:
- Formation accélérée des structures via des "ponts spatiaux" entre secteurs +m et -m
- Facteur d'accélération α (typiquement α = 2-5)
- Temps effectif de formation: t_eff = t_cosmique × α
- Capacité à former des galaxies massives beaucoup plus rapidement que ΛCDM

### Hypothèse de travail

**JANUS explique naturellement les observations JWST de galaxies massives précoces sans nécessiter d'ajustements ad hoc des paramètres de formation stellaire.**

---

## DONNÉES D'ENTRÉE

### Sources observationnelles

**Programmes JWST principaux**:
1. **JADES** (JWST Advanced Deep Extragalactic Survey)
   - Proposal IDs: 1180, 1181, 1210
   - Champs: GOODS-South, GOODS-North
   - Galaxies record: JADES-GS-z14-0 (z=14.32), JADES-GS-z14-1 (z=14.17)

2. **CEERS** (Cosmic Evolution Early Release Science)
   - Proposal ID: 1345
   - Zone: Extended Groth Strip (EGS)
   - Candidats z > 10

3. **UNCOVER**
   - Proposal ID: 2561
   - Amas Abell 2744 (lentille gravitationnelle)

### Catalogue compilé

**16 galaxies confirmées** avec:
- Redshift spectroscopique/photométrique: z = 10.6 - 14.32
- Masses stellaires: log(M*/M☉) = 8.7 - 9.8
- Âges des populations: 160 - 350 Myr
- Taux de formation stellaire: 15 - 70 M☉/yr

**Sources bibliographiques**:
- Carniani et al. 2024
- Robertson et al. 2023
- Harikane et al. 2024
- Bunker et al. 2023
- Castellano et al. 2024
- Naidu et al. 2022

---

## TÂCHES PRINCIPALES

### Phase 1: Analyse rapide (proof-of-concept)
1. Compilation du catalogue de galaxies z > 10
2. Implémentation modèles théoriques (ΛCDM vs JANUS)
3. Calculs statistiques comparatifs
4. Génération figure principale
5. Résumé préliminaire des résultats

### Phase 2: Analyse détaillée (publication)
1. Architecture logicielle complète
2. Analyse bayésienne MCMC
3. Tests statistiques rigoureux
4. Figures publication-quality
5. Rédaction article scientifique
6. Soumission preprint ArXiv
7. Soumission journal peer-reviewed

### Phase 3: Extensions
1. Codes de Boltzmann (CMB predictions)
2. Analyse lentilles gravitationnelles
3. Mesures H(z) à différents redshifts
4. Distribution de vitesses dans les amas

---

## DONNÉES DE SORTIE ATTENDUES

### Résultats scientifiques

1. **Comparaison statistique JANUS vs ΛCDM**
   - χ² réduit pour chaque modèle
   - Facteur de Bayes (évidence relative)
   - Nombre de galaxies en tension avec chaque modèle

2. **Paramètres optimaux JANUS**
   - Facteur d'accélération α avec incertitudes
   - Distributions postérieures (MCMC)
   - Prédictions testables

3. **Visualisations**
   - Distribution masse vs redshift
   - Comparaison fonctions de masse
   - Distributions cumulatives
   - Corner plots (paramètres)

### Publications

1. **Article principal**
   - Titre: "Testing the JANUS Bimetric Model with JWST High-Redshift Galaxies"
   - Format: 8-12 pages + appendices
   - Target: ApJ, A&A, ou MNRAS

2. **Preprint ArXiv**
   - Catégorie: astro-ph.CO (Cosmology)
   - Diffusion rapide communauté

3. **Documentation technique**
   - Code source documenté
   - Notebooks d'analyse
   - Fichiers de données

---

## CRITÈRES DE SUCCÈS

### Scientifiques
- Démonstration quantitative que JANUS explique mieux les données que ΛCDM
- Facteur de Bayes > 10 (évidence forte)
- Prédictions testables pour futures observations

### Académiques
- Publication acceptée dans journal peer-reviewed
- Citations dans la littérature
- Contribution au débat cosmologie standard

### Techniques
- Code reproductible et open-source
- Documentation complète
- Figures publication-ready

---

## CALENDRIER PRÉVISIONNEL

- **Phase 1** (Analyse rapide): 1-2 jours
- **Phase 2** (Publication): 2-4 semaines
- **Phase 3** (Extensions): 2-6 mois

---

## NOTES MÉTHODOLOGIQUES

### Standards académiques respectés
- Documentation complète de tous les calculs
- Traçabilité données d'entrée → résultats
- Horodatage de tous les fichiers
- Format citation académique
- Code reproductible

### Format des documents
```
OBJECTIF: [Description claire]
DONNÉES D'ENTRÉE: [Sources, formats]
TÂCHES: [Étapes détaillées]
DONNÉES DE SORTIE: [Résultats attendus]
DATE: [YYYY-MM-DD HH:MM UTC]
```

---

**Document préparé par**: Claude Sonnet 4.5
**Pour**: Dr. Patrick Guerin
**Projet**: JANUS-Z Cosmological Analysis
**Version**: 1.0
**Date**: 2026-01-03 12:30 UTC
