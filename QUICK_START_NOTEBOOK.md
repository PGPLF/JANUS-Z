# Guide d'utilisation du Notebook Jupyter interactif

**Date**: 2026-01-03
**Fichier**: `scripts/notebooks/01_interactive_analysis.ipynb`

---

## Démarrage rapide

### Option 1: Avec Jupyter Notebook classique

```bash
cd /Users/patrickguerin/Desktop/JANUS-Z/scripts/notebooks

# Installer jupyter si nécessaire
pip3 install jupyter

# Lancer Jupyter
jupyter notebook 01_interactive_analysis.ipynb
```

### Option 2: Avec JupyterLab (recommandé)

```bash
cd /Users/patrickguerin/Desktop/JANUS-Z/scripts/notebooks

# Installer jupyterlab si nécessaire
pip3 install jupyterlab

# Lancer JupyterLab
jupyter lab 01_interactive_analysis.ipynb
```

### Option 3: Avec VS Code

1. Ouvrir VS Code
2. Installer l'extension "Jupyter" de Microsoft
3. Ouvrir le fichier `01_interactive_analysis.ipynb`
4. Cliquer sur "Run All" ou exécuter cellule par cellule

---

## Contenu du notebook

### Section 1: Configuration
- Imports des bibliothèques
- Configuration matplotlib
- Horodatage

### Section 2: Chargement des données
- Lecture du catalogue JWST généré
- Extraction des arrays numpy
- Affichage des premières lignes

### Section 3: Définition des modèles
- Fonctions ΛCDM et JANUS
- Fonction de calcul χ²

### Section 4: Exploration paramètre α
- Test de α = 1.0 à 10.0
- Identification du α optimal
- **Graphique χ² vs α** (fig_02)

### Section 5: Comparaison détaillée
- Statistiques complètes ΛCDM vs JANUS optimal
- Facteur de Bayes
- Niveau d'évidence

### Section 6: Figure principale avec α optimal
- **Graphique masse vs redshift** avec α optimal (fig_03)
- Comparaison multiple α

### Section 7: Sensibilité aux paramètres
- Impact de SFR_max
- Impact de l'efficacité
- **Graphiques de sensibilité** (fig_04)

### Section 8: Export résultats
- Sauvegarde JSON des résultats détaillés
- Tableau récapitulatif

### Section 9: Conclusions
- Synthèse des résultats
- Implications scientifiques
- Prochaines étapes

---

## Figures générées par le notebook

Lors de l'exécution complète, le notebook génère:

1. **fig_02_alpha_optimization_20260103.pdf**
   - χ² en fonction de α
   - Nombre de tensions vs α
   - Identification du α optimal

2. **fig_03_optimal_comparison_20260103.pdf**
   - Masse vs redshift avec α optimal
   - Comparaison ΛCDM vs JANUS (plusieurs α)
   - Annotations statistiques

3. **fig_04_sensitivity_analysis_20260103.pdf**
   - Sensibilité à SFR_max
   - Sensibilité à l'efficacité

4. **interactive_analysis_results_20260103.json**
   - Résultats détaillés en JSON
   - α optimal avec statistiques complètes

---

## Utilisation interactive

### Modifier les paramètres

Dans les cellules de code, vous pouvez modifier:

```python
# Section 4: Test d'autres gammes de α
alpha_range = np.linspace(1.0, 15.0, 200)  # Étendre jusqu'à α=15

# Section 7: Autres paramètres à tester
sfr_range = np.linspace(30, 200, 100)  # SFR plus large
eff_range = np.linspace(0.01, 0.30, 100)  # Efficacité plus large
```

### Ajouter vos propres analyses

Créez de nouvelles cellules pour:

- Tester d'autres modèles
- Analyser des sous-échantillons (par redshift, par masse)
- Créer des visualisations supplémentaires
- Exporter des tableaux personnalisés

---

## Résultats attendus

### α optimal prédit

D'après l'analyse du script principal:
- **α = 2.0**: χ² = 9,672
- **α = 3.0**: χ² = 9,194
- **α = 4.0**: χ² = 8,863
- **α > 4.0**: Probablement encore meilleur

Le notebook va affiner cette recherche avec 100 valeurs de α.

### Temps d'exécution

- **Total**: ~30-60 secondes
- Chargement données: instantané
- Calculs α: ~10 secondes
- Génération figures: ~20 secondes
- Export: instantané

---

## Troubleshooting

### Erreur: "No module named 'jupyter'"
```bash
pip3 install jupyter jupyterlab
```

### Erreur: Impossible de trouver le catalogue
Vérifier que le script principal a été exécuté:
```bash
ls ../../data/catalogs/jwst_highz_catalog_20260103.csv
```

### Figures ne s'affichent pas
Ajouter en début de notebook:
```python
%matplotlib inline
```

### Kernel crash
Réduire la résolution des calculs:
```python
alpha_range = np.linspace(1.0, 10.0, 50)  # Au lieu de 100
```

---

## Prochaines étapes après le notebook

1. **Analyser les résultats**
   - Quel est le α optimal trouvé?
   - Combien de tensions restent?
   - Quelle est l'amélioration vs ΛCDM?

2. **Mettre à jour le README**
   - Ajouter les résultats réels dans le tableau
   - Mentionner le α optimal

3. **Rédiger l'analyse**
   - Créer un document dans `analyses/`
   - Interpréter les résultats
   - Préparer pour publication

4. **Commit sur GitHub**
   - Ajouter les nouveaux résultats
   - Pousser les nouvelles figures

---

## Support

Pour toute question sur le notebook:
1. Vérifier les commentaires dans le code
2. Consulter `docs/DOCUMENTATION_STANDARD.md`
3. Relire `PROJECT_OVERVIEW.md`

---

**Bon travail scientifique!** 🔬📊

*Document créé: 2026-01-03*
