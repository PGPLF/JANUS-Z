# JANUS Publication Standards
## Règles de vérification de forme avant génération PDF

**Version**: 1.0
**Dernière mise à jour**: 2025-01-05
**Projet**: JANUS-Z - Validation du Modèle Cosmologique Bimétrique

---

## 1. STRUCTURE DU DOCUMENT

### 1.1 Sections obligatoires
- [ ] **Abstract** : 150-300 mots, résumé autonome
- [ ] **Introduction** : Contexte, problématique, objectifs
- [ ] **Theoretical Framework** : Équations JANUS, paramètres
- [ ] **Data and Methods** : Sources, échantillon, méthodologie statistique
- [ ] **Results** : Tableaux, figures, valeurs numériques
- [ ] **Discussion** : Interprétation, limites, comparaison littérature
- [ ] **Conclusions** : Résultats clés, perspectives
- [ ] **References** : Format uniforme, complètes

### 1.2 Sections optionnelles
- [ ] Acknowledgements (si applicable)
- [ ] Appendices (si contenu, sinon supprimer)
- [ ] Supplementary Material (si volumineux)

### 1.3 Règle des annexes
> **RÈGLE**: Ne jamais laisser une section vide. Si une annexe est déclarée, elle DOIT contenir du contenu. Sinon, la supprimer.

---

## 2. LANGUE ET STYLE

### 2.1 Uniformité linguistique
- [ ] **Langue unique** : Anglais scientifique (UK ou US, cohérent)
- [ ] **Légendes figures** : Toutes en anglais
- [ ] **Légendes tableaux** : Toutes en anglais
- [ ] **Notes internes** : AUCUNE (supprimer "TODO", "FIXME", "NOTE:", etc.)

### 2.2 Terminologie JANUS
| Terme correct | Termes à éviter |
|---------------|-----------------|
| ξ₀ (density ratio) | ξ, xi, ksi |
| ρ₋/ρ₊ | rho-/rho+ |
| faccel = √ξ₀ | α (paramètre ad-hoc) |
| bimetric gravity | bigravity, bi-metric |
| negative mass sector | antimatter sector |

### 2.3 Mentions à supprimer avant publication
- [ ] "PRELIMINARY DRAFT"
- [ ] "NOT FOR DISTRIBUTION"
- [ ] "v1.0, not published"
- [ ] Notes en couleur (rouge, etc.)
- [ ] Commentaires de révision

---

## 3. RÉFÉRENCES BIBLIOGRAPHIQUES

### 3.1 Format standard (AAS/ApJ style)
```
Auteur, A. B., & Auteur, C. D. ANNÉE, Journal, Volume, Page
```

### 3.2 Références JANUS obligatoires
- [ ] Petit, J.-P., & Zejli, H. 2024, Eur. Phys. J. C, 84, 1226 (EPJ-C fondamental)
- [ ] Petit, J.-P., & d'Agostini, G. 2018, Astrophys. Space Sci., 363, 139 (SNIa)
- [ ] Petit, J.-P. 1977, C. R. Acad. Sci. Paris, Série A, 285, 1217 (Original)

### 3.3 Vérifications
- [ ] **Noms corrects** : "d'Agostini" (pas "D'Ambrosio")
- [ ] **Dates cohérentes** : Pas de revue inexistante à la date citée
- [ ] **DOI/arXiv** : Inclure si disponible
- [ ] **Pas de [?] ou ??** : Toutes références résolues

### 3.4 Références croisées LaTeX
Avant compilation finale:
```bash
# Vérifier les références non résolues
grep -n "??" *.aux
grep -n "undefined" *.log
```

---

## 4. ÉQUATIONS ET SYMBOLES

### 4.1 Numérotation
- [ ] Toutes les équations importantes numérotées
- [ ] Références croisées fonctionnelles (Eq. X, pas Eq. ??)
- [ ] Numérotation continue ou par section (cohérent)

### 4.2 Symboles mathématiques
- [ ] **Pas de ■** ou caractères corrompus
- [ ] Indices/exposants corrects (ξ₀, pas ξ0)
- [ ] Unités SI avec espaces ($M_\odot$, pas $M\odot$)

### 4.3 Équations clés JANUS (à vérifier)
```latex
% Density ratio
\xi_0 \equiv \frac{|\rho_-|}{\rho_+}

% Acceleration factor
f_{\rm accel} = \sqrt{\xi_0}

% Enhanced growth
D_{\rm JANUS}(z) = \sqrt{\xi_0} \times D_{\Lambda{\rm CDM}}(z)

% Effective Friedmann parameter
\Omega_{m,{\rm eff}} = \Omega_m \left(1 - \xi_0^{-1/3}\right)
```

---

## 5. TABLEAUX

### 5.1 Format
- [ ] Titre au-dessus du tableau
- [ ] Unités dans les en-têtes de colonnes
- [ ] Alignement décimal pour les nombres
- [ ] Notes explicatives en bas si nécessaire

### 5.2 Contenu obligatoire
- [ ] **Tableau de données galaxies** : ID, z, log(M*/M☉), σ, source
- [ ] **Tableau de résultats** : χ², DOF, paramètres best-fit
- [ ] **Tableau de comparaison** : ΛCDM vs JANUS

### 5.3 Vérifications
- [ ] Pas de cellules vides inexpliquées
- [ ] Sommes/moyennes cohérentes
- [ ] Nombre de galaxies = somme des bins

---

## 6. FIGURES

### 6.1 Figures obligatoires
- [ ] **Figure 1** : Diagramme masse-redshift (M* vs z) avec données JWST et prédictions
- [ ] **Figure 2** : χ² en fonction du paramètre (ξ₀ ou autre)
- [ ] **Figure 3** (optionnel) : Résidus ou distributions

### 6.2 Format
- [ ] Résolution ≥ 300 DPI pour publication
- [ ] Légendes complètes et autonomes
- [ ] Axes labellisés avec unités
- [ ] Couleurs distinguables en N&B si possible

### 6.3 Vérifications
- [ ] Toutes les figures référencées dans le texte
- [ ] Ordre de citation = ordre d'apparition
- [ ] Pas de figure orpheline

---

## 7. COMPILATION LATEX

### 7.1 Checklist pré-compilation
```bash
# 1. Nettoyer les fichiers auxiliaires
rm -f *.aux *.log *.out *.bbl *.blg

# 2. Compiler 2-3 fois pour références
pdflatex document.tex
bibtex document
pdflatex document.tex
pdflatex document.tex

# 3. Vérifier les warnings
grep -i "warning\|undefined\|multiply" document.log
```

### 7.2 Erreurs critiques à corriger
- [ ] `undefined reference` → Corriger les \ref{} et \cite{}
- [ ] `missing character` → Vérifier encodage UTF-8
- [ ] `Overfull hbox` > 10pt → Reformater paragraphe/équation

---

## 8. CONTRÔLE QUALITÉ FINAL

### 8.1 Checklist avant soumission
- [ ] Relire abstract (cohérent avec conclusions)
- [ ] Vérifier tous les χ² et valeurs numériques
- [ ] Compter le nombre de galaxies (cohérent partout)
- [ ] Vérifier ξ₀ = 64.01 (valeur standard)
- [ ] S'assurer que faccel = √64.01 ≈ 8.00

### 8.2 Validation croisée
- [ ] Résultats reproductibles avec scripts `/scripts/`
- [ ] Données sources dans `/data/`
- [ ] Figures générées automatiquement si possible

---

## 9. VERSIONING

### 9.1 Convention de nommage
```
janus_v{MAJOR}.{MINOR}_{description}.tex
janus_v{MAJOR}.{MINOR}_{description}.pdf
```

Exemples:
- `janus_v17.3_mcmc.tex` (version 17, sous-version 3, analyse MCMC)
- `janus_v18.1_final.tex` (version 18, sous-version 1, version finale)

### 9.2 Changelog
Maintenir `CHANGELOG.md` avec:
- Date de modification
- Changements principaux
- Auteur des modifications

---

## 10. CHECKLIST RAPIDE PRÉ-PDF

```
□ Pas de sections vides
□ Langue uniforme (anglais)
□ Références complètes (pas de ??)
□ Symboles mathématiques corrects
□ Figures présentes et référencées
□ Tableaux avec unités
□ Pas de notes internes
□ Compilation sans erreur critique
□ ξ₀ = 64.01 cohérent partout
```

---

*Document créé le 2025-01-05 - Projet JANUS-Z*
