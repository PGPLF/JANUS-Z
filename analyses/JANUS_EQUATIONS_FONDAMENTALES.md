# ÉQUATIONS FONDAMENTALES DU MODÈLE JANUS
**Source**: Petit & Zejli (2024) - HAL-04583560
**Publication**: "Janus Cosmological Model Mathematically & Physically Consistent"
**Date extraction**: 2026-01-03

---

## 1. ÉQUATIONS DE CHAMP COUPLÉES (Dérivation Lagrangienne)

### Système d'équations bimétrique complet

Les équations de champ du modèle JANUS sont dérivées par principe d'action à partir d'un Lagrangien bimétrique.

**Équation pour les masses positives** (éq. 5.11):
```
R_μν - (1/2) g_μν R = χ [T_μν + √(|ḡ|/|g|) T'_μν]
```

**Équation pour les masses négatives** (éq. 5.12):
```
R̄_μν - (1/2) ḡ_μν R̄ = κχ [T̄_μν + √(|g|/|ḡ|) T̄'_μν]
```

Où:
- `g_μν`, `ḡ_μν`: Deux métriques Lorentziennes couplées
- `R_μν`, `R̄_μν`: Tenseurs de Ricci associés
- `R`, `R̄`: Scalaires de Ricci
- `χ`: Constante gravitationnelle d'Einstein (= 8πG/c⁴)
- `κ = -1`: Pour que les masses négatives s'attirent mutuellement
- `T_μν`, `T̄_μν`: Tenseurs énergie-impulsion (masses +/-)
- `T'_μν`, `T̄'_μν`: **Tenseurs d'interaction** (effet croisé)

### Points clés de la dérivation

1. **Action bimétrique** (éq. 5.1):
   ```
   A = ∫[1/(2χ) R + S + S'] √|g| d⁴x + ∫[κ/(2χ̄) R̄ + S̄ + S̄'] √|ḡ| d⁴x
   ```

2. **Tenseurs d'interaction définis par** (éqs. 5.9, 5.10):
   ```
   √(|ḡ|/|g|) T'_μν = -2 δS'/δg_μν + g_μν S'
   √(|g|/|ḡ|) T̄'_μν = -2 δS̄'/δḡ_μν + ḡ_μν S̄'
   ```

3. **Condition de Bianchi**: Les dérivées covariantes des membres de gauche s'annulent identiquement, imposant:
   ```
   ∇^μ T_μν = 0
   ∇̄^μ T̄_μν = 0
   ```

---

## 2. SOLUTION COSMOLOGIQUE FLRW

### Métriques FLRW bimétriques (éqs. 2.3, 2.4)

**Métrique pour masses positives**:
```
ds² = c²dx⁰² - a²(x⁰) [du²/(1-ku²) + u²(dθ² + sin²θ dφ²)]
```

**Métrique pour masses négatives**:
```
d̄s² = c²dx⁰² - ā²(x⁰) [du²/(1-k̄u²) + u²(dθ² + sin²θ dφ²)]
```

**Résultat clé**: La compatibilité exige `k = k̄ = -1` (courbure négative)

### Tenseurs d'interaction pour cosmologie homogène et isotrope

En approximation newtonienne (pression négligeable, phase radiative), les tenseurs prennent la forme (éqs. 2.1, 2.2):

```
T'_μν ≈ diag(ρc², 0, 0, 0)
T̄'_μν ≈ diag(ρ̄c̄², 0, 0, 0)
```

### Coefficients de couplage (éq. 2.7)

La compatibilité du système impose:

```
Φ(x⁰) = (ā/a)³
φ(x⁰) = (a/ā)³
φ · Φ = 1
```

Ces coefficients relient les deux populations et garantissent la conservation de l'énergie totale.

---

## 3. ÉQUATIONS DE FRIEDMANN MODIFIÉES

### Système différentiel (éqs. 2.9, 2.10)

Avec les tenseurs d'interaction ci-dessus, le système se réduit à:

**Masse positive**:
```
a² d²a/dx⁰² = (χ/2) E
```

**Masse négative**:
```
ā² d²ā/dx⁰² = -(χ/2) E
```

Où `E` est l'énergie totale du système bimétrique.

### Conservation de l'énergie (éq. 2.8)

```
E = ρc² a³ + ρ̄c̄² ā³ = constante
```

**Interprétation physique**:
- L'énergie totale est la somme des contributions positive et négative
- Pour correspondre aux observations: **E < 0** (dominé par masses négatives)
- Ceci produit l'accélération cosmique SANS constante cosmologique!

### Conditions initiales

L'énergie totale fixe la relation initiale entre densités:

```
E₀ = ρ₀c² a₀³ + ρ̄₀c̄² ā₀³
```

Si `|ρ̄₀| > ρ₀`, alors `E₀ < 0`, ce qui génère naturellement l'accélération.

---

## 4. RELATION AVEC LES OBSERVABLES

### Distance luminosité

Pour calculer la distance luminosité en JANUS, il faut:

1. **Résoudre le système différentiel** (2.9, 2.10) pour obtenir `a(x⁰)` et `ā(x⁰)`
2. **Calculer le redshift** via la métrique positive:
   ```
   1 + z = a(x⁰_obs) / a(x⁰_em)
   ```
3. **Intégrer la distance comobile**:
   ```
   d_C = c ∫[x⁰_em → x⁰_obs] dx⁰ / a(x⁰)
   ```
4. **Distance luminosité**:
   ```
   d_L = (1+z) d_C
   ```
5. **Module de distance**:
   ```
   μ = 5 log₁₀(d_L / 10 pc)
   ```

**DIFFÉRENCE CLÉ avec v5.0**: On ne rajoute PAS une correction ad-hoc à μ_ΛCDM. On résout les vraies équations de Friedmann JANUS!

### Paramètre fondamental: Rapport de densité

Le paramètre qui contrôle l'évolution cosmologique est:

```
ξ(x⁰) = ρ̄(x⁰) / ρ(x⁰) = rapport des densités
```

Avec conservation énergie et relations de couplage:
```
ρ a³ = ρ₀ a₀³ (conservation masses +)
ρ̄ ā³ = ρ̄₀ ā₀³ (conservation masses -)
```

Donc:
```
ξ(x⁰) = (ρ̄₀/ρ₀) · (ā₀/ā)³ · (a/a₀)³ = ξ₀ · (a/ā)³ = ξ₀ / Φ
```

---

## 5. LIMITES DU MODÈLE

### Région dominée par masse positive

Quand `ρ ≫ |ρ̄|` localement, le système se réduit à (éq. 3.7):

```
R_μν - (1/2) g_μν R = χ T_μν
```

**C'est l'équation d'Einstein classique!**

Ceci garantit que JANUS reproduit:
- Précession du périhélie de Mercure
- Déviation de la lumière par le Soleil
- Solution de Schwarzschild
- Équation TOV pour étoiles à neutrons

### Région dominée par masse négative

Quand `|ρ̄| ≫ ρ` (exemple: Dipole Repeller), le système devient (éqs. 3.9, 3.10):

```
R_μν - (1/2) g_μν R = χ T'_μν
R̄_μν - (1/2) ḡ_μν R̄ = -χ T̄_μν
```

La deuxième équation décrit l'équilibre hydrostatique dans une sphère de masse négative (comme TOV mais avec signe inversé).

La première équation décrit l'effet répulsif sur les particules de masse positive traversant cette région.

---

## 6. FORME EXPLICITE DES TENSEURS D'INTERACTION

### Approximation champ faible (éq. 4.3)

Dans le régime newtonien (près du Dipole Repeller):

```
T'_μν = diag(ρ̄c̄², -p̄, -p̄, -p̄)
```

Ceci donne l'équation TOV modifiée (éq. 4.4):

```
dp̄/dx⁰ = -(m̄ - 4πGp̄r³/c̄⁴) / [r(r + 2m̄)] · (ρ̄ - p̄/c̄²)
```

### Solution de Schwarzschild intérieur pour masse négative (éq. 4.11)

```
d̄s² = [3/2 √(1 - r̄ₙ²/r̂²) - 1/2 √(1 - r²/r̂²)]² dx⁰²
       - dr²/(1 - r²/r̂²) - r²(dθ² + sin²θ dφ²)
```

Avec:
```
r̂ = √(3c²/(8πGρ̄))
```

---

## 7. PRÉDICTIONS OBSERVATIONNELLES

### Accélération cosmique

Avec `E < 0` et courbure négative `k = -1`:
- Accélération de `a(x⁰)` positive
- Asymptotiquement: expansion linéaire (PAS exponentielle comme avec Λ)

**Contraste avec ΛCDM**: La constante cosmologique implique une croissance exponentielle asymptotique. JANUS prédit une décélération progressive vers expansion linéaire.

### Structure lacunaire (Dipole Repeller)

Le modèle prédit des vides sphéroïdaux de ~100-600 Mpc, confirmés observationnellement:
- Dipole Repeller (Hoffman et al. 2017)
- Autres super-vides découverts depuis

### Atténuation de luminosité

**Prédiction spécifique**: Les objets situés derrière un super-vide de masse négative montreront une atténuation de magnitude en **anneau** (pas sur tout le disque).

Effet maximal en géométrie rasante, nul au centre par symétrie.

---

## 8. APPLICATION AUX SNIa

### Stratégie pour analyse JLA/Pantheon

1. **Paramètres libres**:
   - `ξ₀ = ρ̄₀/ρ₀`: Rapport de densité initial
   - `H₀`: Constante de Hubble
   - `Ω_m = ρ₀/(ρ_c)`: Densité de matière positive (normalisée)

2. **Relation E < 0**:
   ```
   E = ρ₀c² a₀³ [1 - ξ₀ (ā₀/a₀)³]
   ```
   Avec couplage Φ = (ā/a)³, fixe la dynamique.

3. **Intégration numérique**:
   - Résoudre (2.9, 2.10) avec conditions initiales
   - Calculer d_L(z) numériquement
   - Calculer μ_JANUS(z) = 5 log₁₀[d_L(z)/10pc]
   - Comparer aux données JLA

4. **Minimisation χ²**:
   ```
   χ² = Σᵢ [μ_obs,i - μ_JANUS(zᵢ; ξ₀, H₀, Ω_m)]² / σ²ᵢ
   ```

### Résultat attendu (Petit & d'Agostini 2018)

Avec dataset JLA:
- **ξ₀ ≈ 64**
- Amélioration significative du χ² par rapport à ΛCDM

---

## 9. COMPARAISON v5.0 vs v6.0

### Version 5.0 (Phénoménologique)

```python
# Approche ad-hoc
μ_JANUS = μ_ΛCDM + A·[(ξ/64)^α - 1]·z/(1+z)^β
```

**Problèmes**:
- Correction arbitraire sans fondement théorique
- Paramètres A, α, β non justifiés
- Résultat: Δχ² ≈ 0 (échec)

### Version 6.0 (Rigoureuse)

```python
# Résolution système différentiel
solve_coupled_friedmann(a, ā, ξ₀, E)  # éqs. 2.9, 2.10
d_L = integrate_luminosity_distance(a(x⁰))
μ_JANUS = 5·log₁₀(d_L / 10pc)
```

**Avantages**:
- Équations dérivées du Lagrangien
- Paramètres physiquement justifiés
- Conservation d'énergie garantie
- Convergence vers Einstein en limite locale

---

## 10. RÉFÉRENCES

1. **Petit, J.-P. & Zejli, H. (2024)**
   "Janus Cosmological Model Mathematically & Physically Consistent"
   HAL-04583560, 12 pages

2. **Petit, J.-P. & d'Agostini, G. (2021)**
   "Constraints on Janus Cosmological model from recent observations of supernovae type Ia"
   Astrophysics and Space Science, HAL-03426721

3. **Petit, J.-P. & Zejli, H. (2024)**
   "A bimetric cosmological model based on Andrei Sakharov's twin universe approach"
   European Physical Journal C, 84:1226
   DOI: 10.1140/epjc/s10052-024-13569-w

---

**Note pour implémentation v6.0**:

La clé est de résoudre numériquement le système couplé (2.9, 2.10) avec:
- Conditions initiales cohérentes
- Conservation E = cste
- Relation Φ = (ā/a)³

Puis calculer les observables cosmologiques (d_L, μ) directement depuis la solution.

**Pas de corrections phénoménologiques!**
