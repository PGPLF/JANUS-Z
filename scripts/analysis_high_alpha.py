"""
JANUS Analysis with High Alpha Values (α = 4, 5, 10)

OBJECTIF:
    Tester le modèle JANUS avec des valeurs élevées de α pour voir si les tensions
    avec les observations JWST peuvent être résolues.

DONNÉES D'ENTRÉE:
    - Catalogue JWST 16 galaxies (déjà généré)

TÂCHES:
    1. Calculer prédictions JANUS pour α = 4, 5, 10
    2. Comparer statistiques vs ΛCDM
    3. Générer figure comparative
    4. Identifier α optimal

DONNÉES DE SORTIE:
    - Statistiques pour chaque α
    - Figure comparative
    - Recommandations

DATE: 2026-01-03
VERSION: 1.0
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

print("="*70)
print("ANALYSE JANUS avec α élevés (4, 5, 10)")
print(f"Exécution: {datetime.now().isoformat()}")
print("="*70)
print()

# Charger données
df = pd.read_csv('../data/catalogs/jwst_highz_catalog_20260103.csv')
z_array = np.array(df['redshift'])
obs_mass = np.array(df['log_stellar_mass'])
obs_err = np.array(df['log_mass_err'])

print(f"[1/4] Données chargées: {len(df)} galaxies")
print()

# Fonctions
def age_universe_at_z(z):
    H0_inv_myr = 977.8
    return 0.96 * H0_inv_myr / ((1 + z)**1.5)

def max_stellar_mass_lcdm(z, sfr_max=80.0, efficiency=0.10, time_frac=0.5):
    t_available = age_universe_at_z(z)
    M_max = sfr_max * t_available * efficiency * time_frac
    return np.log10(M_max)

def max_stellar_mass_janus(z, alpha=3.0, sfr_max=80.0, efficiency=0.10, time_frac=0.5):
    t_available = age_universe_at_z(z) * alpha
    M_max = sfr_max * t_available * efficiency * time_frac
    return np.log10(M_max)

def compute_chi2_excess(obs_mass, pred_limit, obs_err):
    residuals = np.maximum(0, obs_mass - pred_limit)
    chi2 = np.sum((residuals / obs_err)**2)
    return chi2

# Test pour différents α
print("[2/4] Calcul des prédictions pour différents α...")
print()

alpha_values = [3.0, 4.0, 5.0, 10.0]
results = {}

# ΛCDM baseline
lcdm_limits = max_stellar_mass_lcdm(z_array)
chi2_lcdm = compute_chi2_excess(obs_mass, lcdm_limits, obs_err)
tension_lcdm = np.sum(obs_mass > lcdm_limits)

print(f"ΛCDM:")
print(f"  χ² = {chi2_lcdm:.2f}")
print(f"  Tensions = {tension_lcdm}/{len(df)} galaxies")
print(f"  Masses prédites: min={lcdm_limits.min():.2f}, max={lcdm_limits.max():.2f}")
print()

for alpha in alpha_values:
    janus_limits = max_stellar_mass_janus(z_array, alpha=alpha)
    chi2 = compute_chi2_excess(obs_mass, janus_limits, obs_err)
    tension = np.sum(obs_mass > janus_limits)

    # Calcul amélioration
    improvement = 100 * (chi2_lcdm - chi2) / chi2_lcdm

    results[alpha] = {
        'limits': janus_limits,
        'chi2': chi2,
        'tension_count': tension,
        'improvement': improvement
    }

    print(f"JANUS (α={alpha}):")
    print(f"  χ² = {chi2:.2f} (amélioration: {improvement:.1f}%)")
    print(f"  Tensions = {tension}/{len(df)} galaxies ({100*tension/len(df):.0f}%)")
    print(f"  Masses prédites: min={janus_limits.min():.2f}, max={janus_limits.max():.2f}")

    # Gap moyen
    mean_gap = np.mean(obs_mass - janus_limits)
    print(f"  Gap moyen: {mean_gap:.2f} dex")
    print()

# Identifier meilleur α
best_alpha = min(alpha_values, key=lambda a: results[a]['chi2'])
print(f"✓ Meilleur α parmi les valeurs testées: α = {best_alpha}")
print(f"  Tensions restantes: {results[best_alpha]['tension_count']}/{len(df)}")
print()

# Figure comparative
print("[3/4] Génération de la figure comparative...")

plt.style.use('seaborn-v0_8-whitegrid')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

# Subplot 1: Masse vs Redshift
z_range = np.linspace(10.5, 14.5, 200)

# Courbes
ax1.plot(z_range, max_stellar_mass_lcdm(z_range), 'r--', linewidth=3,
         label='ΛCDM', zorder=2)

colors = ['green', 'blue', 'purple', 'orange']
styles = [':', '-', '--', '-.']
for i, alpha in enumerate(alpha_values):
    curve = max_stellar_mass_janus(z_range, alpha=alpha)
    ax1.plot(z_range, curve, color=colors[i], linestyle=styles[i],
             linewidth=3, label=f'JANUS (α={alpha})', zorder=2)

# Observations
ax1.errorbar(z_array, obs_mass, yerr=obs_err,
            fmt='o', markersize=12, capsize=6, capthick=2,
            color='black', ecolor='gray', alpha=0.9,
            label='JWST observations', zorder=5)

ax1.set_xlabel('Redshift z', fontsize=14, fontweight='bold')
ax1.set_ylabel('log₁₀(M*/M☉)', fontsize=14, fontweight='bold')
ax1.set_title('Comparaison JANUS (α élevés) vs ΛCDM', fontsize=16, fontweight='bold')
ax1.set_xlim(10.4, 14.6)
ax1.set_ylim(1.5, 10.2)
ax1.legend(fontsize=12, loc='upper right')
ax1.grid(True, alpha=0.3)

# Subplot 2: χ² vs α
alpha_range_extended = np.linspace(1, 15, 100)
chi2_curve = []
for a in alpha_range_extended:
    limits = max_stellar_mass_janus(z_array, alpha=a)
    chi2 = compute_chi2_excess(obs_mass, limits, obs_err)
    chi2_curve.append(chi2)

ax2.plot(alpha_range_extended, chi2_curve, 'b-', linewidth=2)
ax2.scatter([chi2_lcdm], [0], color='red', s=200, marker='*',
            label=f'ΛCDM: χ²={chi2_lcdm:.0f}', zorder=5)

for alpha in alpha_values:
    ax2.scatter([alpha], [results[alpha]['chi2']], s=150, zorder=5,
                label=f'α={alpha}: χ²={results[alpha]["chi2"]:.0f}')

ax2.set_xlabel('Facteur α', fontsize=14, fontweight='bold')
ax2.set_ylabel('χ²', fontsize=14, fontweight='bold')
ax2.set_title('χ² en fonction de α', fontsize=16, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 15)

plt.tight_layout()

output_path = '../results/figures/fig_HIGH_ALPHA_comparison_20260103.pdf'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Figure sauvegardée: {output_path}")
print()

# Export résultats
print("[4/4] Export des résultats...")

import json

summary = {
    'date': datetime.now().isoformat(),
    'n_galaxies': len(df),
    'LCDM': {
        'chi2': float(chi2_lcdm),
        'tension_count': int(tension_lcdm)
    },
    'JANUS_results': {
        f'alpha_{a}': {
            'chi2': float(results[a]['chi2']),
            'tension_count': int(results[a]['tension_count']),
            'improvement_percent': float(results[a]['improvement'])
        } for a in alpha_values
    },
    'best_alpha': float(best_alpha),
    'best_chi2': float(results[best_alpha]['chi2']),
    'best_tensions': int(results[best_alpha]['tension_count'])
}

json_path = '../results/high_alpha_analysis_20260103.json'
with open(json_path, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"✓ Résultats JSON: {json_path}")
print()

# Synthèse finale
print("="*70)
print("SYNTHÈSE")
print("="*70)
print()
print(f"ΛCDM: χ²={chi2_lcdm:.0f}, tensions={tension_lcdm}/16")
print()
for alpha in alpha_values:
    r = results[alpha]
    print(f"JANUS α={alpha}: χ²={r['chi2']:.0f}, tensions={r['tension_count']}/16, amélioration={r['improvement']:.1f}%")
print()
print(f"→ Meilleur: α={best_alpha} avec {results[best_alpha]['tension_count']} tensions restantes")
print()

if results[best_alpha]['tension_count'] == 0:
    print("🎉 SUCCÈS: JANUS explique TOUTES les observations!")
elif results[best_alpha]['tension_count'] < tension_lcdm:
    print(f"✓ AMÉLIORATION: Tensions réduites de {tension_lcdm} à {results[best_alpha]['tension_count']}")
else:
    print("⚠ PROBLÈME: Tensions persistent même avec α élevé")
    print("   → Peut-être besoin d'ajuster d'autres paramètres (SFR, efficacité)")

print()
print("="*70)

plt.show()
