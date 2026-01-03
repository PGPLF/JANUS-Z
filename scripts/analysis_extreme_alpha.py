"""
JANUS Analysis with EXTREME Alpha Values (α = 100, 1000, 10000)

OBJECTIF:
    Tester le modèle JANUS avec des valeurs extrêmes de α pour déterminer
    quelle valeur serait nécessaire pour résoudre complètement les tensions.

DONNÉES D'ENTRÉE:
    - Catalogue JWST 16 galaxies z > 10

TÂCHES:
    1. Calculer prédictions JANUS pour α = 100, 1000, 10000
    2. Identifier à quel α les tensions disparaissent
    3. Analyser implications physiques
    4. Générer figure et rapport

DONNÉES DE SORTIE:
    - Statistiques complètes pour chaque α
    - Figure comparative avec échelle log pour α
    - Rapport d'analyse
    - JSON avec résultats

DATE: 2026-01-03 13:00 UTC
VERSION: 1.0
AUTEUR: Patrick Guerin (Claude Sonnet 4.5)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import json

print("="*80)
print("ANALYSE JANUS - VALEURS EXTRÊMES DE α")
print(f"Exécution: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
print("="*80)
print()

# ============================================================================
# CHARGEMENT DONNÉES
# ============================================================================

df = pd.read_csv('../data/catalogs/jwst_highz_catalog_20260103.csv')
z_array = np.array(df['redshift'])
obs_mass = np.array(df['log_stellar_mass'])
obs_err = np.array(df['log_mass_err'])

print(f"[1/5] Données chargées: {len(df)} galaxies")
print(f"  Redshift: z = {z_array.min():.2f} - {z_array.max():.2f}")
print(f"  Masses: log(M*/M☉) = {obs_mass.min():.2f} - {obs_mass.max():.2f}")
print()

# ============================================================================
# FONCTIONS MODÈLES
# ============================================================================

def age_universe_at_z(z, H0=70.0):
    """Âge univers à redshift z (approximation ΛCDM)"""
    H0_inv_myr = 977.8  # pour H0=70 km/s/Mpc
    return 0.96 * H0_inv_myr / ((1 + z)**1.5)

def max_stellar_mass_lcdm(z, sfr_max=80.0, efficiency=0.10, time_frac=0.5):
    """Masse max formable sous ΛCDM"""
    t_available = age_universe_at_z(z)
    M_max = sfr_max * t_available * efficiency * time_frac
    return np.log10(M_max)

def max_stellar_mass_janus(z, alpha, sfr_max=80.0, efficiency=0.10, time_frac=0.5):
    """Masse max formable sous JANUS avec facteur α"""
    t_available = age_universe_at_z(z) * alpha
    M_max = sfr_max * t_available * efficiency * time_frac
    return np.log10(M_max)

def compute_chi2_excess(obs_mass, pred_limit, obs_err):
    """χ² pour galaxies dépassant limite"""
    residuals = np.maximum(0, obs_mass - pred_limit)
    chi2 = np.sum((residuals / obs_err)**2)
    return chi2

# ============================================================================
# ANALYSE GAMME COMPLÈTE α
# ============================================================================

print("[2/5] Calcul pour gamme complète de α...")
print()

# Gamme complète: 1 à 10000 (échelle log)
alpha_range_full = np.logspace(0, 4, 200)  # 10^0=1 à 10^4=10000
chi2_full = []
tensions_full = []

for alpha in alpha_range_full:
    limits = max_stellar_mass_janus(z_array, alpha=alpha)
    chi2 = compute_chi2_excess(obs_mass, limits, obs_err)
    tension = np.sum(obs_mass > limits)

    chi2_full.append(chi2)
    tensions_full.append(tension)

chi2_full = np.array(chi2_full)
tensions_full = np.array(tensions_full)

# ============================================================================
# TESTS VALEURS SPÉCIFIQUES
# ============================================================================

print("[3/5] Tests valeurs spécifiques de α...")
print()

# ΛCDM baseline
lcdm_limits = max_stellar_mass_lcdm(z_array)
chi2_lcdm = compute_chi2_excess(obs_mass, lcdm_limits, obs_err)
tension_lcdm = np.sum(obs_mass > lcdm_limits)

print(f"ΛCDM (α=1 équivalent):")
print(f"  χ² = {chi2_lcdm:.2f}")
print(f"  Tensions = {tension_lcdm}/{len(df)} ({100*tension_lcdm/len(df):.0f}%)")
print(f"  Masses prédites: {lcdm_limits.min():.2f} - {lcdm_limits.max():.2f}")
print()

# Valeurs spécifiques à tester
alpha_values = [10, 100, 1000, 10000]
results = {}

for alpha in alpha_values:
    janus_limits = max_stellar_mass_janus(z_array, alpha=alpha)
    chi2 = compute_chi2_excess(obs_mass, janus_limits, obs_err)
    tension = np.sum(obs_mass > janus_limits)

    improvement = 100 * (chi2_lcdm - chi2) / chi2_lcdm
    mean_gap = np.mean(obs_mass - janus_limits)
    max_gap = np.max(obs_mass - janus_limits)

    results[alpha] = {
        'limits': janus_limits,
        'chi2': chi2,
        'tension_count': tension,
        'improvement': improvement,
        'mean_gap': mean_gap,
        'max_gap': max_gap
    }

    print(f"JANUS (α={alpha}):")
    print(f"  χ² = {chi2:.2f} (amélioration: {improvement:.1f}%)")
    print(f"  Tensions = {tension}/{len(df)} ({100*tension/len(df):.0f}%)")
    print(f"  Masses prédites: {janus_limits.min():.2f} - {janus_limits.max():.2f}")
    print(f"  Gap moyen: {mean_gap:.2f} dex")
    print(f"  Gap max: {max_gap:.2f} dex")
    print()

# ============================================================================
# TROUVER α CRITIQUE
# ============================================================================

print("[4/5] Recherche α critique (tensions = 0)...")
print()

# α où toutes tensions disparaissent
idx_zero_tensions = np.where(tensions_full == 0)[0]

if len(idx_zero_tensions) > 0:
    alpha_critical = alpha_range_full[idx_zero_tensions[0]]
    chi2_critical = chi2_full[idx_zero_tensions[0]]

    print(f"✓ α critique trouvé: α = {alpha_critical:.2f}")
    print(f"  À ce α, TOUTES les galaxies sont expliquées")
    print(f"  χ² = {chi2_critical:.2f}")
    print()

    # Calculer limites à α critique
    critical_limits = max_stellar_mass_janus(z_array, alpha=alpha_critical)
    print(f"  Masses prédites à α={alpha_critical:.0f}:")
    print(f"    min = {critical_limits.min():.2f}")
    print(f"    max = {critical_limits.max():.2f}")
    print()
else:
    alpha_critical = None
    print("⚠ α critique NON TROUVÉ dans la gamme testée (α ≤ 10000)")
    print("  Les tensions persistent même avec α = 10000")
    print()

# ============================================================================
# FIGURES
# ============================================================================

print("[5/5] Génération des figures...")
print()

plt.style.use('seaborn-v0_8-whitegrid')

# Figure 1: χ² et tensions vs α (échelle log)
fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# Subplot 1: χ² vs α
ax1.semilogx(alpha_range_full, chi2_full, 'b-', linewidth=2, label='JANUS')
ax1.axhline(chi2_lcdm, color='r', linestyle='--', linewidth=2, label='ΛCDM')

# Marquer valeurs spécifiques
for alpha in alpha_values:
    ax1.scatter([alpha], [results[alpha]['chi2']], s=150, zorder=5,
                label=f'α={alpha}')

if alpha_critical:
    ax1.axvline(alpha_critical, color='green', linestyle=':', linewidth=2,
                label=f'α critique = {alpha_critical:.0f}')

ax1.set_xlabel('Facteur α (échelle log)', fontsize=13, fontweight='bold')
ax1.set_ylabel('χ²', fontsize=13, fontweight='bold')
ax1.set_title('χ² en fonction de α', fontsize=15, fontweight='bold')
ax1.legend(fontsize=10, loc='upper right')
ax1.grid(True, alpha=0.3, which='both')
ax1.set_xlim(1, 10000)

# Subplot 2: Nombre de tensions vs α
ax2.semilogx(alpha_range_full, tensions_full, 'g-', linewidth=2)
ax2.axhline(tension_lcdm, color='r', linestyle='--', linewidth=2, label='ΛCDM')

# Marquer valeurs spécifiques
for alpha in alpha_values:
    ax2.scatter([alpha], [results[alpha]['tension_count']], s=150, zorder=5)

if alpha_critical:
    ax2.axvline(alpha_critical, color='green', linestyle=':', linewidth=2,
                label=f'α critique = {alpha_critical:.0f}')
    ax2.axhline(0, color='green', linestyle=':', linewidth=1.5, alpha=0.5)

ax2.set_xlabel('Facteur α (échelle log)', fontsize=13, fontweight='bold')
ax2.set_ylabel('Nombre de galaxies en tension', fontsize=13, fontweight='bold')
ax2.set_title('Tensions en fonction de α', fontsize=15, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, which='both')
ax2.set_xlim(1, 10000)
ax2.set_ylim(-0.5, 17)

plt.tight_layout()
fig1_path = '../results/figures/fig_EXTREME_ALPHA_analysis_20260103.pdf'
plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
print(f"✓ Figure 1 sauvegardée: {fig1_path}")

# Figure 2: Masse vs Redshift avec α extrêmes
fig2, ax = plt.subplots(figsize=(16, 10))

z_range = np.linspace(10.5, 14.5, 200)

# Courbes
ax.plot(z_range, max_stellar_mass_lcdm(z_range), 'r--', linewidth=3,
        label='ΛCDM', zorder=2)

colors = ['green', 'blue', 'purple', 'orange']
for i, alpha in enumerate(alpha_values):
    curve = max_stellar_mass_janus(z_range, alpha=alpha)
    ax.plot(z_range, curve, color=colors[i], linewidth=3,
            label=f'JANUS (α={alpha})', zorder=2)

if alpha_critical:
    curve_crit = max_stellar_mass_janus(z_range, alpha=alpha_critical)
    ax.plot(z_range, curve_crit, 'c-', linewidth=4, linestyle='-.',
            label=f'JANUS (α={alpha_critical:.0f} critique)', zorder=3)

# Observations
ax.errorbar(z_array, obs_mass, yerr=obs_err,
            fmt='o', markersize=12, capsize=6, capthick=2,
            color='black', ecolor='gray', alpha=0.9,
            label='JWST observations', zorder=5)

ax.set_xlabel('Redshift z', fontsize=14, fontweight='bold')
ax.set_ylabel('log₁₀(M*/M☉)', fontsize=14, fontweight='bold')
ax.set_title('Comparaison JANUS (α extrêmes) vs ΛCDM vs Observations',
            fontsize=16, fontweight='bold')
ax.set_xlim(10.4, 14.6)
ax.set_ylim(1.5, 10.5)
ax.legend(fontsize=11, loc='upper right', ncol=2)
ax.grid(True, alpha=0.3)

# Annotation
info_box = f"""Paramètres actuels:
SFR_max = 80 M☉/yr
Efficacité = 10%
Fraction temps = 50%

Résultats:
ΛCDM: {tension_lcdm}/16 tensions
α=10: {results[10]['tension_count']}/16 tensions
α=100: {results[100]['tension_count']}/16 tensions
α=1000: {results[1000]['tension_count']}/16 tensions
α=10000: {results[10000]['tension_count']}/16 tensions"""

if alpha_critical:
    info_box += f"\n\nα critique = {alpha_critical:.0f}"

ax.text(0.02, 0.98, info_box, transform=ax.transAxes,
        fontsize=10, verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
fig2_path = '../results/figures/fig_EXTREME_ALPHA_comparison_20260103.pdf'
plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
print(f"✓ Figure 2 sauvegardée: {fig2_path}")

# ============================================================================
# EXPORT RÉSULTATS
# ============================================================================

summary = {
    'metadata': {
        'date': datetime.now().isoformat(),
        'n_galaxies': len(df),
        'redshift_range': [float(z_array.min()), float(z_array.max())],
        'parameters': {
            'SFR_max': 80.0,
            'efficiency': 0.10,
            'time_fraction': 0.5
        }
    },
    'LCDM': {
        'chi2': float(chi2_lcdm),
        'tension_count': int(tension_lcdm),
        'tension_fraction': float(tension_lcdm / len(df))
    },
    'JANUS_extreme': {
        f'alpha_{a}': {
            'chi2': float(results[a]['chi2']),
            'tension_count': int(results[a]['tension_count']),
            'improvement_percent': float(results[a]['improvement']),
            'mean_gap_dex': float(results[a]['mean_gap']),
            'max_gap_dex': float(results[a]['max_gap'])
        } for a in alpha_values
    },
    'alpha_critical': float(alpha_critical) if alpha_critical else None,
    'chi2_at_critical': float(chi2_critical) if alpha_critical else None
}

json_path = '../results/extreme_alpha_analysis_20260103.json'
with open(json_path, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"✓ Résultats JSON: {json_path}")
print()

# ============================================================================
# SYNTHÈSE
# ============================================================================

print("="*80)
print("SYNTHÈSE - ANALYSE α EXTRÊMES")
print("="*80)
print()

print("RÉSULTATS PAR α:")
print("-" * 80)
print(f"{'Modèle':<20} {'χ²':>10} {'Tensions':>12} {'Gap moyen':>12}")
print("-" * 80)
print(f"{'ΛCDM':<20} {chi2_lcdm:>10.0f} {tension_lcdm:>6}/{len(df):<5} {'N/A':>12}")
for alpha in alpha_values:
    r = results[alpha]
    print(f"{'JANUS α='+str(alpha):<20} {r['chi2']:>10.0f} {r['tension_count']:>6}/{len(df):<5} {r['mean_gap']:>11.2f}dex")
print("-" * 80)
print()

if alpha_critical:
    print(f"✓ SUCCÈS: α critique trouvé = {alpha_critical:.0f}")
    print(f"  À ce α, toutes les tensions sont résolues")
    print()
    print(f"IMPLICATIONS PHYSIQUES:")
    print(f"  Temps effectif = {alpha_critical:.0f} × temps cosmique")
    print(f"  À z=14: t_eff = {alpha_critical * age_universe_at_z(14.0):.0f} Myr")
    print(f"           (vs t_cosmique = {age_universe_at_z(14.0):.0f} Myr)")
    print()
else:
    print(f"⚠ PROBLÈME: Aucun α ≤ 10000 ne résout toutes les tensions")
    print()
    print(f"MÊME avec α=10000:")
    print(f"  - {results[10000]['tension_count']} galaxies restent en tension")
    print(f"  - Gap moyen = {results[10000]['mean_gap']:.2f} dex")
    print(f"  - Gap max = {results[10000]['max_gap']:.2f} dex")
    print()
    print(f"CONCLUSION:")
    print(f"  Les paramètres actuels (SFR=80, eff=0.1) sont TROP conservateurs")
    print(f"  OU le modèle simplifié ne capture pas la physique réelle")
    print()

print("RECOMMANDATIONS:")
print("  1. Tester paramètres plus réalistes (SFR_max=500-1000, eff=0.3-0.5)")
print("  2. Consulter littérature: comment sont calculées les limites théoriques?")
print("  3. Considérer modèles semi-analytiques plus sophistiqués")
print()
print("="*80)

plt.show()
