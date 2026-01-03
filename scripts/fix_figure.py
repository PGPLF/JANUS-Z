"""
Script pour corriger la figure et montrer clairement les limites théoriques

OBJECTIF: Créer une figure avec échelle Y élargie montrant les courbes ΛCDM et JANUS

DATE: 2026-01-03
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Charger données
df = pd.read_csv('../data/catalogs/jwst_highz_catalog_20260103.csv')
z_array = np.array(df['redshift'])
obs_mass = np.array(df['log_stellar_mass'])
obs_err = np.array(df['log_mass_err'])

# Fonctions modèles
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

# Créer figure
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(14, 10))

# Range de redshift pour les courbes
z_range = np.linspace(10.5, 14.5, 200)

# Courbes théoriques
lcdm_curve = max_stellar_mass_lcdm(z_range)
janus_2_5 = max_stellar_mass_janus(z_range, alpha=2.5)
janus_3_0 = max_stellar_mass_janus(z_range, alpha=3.0)
janus_4_0 = max_stellar_mass_janus(z_range, alpha=4.0)

# Zone interdite ΛCDM
ax.fill_between(z_range, 6.5, lcdm_curve, alpha=0.2, color='red',
                label='Zone interdite ΛCDM', zorder=1)

# Courbes limites (LIGNES ÉPAISSES)
ax.plot(z_range, lcdm_curve, 'r--', linewidth=4,
        label='Limite ΛCDM', zorder=3)

ax.plot(z_range, janus_2_5, color='green', linestyle=':', linewidth=3,
        label='JANUS (α=2.5)', zorder=3)

ax.plot(z_range, janus_3_0, color='blue', linestyle='-', linewidth=4,
        label='JANUS (α=3.0)', zorder=3)

ax.plot(z_range, janus_4_0, color='purple', linestyle=':', linewidth=3,
        label='JANUS (α=4.0)', zorder=3)

# Points observés (EN DERNIER pour être au-dessus)
ax.errorbar(z_array, obs_mass, yerr=obs_err,
            fmt='o', markersize=12, capsize=6, capthick=3,
            color='black', ecolor='gray', alpha=0.9, linewidth=2,
            label='JWST observations', zorder=5)

# Configuration axes - ÉCHELLE ÉLARGIE
ax.set_xlabel('Redshift z', fontsize=16, fontweight='bold')
ax.set_ylabel('log₁₀(M*/M☉)', fontsize=16, fontweight='bold')
ax.set_title('Masses stellaires à haut redshift: JANUS vs ΛCDM\\n(Échelle complète montrant les limites théoriques)',
            fontsize=18, fontweight='bold', pad=20)

ax.set_xlim(10.4, 14.6)
ax.set_ylim(6.5, 10.2)  # ÉCHELLE ÉLARGIE pour voir les courbes

ax.legend(fontsize=13, loc='upper right', framealpha=0.95)
ax.grid(True, alpha=0.3, linewidth=1)
ax.tick_params(labelsize=12)

# Annotations détaillées
tension_text = f"""n = 16 galaxies
Redshift: z = {z_array.min():.1f} - {z_array.max():.1f}
Masses obs: log(M*/M☉) = {obs_mass.min():.1f} - {obs_mass.max():.1f}

Tensions ΛCDM: 16/16 (100%)
→ TOUTES les galaxies dépassent la limite ΛCDM

Gap moyen ΛCDM: ~{np.mean(obs_mass - max_stellar_mass_lcdm(z_array)):.1f} dex
Gap moyen JANUS(α=3): ~{np.mean(obs_mass - max_stellar_mass_janus(z_array, 3.0)):.1f} dex"""

ax.text(0.02, 0.98, tension_text,
        transform=ax.transAxes, fontsize=11, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, pad=0.8),
        family='monospace')

# Flèches pointant vers les limites
ax.annotate('Limite ΛCDM\\n(trop basse!)',
            xy=(13.5, max_stellar_mass_lcdm(13.5)), xytext=(13.0, 7.5),
            fontsize=11, color='red', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='red', lw=2))

ax.annotate('JANUS (α=3)\\naméliore mais\\ninsuffisant',
            xy=(13.5, max_stellar_mass_janus(13.5, 3.0)), xytext=(12.5, 8.0),
            fontsize=11, color='blue', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='blue', lw=2))

plt.tight_layout()

# Sauvegarder
output_path = '../results/figures/fig_01_CORRECTED_mass_vs_redshift_20260103.pdf'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Figure corrigée sauvegardée: {output_path}")

# Aussi en PNG pour visualisation
output_png = '../results/figures/fig_01_CORRECTED_mass_vs_redshift_20260103.png'
plt.savefig(output_png, dpi=150, bbox_inches='tight')
print(f"✓ Version PNG: {output_png}")

# Afficher les valeurs pour comprendre
print("\\n" + "="*70)
print("VALEURS DES LIMITES THÉORIQUES")
print("="*70)
print(f"À z = 14.0:")
print(f"  Limite ΛCDM: log(M*/M☉) = {max_stellar_mass_lcdm(14.0):.2f}")
print(f"  JANUS α=2.5: log(M*/M☉) = {max_stellar_mass_janus(14.0, 2.5):.2f}")
print(f"  JANUS α=3.0: log(M*/M☉) = {max_stellar_mass_janus(14.0, 3.0):.2f}")
print(f"  JANUS α=4.0: log(M*/M☉) = {max_stellar_mass_janus(14.0, 4.0):.2f}")
print(f"  Observations JADES-z14-0: log(M*/M☉) = 8.9")
print(f"\\n  GAP vs ΛCDM: {8.9 - max_stellar_mass_lcdm(14.0):.2f} dex")
print(f"  GAP vs JANUS(α=3): {8.9 - max_stellar_mass_janus(14.0, 3.0):.2f} dex")
print("="*70)

plt.show()
