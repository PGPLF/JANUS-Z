"""
ANALYSE JANUS vs ΛCDM - VERSION 3.0 ÉQUATIONS BIMÉTRIQUE

OBJECTIF:
    Tester modèle JANUS avec approximation bimétrique AMÉLIORÉE basée sur
    les équations de champ complètes (théorie des perturbations linéaires)

AMÉLIORATION vs v2.0:
    v2.0: f_accel ≈ √ξ (approximation simpliste)
    v3.0: f_accel ≈ √(1 + χ·ξ) (dérivée des équations bimétrique)

DONNÉES D'ENTRÉE:
    - 16 galaxies JWST (z = 10.60-14.32)
    - Paramètres astrophysiques réalistes (SFR=800, ε=0.70, f=0.90)
    - Paramètre JANUS: ρ₋/ρ₊ = 64 (historique)

TÂCHES:
    1. Implémenter facteur d'accélération bimétrique amélioré
    2. Comparer v2.0 (√ξ) vs v3.0 (√(1+ξ))
    3. Tests de sensibilité sur ξ et χ (couplage)
    4. Générer figures comparatives
    5. Exporter résultats avec comparaison v2/v3

DONNÉES DE SORTIE:
    - results/janus_v3_bimetric_20260103.json
    - results/figures/fig_JANUS_V3_BIMETRIC_20260103.pdf
    - results/figures/fig_V2_VS_V3_COMPARISON_20260103.pdf

FONDEMENT THÉORIQUE:
    Équation de croissance des perturbations dans espace bimétrique:
    δ̈ + 2H δ̇ = 4πG(ρ₊ + χ·ρ₋)δ

    Facteur d'accélération effectif:
    f_accel = √[(ρ₊ + χ·ρ₋)/ρ₊] = √(1 + χ·ξ)

    Cas limites:
    - ξ → 0: f_accel → 1 (ΛCDM) ✓
    - χ = 0: f_accel = 1 (pas de couplage) ✓
    - ξ >> 1, χ=1: f_accel ≈ √ξ (retrouve v2.0) ✓

AUTEUR: Patrick Guerin
DATE: 2026-01-03
VERSION: 3.0 - BIMÉTRIQUE AMÉLIORÉ
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import json

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paramètres cosmologiques ΛCDM (Planck 2018)
H0 = 70.0          # km/s/Mpc
OMEGA_M = 0.3      # Densité matière
OMEGA_L = 0.7      # Constante cosmologique

# Paramètres astrophysiques réalistes (Boylan-Kolchin 2023)
SFR_MAX = 800.0    # M☉/yr - Taux formation stellaire maximal
EFFICIENCY = 0.70  # Efficacité conversion gaz → étoiles
TIME_FRAC = 0.90   # Fraction temps en formation active

# Paramètres JANUS
DENSITY_RATIO_HISTORICAL = 64  # ρ₋/ρ₊ historique (DESY 1992)
COUPLING_DEFAULT = 1.0         # χ = 1 (couplage maximal)

# ============================================================================
# DONNÉES JWST
# ============================================================================

galaxies_data = {
    'galaxy_id': [
        'JADES-GS-z14-0', 'JADES-GS-z14-1', 'JADES-GS-z13-0', 'CEERS-93316',
        'GLASS-z12', 'CEERS-1670', 'MACS1149-JD1', 'SMACS-z11.5',
        'UHZ1', 'GHZ9', 'UNCOVER-z11', 'GHZ2',
        'CEERS-1019', 'GN-z11', 'HD1', 'GLASS-z10'
    ],
    'redshift': [
        14.32, 13.90, 13.20, 12.50, 12.34, 11.95, 11.85, 11.50,
        11.20, 11.09, 11.00, 10.95, 10.85, 10.60, 10.70, 10.65
    ],
    'log_mass_obs': [
        9.80, 9.89, 9.40, 9.10, 9.00, 9.41, 9.20, 9.00,
        9.15, 9.08, 8.79, 9.26, 9.10, 9.11, 8.70, 9.05
    ],
    'log_mass_err': [
        0.30, 0.35, 0.25, 0.20, 0.22, 0.28, 0.25, 0.30,
        0.24, 0.26, 0.20, 0.27, 0.23, 0.18, 0.25, 0.22
    ]
}

df = pd.DataFrame(galaxies_data)
z_array = df['redshift'].values
obs_mass = df['log_mass_obs'].values
obs_err = df['log_mass_err'].values

# ============================================================================
# FONCTIONS COSMOLOGIQUES
# ============================================================================

def age_universe_at_z(z):
    """Âge de l'univers au redshift z (approximation ΛCDM)"""
    from scipy.integrate import quad

    def integrand(zp):
        return 1 / ((1 + zp) * np.sqrt(OMEGA_M * (1 + zp)**3 + OMEGA_L))

    integral, _ = quad(integrand, z, np.inf)
    age_gyr = integral / (H0 * 1.022e-3)  # Conversion en Gyr
    age_myr = age_gyr * 1000  # Conversion en Myr

    return age_myr

age_universe_at_z_vec = np.vectorize(age_universe_at_z)

# ============================================================================
# MODÈLE ΛCDM
# ============================================================================

def max_stellar_mass_lcdm(z, sfr_max=SFR_MAX, efficiency=EFFICIENCY, time_frac=TIME_FRAC):
    """
    Masse stellaire maximale selon ΛCDM.

    M_max = SFR_max × t(z) × ε × f_time
    """
    t_available = age_universe_at_z_vec(z)
    M_max = sfr_max * t_available * efficiency * time_frac
    return np.log10(M_max)

# ============================================================================
# MODÈLE JANUS v2.0 (pour comparaison)
# ============================================================================

def acceleration_factor_v2(density_ratio):
    """
    v2.0: Approximation simpliste √ξ

    ❌ Trop simple, pas dérivée des équations
    """
    return np.sqrt(density_ratio)

def max_stellar_mass_janus_v2(z, density_ratio=DENSITY_RATIO_HISTORICAL,
                               sfr_max=SFR_MAX, efficiency=EFFICIENCY, time_frac=TIME_FRAC):
    """
    JANUS v2.0: f_accel ≈ √ξ
    """
    accel_v2 = acceleration_factor_v2(density_ratio)
    t_available = age_universe_at_z_vec(z)
    M_max = sfr_max * t_available * efficiency * time_frac * accel_v2
    return np.log10(M_max)

# ============================================================================
# MODÈLE JANUS v3.0 - BIMÉTRIQUE AMÉLIORÉ ⭐ NOUVEAU
# ============================================================================

def acceleration_factor_v3(density_ratio, coupling=COUPLING_DEFAULT):
    """
    v3.0: Approximation bimétrique améliorée √(1 + χ·ξ)

    ✅ Dérivée des équations de perturbation bimétrique

    Théorie:
        Équation de croissance: δ̈ + 2H δ̇ = 4πG(ρ₊ + χ·ρ₋)δ
        Facteur d'accélération: f = √[(ρ₊ + χ·ρ₋)/ρ₊] = √(1 + χ·ξ)

    Arguments:
        density_ratio (float): ξ = ρ₋/ρ₊
        coupling (float): χ ∈ [0,1], force couplage bimétrique

    Retourne:
        float: Facteur d'accélération bimétrique

    Cas limites:
        ξ → 0: f → 1 (ΛCDM)
        χ = 0: f = 1 (pas de couplage)
        ξ >> 1, χ=1: f ≈ √ξ (asymptote v2.0)

    Exemples:
        >>> acceleration_factor_v3(64, 1.0)
        8.062257748...
        >>> acceleration_factor_v3(0, 1.0)
        1.0
        >>> acceleration_factor_v3(64, 0.0)
        1.0
    """
    return np.sqrt(1.0 + coupling * density_ratio)

def max_stellar_mass_janus_v3(z, density_ratio=DENSITY_RATIO_HISTORICAL,
                               coupling=COUPLING_DEFAULT,
                               sfr_max=SFR_MAX, efficiency=EFFICIENCY, time_frac=TIME_FRAC):
    """
    JANUS v3.0: f_accel ≈ √(1 + χ·ξ)

    M_max = SFR_max × t(z) × ε × f_time × √(1 + χ·ξ)
    """
    accel_v3 = acceleration_factor_v3(density_ratio, coupling)
    t_available = age_universe_at_z_vec(z)
    M_max = sfr_max * t_available * efficiency * time_frac * accel_v3
    return np.log10(M_max)

# ============================================================================
# STATISTIQUES
# ============================================================================

def compute_chi2_excess(obs_masses, predicted_limits, errors):
    """Calcule χ² pour galaxies au-dessus de la limite prédite"""
    excess_mask = obs_masses > predicted_limits
    if not np.any(excess_mask):
        return 0.0

    diff = obs_masses[excess_mask] - predicted_limits[excess_mask]
    err = errors[excess_mask]
    chi2 = np.sum((diff / err) ** 2)

    return chi2

def count_tensions(obs_masses, predicted_limits):
    """Compte galaxies en tension (masse obs > limite prédite)"""
    return np.sum(obs_masses > predicted_limits)

def mean_gap(obs_masses, predicted_limits):
    """Gap moyen en dex"""
    return np.mean(obs_masses - predicted_limits)

# ============================================================================
# PROGRAMME PRINCIPAL
# ============================================================================

if __name__ == "__main__":

    print("="*80)
    print("ANALYSE JANUS v3.0 - ÉQUATIONS BIMÉTRIQUE AMÉLIORÉES")
    print("="*80)
    print(f"Exécution: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print()

    print("AMÉLIORATION v2.0 → v3.0:")
    print("  v2.0: f_accel ≈ √ξ (approximation simpliste)")
    print("  v3.0: f_accel ≈ √(1 + χ·ξ) (théorie perturbations bimétrique)")
    print("="*80)
    print()

    # ========================================================================
    # ÉTAPE 1: Calcul ΛCDM et JANUS v2/v3 avec ξ = 64
    # ========================================================================

    print("[1/5] Calcul ΛCDM et JANUS (v2.0 vs v3.0) avec ξ = 64...")
    print()

    # ΛCDM
    limits_lcdm = max_stellar_mass_lcdm(z_array)
    chi2_lcdm = compute_chi2_excess(obs_mass, limits_lcdm, obs_err)
    tensions_lcdm = count_tensions(obs_mass, limits_lcdm)
    gap_lcdm = mean_gap(obs_mass, limits_lcdm)

    # JANUS v2.0 (ξ=64)
    limits_janus_v2 = max_stellar_mass_janus_v2(z_array, density_ratio=64)
    chi2_janus_v2 = compute_chi2_excess(obs_mass, limits_janus_v2, obs_err)
    tensions_janus_v2 = count_tensions(obs_mass, limits_janus_v2)
    gap_janus_v2 = mean_gap(obs_mass, limits_janus_v2)
    accel_v2_64 = acceleration_factor_v2(64)

    # JANUS v3.0 (ξ=64, χ=1.0)
    limits_janus_v3 = max_stellar_mass_janus_v3(z_array, density_ratio=64, coupling=1.0)
    chi2_janus_v3 = compute_chi2_excess(obs_mass, limits_janus_v3, obs_err)
    tensions_janus_v3 = count_tensions(obs_mass, limits_janus_v3)
    gap_janus_v3 = mean_gap(obs_mass, limits_janus_v3)
    accel_v3_64 = acceleration_factor_v3(64, 1.0)

    print(f"ΛCDM:")
    print(f"  χ² = {chi2_lcdm:.2f}")
    print(f"  Tensions = {tensions_lcdm}/16")
    print(f"  Gap moyen = {gap_lcdm:.3f} dex")
    print()

    print(f"JANUS v2.0 (ξ=64, f_accel={accel_v2_64:.3f}):")
    print(f"  χ² = {chi2_janus_v2:.2f}")
    print(f"  Tensions = {tensions_janus_v2}/16")
    print(f"  Gap moyen = {gap_janus_v2:.3f} dex")
    print(f"  Amélioration = {100*(chi2_lcdm-chi2_janus_v2)/chi2_lcdm:.1f}%")
    print()

    print(f"JANUS v3.0 (ξ=64, χ=1.0, f_accel={accel_v3_64:.3f}):")
    print(f"  χ² = {chi2_janus_v3:.2f}")
    print(f"  Tensions = {tensions_janus_v3}/16")
    print(f"  Gap moyen = {gap_janus_v3:.3f} dex")
    print(f"  Amélioration vs ΛCDM = {100*(chi2_lcdm-chi2_janus_v3)/chi2_lcdm:.1f}%")
    print(f"  Amélioration vs v2.0 = {100*(chi2_janus_v2-chi2_janus_v3)/chi2_janus_v2:.2f}%")
    print()

    print(f"→ Différence f_accel: v3.0 - v2.0 = {accel_v3_64 - accel_v2_64:+.3f}")
    print(f"→ Différence χ²: v3.0 - v2.0 = {chi2_janus_v3 - chi2_janus_v2:+.2f}")
    print()

    # ========================================================================
    # ÉTAPE 2: Tests de sensibilité - Variation ξ (v2 vs v3)
    # ========================================================================

    print("[2/5] Tests sensibilité: variation ξ (v2.0 vs v3.0)...")
    print()

    density_ratios = [16, 32, 64, 128, 256]
    results_v2 = {}
    results_v3 = {}

    for xi in density_ratios:
        # v2.0
        limits_v2 = max_stellar_mass_janus_v2(z_array, density_ratio=xi)
        chi2_v2 = compute_chi2_excess(obs_mass, limits_v2, obs_err)
        accel_v2 = acceleration_factor_v2(xi)
        improvement_v2 = 100 * (chi2_lcdm - chi2_v2) / chi2_lcdm

        results_v2[xi] = {
            'chi2': chi2_v2,
            'tensions': count_tensions(obs_mass, limits_v2),
            'gap': mean_gap(obs_mass, limits_v2),
            'accel': accel_v2,
            'improvement': improvement_v2
        }

        # v3.0
        limits_v3 = max_stellar_mass_janus_v3(z_array, density_ratio=xi, coupling=1.0)
        chi2_v3 = compute_chi2_excess(obs_mass, limits_v3, obs_err)
        accel_v3 = acceleration_factor_v3(xi, 1.0)
        improvement_v3 = 100 * (chi2_lcdm - chi2_v3) / chi2_lcdm

        results_v3[xi] = {
            'chi2': chi2_v3,
            'tensions': count_tensions(obs_mass, limits_v3),
            'gap': mean_gap(obs_mass, limits_v3),
            'accel': accel_v3,
            'improvement': improvement_v3
        }

        print(f"ξ={xi:3d}: v2 χ²={chi2_v2:7.1f} (accel={accel_v2:5.2f}×) | "
              f"v3 χ²={chi2_v3:7.1f} (accel={accel_v3:5.2f}×) | "
              f"Δχ²={chi2_v3-chi2_v2:+6.1f}")

    print()

    # ========================================================================
    # ÉTAPE 3: Tests sensibilité - Variation χ (couplage) NOUVEAU v3.0
    # ========================================================================

    print("[3/5] Tests sensibilité: variation χ (couplage bimétrique) - NOUVEAU v3.0...")
    print()

    couplings = [0.5, 0.75, 1.0]
    results_coupling = {}

    for chi in couplings:
        limits = max_stellar_mass_janus_v3(z_array, density_ratio=64, coupling=chi)
        chi2 = compute_chi2_excess(obs_mass, limits, obs_err)
        accel = acceleration_factor_v3(64, chi)
        improvement = 100 * (chi2_lcdm - chi2) / chi2_lcdm

        results_coupling[chi] = {
            'chi2': chi2,
            'tensions': count_tensions(obs_mass, limits),
            'gap': mean_gap(obs_mass, limits),
            'accel': accel,
            'improvement': improvement
        }

        print(f"χ={chi:.2f} (ξ=64): χ²={chi2:7.1f}, accel={accel:.3f}×, amélioration={improvement:.1f}%")

    print()

    # Meilleur couplage
    best_coupling = min(results_coupling.keys(), key=lambda k: results_coupling[k]['chi2'])
    print(f"→ Meilleur couplage: χ = {best_coupling:.2f}")
    print(f"  χ² = {results_coupling[best_coupling]['chi2']:.1f}")
    print()

    # ========================================================================
    # ÉTAPE 4: Génération figures comparatives
    # ========================================================================

    print("[4/5] Génération figures...")

    plt.style.use('seaborn-v0_8-whitegrid')

    # FIGURE 1: Comparaison v2.0 vs v3.0 (masse-redshift)
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    z_range = np.linspace(10.5, 14.5, 200)

    # Subplot 1: Courbes v2 et v3 avec ξ=64
    ax1.plot(z_range, max_stellar_mass_lcdm(z_range), 'r--', linewidth=3,
             label='ΛCDM', zorder=2)
    ax1.plot(z_range, max_stellar_mass_janus_v2(z_range, 64), 'b--', linewidth=2,
             label=f'JANUS v2.0 (√ξ, accel={accel_v2_64:.2f}×)', zorder=3, alpha=0.7)
    ax1.plot(z_range, max_stellar_mass_janus_v3(z_range, 64, 1.0), 'purple', linewidth=3,
             label=f'JANUS v3.0 (√(1+ξ), accel={accel_v3_64:.2f}×)', zorder=4)

    ax1.errorbar(z_array, obs_mass, yerr=obs_err, fmt='o', markersize=12,
                capsize=6, capthick=2, color='black', ecolor='gray',
                alpha=0.9, label='JWST observations', zorder=10)

    ax1.set_xlabel('Redshift z', fontsize=14, fontweight='bold')
    ax1.set_ylabel('log₁₀(M*/M☉)', fontsize=14, fontweight='bold')
    ax1.set_title('JANUS v2.0 vs v3.0 (ξ = 64)', fontsize=16, fontweight='bold')
    ax1.set_xlim(10.4, 14.6)
    ax1.set_ylim(3.5, 10.2)
    ax1.legend(fontsize=11, loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Facteur d'accélération v2 vs v3
    xi_range = np.linspace(1, 300, 500)
    accel_v2_range = acceleration_factor_v2(xi_range)
    accel_v3_range = acceleration_factor_v3(xi_range, 1.0)

    ax2.plot(xi_range, accel_v2_range, 'b--', linewidth=2, label='v2.0: √ξ')
    ax2.plot(xi_range, accel_v3_range, 'purple', linewidth=3, label='v3.0: √(1+ξ)')
    ax2.axvline(64, color='red', linestyle=':', linewidth=2, alpha=0.5, label='ξ = 64 (historique)')

    ax2.set_xlabel('Rapport densité ξ = ρ₋/ρ₊', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Facteur accélération f_accel', fontsize=14, fontweight='bold')
    ax2.set_title('Comparaison formules v2.0 vs v3.0', fontsize=16, fontweight='bold')
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 300)

    plt.tight_layout()
    output_fig1 = '../results/figures/fig_V2_VS_V3_COMPARISON_20260103.pdf'
    plt.savefig(output_fig1, dpi=300, bbox_inches='tight')
    print(f"✓ Figure 1 sauvegardée: {output_fig1}")

    # FIGURE 2: χ² heatmap (ξ, χ) - NOUVEAU v3.0
    fig2, ax = plt.subplots(1, 1, figsize=(10, 8))

    xi_grid = np.linspace(10, 300, 50)
    chi_grid = np.linspace(0.1, 1.0, 30)
    chi2_grid = np.zeros((len(chi_grid), len(xi_grid)))

    for i, chi in enumerate(chi_grid):
        for j, xi in enumerate(xi_grid):
            limits = max_stellar_mass_janus_v3(z_array, density_ratio=xi, coupling=chi)
            chi2_grid[i, j] = compute_chi2_excess(obs_mass, limits, obs_err)

    im = ax.contourf(xi_grid, chi_grid, chi2_grid, levels=20, cmap='viridis_r')
    ax.contour(xi_grid, chi_grid, chi2_grid, levels=10, colors='white', alpha=0.3, linewidths=0.5)
    plt.colorbar(im, ax=ax, label='χ²')

    ax.scatter([64], [1.0], s=200, marker='*', color='red', edgecolors='white',
              linewidths=2, zorder=10, label='ξ=64, χ=1.0 (historique)')

    ax.set_xlabel('Rapport densité ξ = ρ₋/ρ₊', fontsize=14, fontweight='bold')
    ax.set_ylabel('Couplage bimétrique χ', fontsize=14, fontweight='bold')
    ax.set_title('χ² en fonction de (ξ, χ) - JANUS v3.0', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    output_fig2 = '../results/figures/fig_JANUS_V3_BIMETRIC_20260103.pdf'
    plt.savefig(output_fig2, dpi=300, bbox_inches='tight')
    print(f"✓ Figure 2 sauvegardée: {output_fig2}")
    print()

    # ========================================================================
    # ÉTAPE 5: Export résultats JSON
    # ========================================================================

    print("[5/5] Export résultats...")

    results_json = {
        'metadata': {
            'date': datetime.utcnow().isoformat(),
            'version': '3.0 - BIMÉTRIQUE AMÉLIORÉ',
            'n_galaxies': 16,
            'redshift_range': [float(z_array.min()), float(z_array.max())],
            'improvement': 'v2.0: √ξ → v3.0: √(1 + χ·ξ)'
        },
        'LCDM': {
            'chi2': float(chi2_lcdm),
            'tensions': int(tensions_lcdm),
            'gap_dex': float(gap_lcdm)
        },
        'JANUS_v2': {
            'formula': 'f_accel = √ξ',
            'xi_64': {
                'chi2': float(chi2_janus_v2),
                'tensions': int(tensions_janus_v2),
                'gap_dex': float(gap_janus_v2),
                'accel': float(accel_v2_64),
                'improvement_pct': float(100*(chi2_lcdm-chi2_janus_v2)/chi2_lcdm)
            },
            'sensitivity_xi': {str(k): {kk: float(vv) if isinstance(vv, (int, float)) else int(vv)
                                       for kk, vv in v.items()}
                              for k, v in results_v2.items()}
        },
        'JANUS_v3': {
            'formula': 'f_accel = √(1 + χ·ξ)',
            'theoretical_foundation': 'Derived from bimetric perturbation theory',
            'xi_64_chi_1': {
                'chi2': float(chi2_janus_v3),
                'tensions': int(tensions_janus_v3),
                'gap_dex': float(gap_janus_v3),
                'accel': float(accel_v3_64),
                'improvement_vs_LCDM_pct': float(100*(chi2_lcdm-chi2_janus_v3)/chi2_lcdm),
                'improvement_vs_v2_pct': float(100*(chi2_janus_v2-chi2_janus_v3)/chi2_janus_v2)
            },
            'sensitivity_xi': {str(k): {kk: float(vv) if isinstance(vv, (int, float)) else int(vv)
                                       for kk, vv in v.items()}
                              for k, v in results_v3.items()},
            'sensitivity_coupling': {str(k): {kk: float(vv) if isinstance(vv, (int, float)) else int(vv)
                                             for kk, vv in v.items()}
                                    for k, v in results_coupling.items()},
            'best_coupling': float(best_coupling)
        },
        'comparison': {
            'accel_difference_xi64': float(accel_v3_64 - accel_v2_64),
            'chi2_difference_xi64': float(chi2_janus_v3 - chi2_janus_v2),
            'percent_improvement_v3_over_v2': float(100*(chi2_janus_v2-chi2_janus_v3)/chi2_janus_v2)
        }
    }

    output_json = '../results/janus_v3_bimetric_20260103.json'
    with open(output_json, 'w') as f:
        json.dump(results_json, f, indent=2)

    print(f"✓ Résultats JSON: {output_json}")
    print()

    # ========================================================================
    # SYNTHÈSE FINALE
    # ========================================================================

    print("="*80)
    print("SYNTHÈSE - JANUS v3.0 BIMÉTRIQUE")
    print("="*80)
    print()
    print("AMÉLIORATION THÉORIQUE:")
    print("  ✓ Formule dérivée des équations de perturbation bimétrique")
    print("  ✓ Limite ΛCDM correcte (ξ → 0)")
    print("  ✓ Asymptote v2.0 correcte (ξ >> 1)")
    print()
    print(f"RÉSULTATS (ξ = 64, χ = 1.0):")
    print(f"  ΛCDM:        χ² = {chi2_lcdm:.1f}")
    print(f"  JANUS v2.0:  χ² = {chi2_janus_v2:.1f} (amélioration {100*(chi2_lcdm-chi2_janus_v2)/chi2_lcdm:.1f}%)")
    print(f"  JANUS v3.0:  χ² = {chi2_janus_v3:.1f} (amélioration {100*(chi2_lcdm-chi2_janus_v3)/chi2_lcdm:.1f}%)")
    print()
    print(f"AMÉLIORATION v3.0 vs v2.0:")
    print(f"  Δχ² = {chi2_janus_v3 - chi2_janus_v2:+.2f}")
    print(f"  Amélioration relative = {100*(chi2_janus_v2-chi2_janus_v3)/chi2_janus_v2:+.2f}%")
    print()
    print("INTERPRÉTATION:")
    if chi2_janus_v3 < chi2_janus_v2:
        print("  ✓ v3.0 meilleure que v2.0 (amélioration marginale attendue)")
    else:
        print("  ≈ v3.0 et v2.0 équivalentes (ξ grand → asymptote)")
    print("  ✓ Fondement théorique plus solide")
    print("  ✓ Nouveau paramètre χ à explorer")
    print()
    print("PROCHAINES ÉTAPES:")
    print("  1. Optimisation MCMC sur (ξ, χ)")
    print("  2. Cosmologie complète (Friedmann bimétrique)")
    print("  3. Publication v3.0")
    print()
    print("="*80)
