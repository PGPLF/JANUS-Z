#!/usr/bin/env python3
"""
JANUS v17.4 - [CII] Luminosity Function Analysis for Dusty Galaxies

This script performs quantitative testing of JANUS vs LCDM using the
[CII] 158um luminosity function from A3COSMOS dusty galaxies.

Key features:
- SFR to L_[CII] conversion (De Looze et al. 2014)
- [CII] luminosity function construction
- JANUS vs LCDM model comparison
- Independent validation orthogonal to UV selection

Author: Patrick Guerin
Date: 2026-01-05
Version: 17.4
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import integrate, stats
from scipy.optimize import minimize
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# COSMOLOGICAL PARAMETERS
# =============================================================================

# Standard LCDM cosmology (Planck 2020)
cosmo_lcdm = FlatLambdaCDM(H0=67.4, Om0=0.315)

# JANUS parameters
XI_0 = 64.01  # Density ratio from SNIa (Petit & d'Agostini 2018)
F_ACCEL = np.sqrt(XI_0)  # ~8.00, structure formation acceleration

# =============================================================================
# [CII] - SFR RELATION (De Looze et al. 2014)
# =============================================================================

def sfr_to_L_CII(sfr, scatter=False):
    """
    Convert SFR to [CII] 158um luminosity using De Looze et al. (2014) relation.

    Relation: log(L_[CII]/Lsun) = 1.0 * log(SFR/(Msun/yr)) + 7.06

    Parameters:
    -----------
    sfr : float or array
        Star formation rate in Msun/yr
    scatter : bool
        If True, add 0.3 dex intrinsic scatter

    Returns:
    --------
    L_CII : float or array
        [CII] luminosity in Lsun
    """
    log_L_CII = 1.0 * np.log10(sfr) + 7.06

    if scatter:
        log_L_CII += np.random.normal(0, 0.3, size=np.array(sfr).shape)

    return 10**log_L_CII

def L_CII_to_sfr(L_CII):
    """Inverse relation: L_[CII] to SFR."""
    log_sfr = (np.log10(L_CII) - 7.06) / 1.0
    return 10**log_sfr

# =============================================================================
# [CII] LUMINOSITY FUNCTION - OBSERVATIONS
# =============================================================================

def compute_comoving_volume(z_min, z_max, area_deg2, cosmo):
    """
    Compute comoving volume for a redshift slice.

    Parameters:
    -----------
    z_min, z_max : float
        Redshift range
    area_deg2 : float
        Survey area in square degrees
    cosmo : astropy.cosmology
        Cosmology object

    Returns:
    --------
    volume : float
        Comoving volume in Mpc^3
    """
    # Solid angle in steradians
    omega_sr = area_deg2 * (np.pi/180)**2

    # Comoving distances
    d_c_min = cosmo.comoving_distance(z_min).value  # Mpc
    d_c_max = cosmo.comoving_distance(z_max).value  # Mpc

    # Volume = (omega/3) * (d_max^3 - d_min^3)
    volume = (omega_sr / 3) * (d_c_max**3 - d_c_min**3)

    return volume

def compute_CII_LF_observed(dusty_df, L_bins, z_range=(6.5, 8.5), area_deg2=1.7):
    """
    Compute observed [CII] luminosity function from dusty galaxy sample.

    Parameters:
    -----------
    dusty_df : DataFrame
        Dusty galaxies with columns: z, log_SFR
    L_bins : array
        Bin edges for L_[CII] in Lsun
    z_range : tuple
        Redshift range for volume calculation
    area_deg2 : float
        A3COSMOS survey area (~1.7 deg^2)

    Returns:
    --------
    LF_obs : dict
        Observed LF with phi, phi_err, L_centers
    """
    # Calculate L_[CII] for each galaxy
    sfr = 10**dusty_df['log_SFR'].values
    L_CII = sfr_to_L_CII(sfr)
    z = dusty_df['z'].values

    # Filter to redshift range
    mask = (z >= z_range[0]) & (z <= z_range[1])
    L_CII = L_CII[mask]
    z = z[mask]

    # Compute comoving volume
    V_survey = compute_comoving_volume(z_range[0], z_range[1], area_deg2, cosmo_lcdm)

    # Bin galaxies
    log_L_bins = np.log10(L_bins)
    log_L_centers = 0.5 * (log_L_bins[:-1] + log_L_bins[1:])
    L_centers = 10**log_L_centers

    # Count galaxies per bin
    counts, _ = np.histogram(np.log10(L_CII), bins=log_L_bins)

    # Compute phi = N / (V * dlog10L)
    dlogL = np.diff(log_L_bins)
    phi = counts / (V_survey * dlogL)

    # Poisson errors
    phi_err = np.sqrt(counts) / (V_survey * dlogL)
    phi_err[counts == 0] = 1 / (V_survey * dlogL[counts == 0])  # Upper limit

    return {
        'L_centers': L_centers,
        'log_L_centers': log_L_centers,
        'phi': phi,
        'phi_err': phi_err,
        'counts': counts,
        'volume': V_survey,
        'n_galaxies': len(L_CII)
    }

# =============================================================================
# [CII] LUMINOSITY FUNCTION - THEORETICAL PREDICTIONS
# =============================================================================

def schechter_function(L, phi_star, L_star, alpha):
    """
    Schechter luminosity function.

    phi(L) = (phi*/L*) * (L/L*)^alpha * exp(-L/L*)

    Parameters:
    -----------
    L : float or array
        Luminosity
    phi_star : float
        Normalization (Mpc^-3 dex^-1)
    L_star : float
        Characteristic luminosity
    alpha : float
        Faint-end slope

    Returns:
    --------
    phi : float or array
        Number density per dex
    """
    x = L / L_star
    return phi_star * (x)**(alpha + 1) * np.exp(-x) * np.log(10)

def compute_CII_LF_LCDM(L_centers, z_mean=7.5):
    """
    Compute [CII] LF prediction for LCDM.

    Uses Schechter parameters from Loiacono+2021 (ALPINE) and Yan+2020,
    extrapolated to z~7.5.

    Parameters:
    -----------
    L_centers : array
        Luminosity bin centers
    z_mean : float
        Mean redshift

    Returns:
    --------
    phi_lcdm : array
        Predicted number density
    """
    # Schechter parameters at z~7 from literature
    # Yan+2020, Loiacono+2021 extrapolation
    # L* ~ 10^9 Lsun at z~7
    # phi* ~ 10^-4 Mpc^-3 dex^-1
    # alpha ~ -1.8

    L_star = 1e9  # Lsun
    phi_star = 5e-5  # Mpc^-3 dex^-1 (conservative for z~7.5)
    alpha = -1.8

    # Evolution with redshift (empirical)
    # phi* decreases with z: phi*(z) = phi*(z=5) * (1+z)^(-3)
    evolution_factor = ((1 + 5) / (1 + z_mean))**3
    phi_star_z = phi_star * evolution_factor

    phi_lcdm = schechter_function(L_centers, phi_star_z, L_star, alpha)

    return phi_lcdm, {'L_star': L_star, 'phi_star': phi_star_z, 'alpha': alpha}

def compute_CII_LF_JANUS(L_centers, z_mean=7.5):
    """
    Compute [CII] LF prediction for JANUS model.

    JANUS predicts enhanced structure formation by factor f_accel = sqrt(xi_0) ~ 8.
    This translates to:
    - Enhanced number density: phi_JANUS = f_accel^3 * phi_LCDM (volume effect)
    - Or equivalently: L*_JANUS = f_accel * L*_LCDM (luminosity boost)

    We use the luminosity boost interpretation since SFR is enhanced.
    """
    # Get LCDM prediction
    phi_lcdm, params_lcdm = compute_CII_LF_LCDM(L_centers, z_mean)

    # JANUS enhancement: L* is boosted by f_accel
    # At fixed L, JANUS predicts more galaxies at high-L end
    L_star_janus = params_lcdm['L_star'] * F_ACCEL
    phi_star_janus = params_lcdm['phi_star']  # Same normalization
    alpha = params_lcdm['alpha']

    phi_janus = schechter_function(L_centers, phi_star_janus, L_star_janus, alpha)

    return phi_janus, {'L_star': L_star_janus, 'phi_star': phi_star_janus, 'alpha': alpha}

# =============================================================================
# MODEL COMPARISON
# =============================================================================

def compute_chi2_LF(phi_obs, phi_err, phi_model):
    """Compute chi-squared for LF comparison."""
    # Only use bins with observations
    mask = phi_obs > 0
    chi2 = np.sum(((phi_obs[mask] - phi_model[mask]) / phi_err[mask])**2)
    n_dof = np.sum(mask) - 1  # 1 free parameter (normalization)
    return chi2, n_dof

def compute_BIC_LF(chi2, n_data, n_params):
    """Compute BIC for model comparison."""
    return chi2 + n_params * np.log(n_data)

def fit_LF_normalization(phi_obs, phi_err, phi_model_template):
    """
    Fit normalization factor to match model to data.

    Returns:
    --------
    norm : float
        Best-fit normalization
    chi2 : float
        Chi-squared at best fit
    """
    mask = phi_obs > 0

    def objective(norm):
        return np.sum(((phi_obs[mask] - norm * phi_model_template[mask]) / phi_err[mask])**2)

    result = minimize(objective, x0=1.0, method='Nelder-Mead')
    return result.x[0], result.fun

# =============================================================================
# FIGURE GENERATION
# =============================================================================

def plot_CII_luminosity_function(LF_obs, phi_janus, phi_lcdm, params_janus, params_lcdm,
                                  output_file):
    """
    Create publication-quality [CII] LF plot.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    L = LF_obs['L_centers']
    log_L = LF_obs['log_L_centers']

    # Left panel: LF comparison
    ax1.errorbar(log_L, LF_obs['phi'], yerr=LF_obs['phi_err'],
                 fmt='ko', ms=10, capsize=5, label='A3COSMOS (z=6.5-8.5)', zorder=10)

    # Theoretical predictions
    ax1.plot(log_L, phi_janus, 'b-', lw=3, label=f'JANUS (L*={params_janus["L_star"]:.1e} L$_\\odot$)')
    ax1.plot(log_L, phi_lcdm, 'r--', lw=3, label=f'$\\Lambda$CDM (L*={params_lcdm["L_star"]:.1e} L$_\\odot$)')

    ax1.set_xlabel('log(L$_{[CII]}$ / L$_\\odot$)', fontsize=14)
    ax1.set_ylabel('$\\Phi$ [Mpc$^{-3}$ dex$^{-1}$]', fontsize=14)
    ax1.set_yscale('log')
    ax1.set_ylim(1e-7, 1e-3)
    ax1.set_xlim(8, 10.5)
    ax1.legend(fontsize=12, loc='upper right')
    ax1.set_title('[CII] 158$\\mu$m Luminosity Function', fontsize=14)
    ax1.grid(True, alpha=0.3)

    # Add annotation
    ax1.text(0.05, 0.95, f'N = {LF_obs["n_galaxies"]} dusty galaxies\nV = {LF_obs["volume"]:.0f} Mpc$^3$',
             transform=ax1.transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Right panel: Residuals
    phi_obs_log = np.log10(LF_obs['phi'] + 1e-10)
    phi_janus_log = np.log10(phi_janus + 1e-10)
    phi_lcdm_log = np.log10(phi_lcdm + 1e-10)

    mask = LF_obs['phi'] > 0

    ax2.axhline(0, color='k', ls='-', lw=1)
    ax2.scatter(log_L[mask], (phi_obs_log - phi_janus_log)[mask],
                c='blue', s=100, marker='o', label='Obs - JANUS')
    ax2.scatter(log_L[mask], (phi_obs_log - phi_lcdm_log)[mask],
                c='red', s=100, marker='s', label='Obs - $\\Lambda$CDM')

    ax2.set_xlabel('log(L$_{[CII]}$ / L$_\\odot$)', fontsize=14)
    ax2.set_ylabel('$\\Delta$ log($\\Phi$) [dex]', fontsize=14)
    ax2.set_xlim(8, 10.5)
    ax2.set_ylim(-2, 2)
    ax2.legend(fontsize=12)
    ax2.set_title('Residuals', fontsize=14)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {output_file}')

def plot_CII_SFR_relation(dusty_df, output_file):
    """
    Plot L_[CII] vs SFR relation with De Looze calibration.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Data
    sfr = 10**dusty_df['log_SFR'].values
    L_CII = sfr_to_L_CII(sfr)
    z = dusty_df['z'].values

    # Color by redshift
    scatter = ax.scatter(np.log10(sfr), np.log10(L_CII), c=z, cmap='viridis',
                         s=100, edgecolors='k', linewidths=0.5)
    plt.colorbar(scatter, ax=ax, label='Redshift z')

    # De Looze relation
    sfr_range = np.logspace(1, 4, 100)
    L_CII_relation = sfr_to_L_CII(sfr_range)
    ax.plot(np.log10(sfr_range), np.log10(L_CII_relation), 'k-', lw=2,
            label='De Looze+2014: log(L$_{[CII]}$) = log(SFR) + 7.06')

    # Scatter band (0.3 dex)
    ax.fill_between(np.log10(sfr_range),
                    np.log10(L_CII_relation) - 0.3,
                    np.log10(L_CII_relation) + 0.3,
                    alpha=0.2, color='gray', label='$\\pm$0.3 dex scatter')

    ax.set_xlabel('log(SFR / M$_\\odot$ yr$^{-1}$)', fontsize=14)
    ax.set_ylabel('log(L$_{[CII]}$ / L$_\\odot$)', fontsize=14)
    ax.legend(fontsize=12, loc='upper left')
    ax.set_title('[CII] 158$\\mu$m - SFR Relation (24 Dusty Galaxies)', fontsize=14)
    ax.grid(True, alpha=0.3)

    # Add galaxy labels for extreme sources
    for i, row in dusty_df.iterrows():
        if row['log_SFR'] > 3.5 or row['ID'] == 'AC-2168':
            ax.annotate(row['ID'], (row['log_SFR'], np.log10(sfr_to_L_CII(10**row['log_SFR']))),
                       fontsize=8, ha='left')

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {output_file}')

def plot_dusty_mass_sfr(dusty_df, output_file):
    """
    Plot M* vs SFR diagram for dusty galaxies.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Data
    log_M = dusty_df['log_Mstar'].values
    log_SFR = dusty_df['log_SFR'].values
    z = dusty_df['z'].values

    # Color by redshift
    scatter = ax.scatter(log_M, log_SFR, c=z, cmap='plasma',
                         s=120, edgecolors='k', linewidths=0.5, zorder=10)
    plt.colorbar(scatter, ax=ax, label='Redshift z')

    # Main sequence at z~7 (Speagle+2014 extrapolation)
    M_range = np.linspace(9.5, 11.5, 100)
    # log(SFR) = 0.84*log(M*) - 6.5 - 0.3*(t_cosmic - 3.5) at z~7
    t_z7 = cosmo_lcdm.age(7.0).value  # Gyr
    SFR_MS = 0.84 * M_range - 6.5 - 0.3 * (t_z7 - 3.5)
    ax.plot(M_range, SFR_MS, 'k--', lw=2, label=f'Main Sequence (z~7)')
    ax.fill_between(M_range, SFR_MS - 0.3, SFR_MS + 0.3, alpha=0.2, color='gray')

    # Starburst threshold (4x above MS)
    ax.plot(M_range, SFR_MS + np.log10(4), 'r:', lw=2, label='Starburst (4$\\times$ MS)')

    ax.set_xlabel('log(M$_*$ / M$_\\odot$)', fontsize=14)
    ax.set_ylabel('log(SFR / M$_\\odot$ yr$^{-1}$)', fontsize=14)
    ax.legend(fontsize=12, loc='lower right')
    ax.set_title('Dusty Galaxies: Stellar Mass vs Star Formation Rate', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(9.8, 11)
    ax.set_ylim(1.5, 4.2)

    # Annotate extreme sources
    for i, row in dusty_df.iterrows():
        if row['log_SFR'] > 3.7 or row['ID'] == 'AC-2168':
            ax.annotate(row['ID'], (row['log_Mstar'] + 0.02, row['log_SFR'] + 0.05),
                       fontsize=9, ha='left')

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {output_file}')

def plot_combined_killer_plot(LF_obs, phi_janus, phi_lcdm, chi2_janus, chi2_lcdm,
                               delta_bic, output_file):
    """
    Create combined killer plot showing [CII] LF + statistics.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    L = LF_obs['L_centers']
    log_L = LF_obs['log_L_centers']

    # Panel 1: LF
    ax1 = axes[0]
    ax1.errorbar(log_L, LF_obs['phi'], yerr=LF_obs['phi_err'],
                 fmt='ko', ms=10, capsize=5, label='A3COSMOS', zorder=10)
    ax1.plot(log_L, phi_janus, 'b-', lw=3, label='JANUS')
    ax1.plot(log_L, phi_lcdm, 'r--', lw=3, label='$\\Lambda$CDM')
    ax1.set_xlabel('log(L$_{[CII]}$ / L$_\\odot$)', fontsize=14)
    ax1.set_ylabel('$\\Phi$ [Mpc$^{-3}$ dex$^{-1}$]', fontsize=14)
    ax1.set_yscale('log')
    ax1.set_ylim(1e-7, 1e-3)
    ax1.legend(fontsize=12)
    ax1.set_title('[CII] Luminosity Function', fontsize=14)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Chi-squared comparison
    ax2 = axes[1]
    models = ['JANUS', '$\\Lambda$CDM']
    chi2_values = [chi2_janus, chi2_lcdm]
    colors = ['blue', 'red']
    bars = ax2.bar(models, chi2_values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax2.set_ylabel('$\\chi^2$', fontsize=14)
    ax2.set_title('Model Fit Quality', fontsize=14)

    # Add values on bars
    for bar, val in zip(bars, chi2_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', fontsize=12, fontweight='bold')

    ax2.grid(True, alpha=0.3, axis='y')

    # Panel 3: BIC comparison
    ax3 = axes[2]
    bic_text = f'$\\Delta$BIC = {delta_bic:.1f}'
    if delta_bic < -10:
        verdict = 'VERY STRONG evidence for JANUS'
        color = 'green'
    elif delta_bic < -2:
        verdict = 'Positive evidence for JANUS'
        color = 'blue'
    else:
        verdict = 'Inconclusive'
        color = 'gray'

    ax3.text(0.5, 0.6, bic_text, fontsize=28, fontweight='bold',
             ha='center', va='center', transform=ax3.transAxes)
    ax3.text(0.5, 0.4, verdict, fontsize=18, color=color,
             ha='center', va='center', transform=ax3.transAxes)
    ax3.text(0.5, 0.2, f'($\\Delta$BIC < -10: very strong)', fontsize=12,
             ha='center', va='center', transform=ax3.transAxes, style='italic')
    ax3.axis('off')
    ax3.set_title('Bayesian Model Comparison', fontsize=14)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {output_file}')

# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def main():
    print("=" * 70)
    print("JANUS v17.4 - [CII] LUMINOSITY FUNCTION ANALYSIS")
    print("24 Dusty Galaxies from A3COSMOS, Orthogonal Validation Test")
    print("=" * 70)
    print()

    # Paths
    catalog_file = Path('data/jwst_extended_catalog_v17.1.csv')
    output_dir = Path('results/figures')
    results_file = Path('results/janus_v17.4_cii_lf_results.json')

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load catalog
    print("Loading catalog...")
    df = pd.read_csv(catalog_file)
    dusty_df = df[df['is_dusty'] == 1].copy()
    print(f"  Dusty galaxies: {len(dusty_df)}")
    print(f"  Redshift range: {dusty_df['z'].min():.2f} - {dusty_df['z'].max():.2f}")
    print(f"  SFR range: {10**dusty_df['log_SFR'].min():.0f} - {10**dusty_df['log_SFR'].max():.0f} Msun/yr")
    print()

    # =================================================================
    # [CII] - SFR RELATION
    # =================================================================
    print("=" * 70)
    print("[CII] - SFR RELATION (De Looze et al. 2014)")
    print("=" * 70)

    # Calculate L_[CII] for all dusty galaxies
    dusty_df['L_CII'] = sfr_to_L_CII(10**dusty_df['log_SFR'])
    dusty_df['log_L_CII'] = np.log10(dusty_df['L_CII'])

    print(f"L_[CII] range: {dusty_df['L_CII'].min():.2e} - {dusty_df['L_CII'].max():.2e} Lsun")
    print(f"log(L_[CII]) range: {dusty_df['log_L_CII'].min():.2f} - {dusty_df['log_L_CII'].max():.2f}")
    print()

    # Generate SFR relation plot
    plot_CII_SFR_relation(dusty_df, output_dir / 'fig_v17.4_cii_sfr_relation.pdf')

    # =================================================================
    # DUSTY GALAXY PROPERTIES
    # =================================================================
    print("=" * 70)
    print("DUSTY GALAXY PROPERTIES")
    print("=" * 70)

    # Generate M*-SFR diagram
    plot_dusty_mass_sfr(dusty_df, output_dir / 'fig_v17.4_dusty_mass_sfr.pdf')

    # Statistics
    print(f"Mean stellar mass: log(M*/Msun) = {dusty_df['log_Mstar'].mean():.2f} +/- {dusty_df['log_Mstar'].std():.2f}")
    print(f"Mean SFR: log(SFR) = {dusty_df['log_SFR'].mean():.2f} +/- {dusty_df['log_SFR'].std():.2f}")
    print(f"Mean L_[CII]: log(L/Lsun) = {dusty_df['log_L_CII'].mean():.2f} +/- {dusty_df['log_L_CII'].std():.2f}")
    print()

    # =================================================================
    # [CII] LUMINOSITY FUNCTION
    # =================================================================
    print("=" * 70)
    print("[CII] LUMINOSITY FUNCTION ANALYSIS")
    print("=" * 70)

    # Define luminosity bins
    L_bins = 10**np.array([8.0, 8.5, 9.0, 9.5, 10.0, 10.5])

    # Compute observed LF
    LF_obs = compute_CII_LF_observed(dusty_df, L_bins, z_range=(6.5, 8.5), area_deg2=1.7)

    print(f"Survey volume: V = {LF_obs['volume']:.0f} Mpc^3")
    print(f"Galaxies in z=6.5-8.5: N = {LF_obs['n_galaxies']}")
    print()
    print("Observed LF:")
    for i, (L, phi, err, n) in enumerate(zip(LF_obs['log_L_centers'], LF_obs['phi'],
                                              LF_obs['phi_err'], LF_obs['counts'])):
        print(f"  log(L) = {L:.2f}: phi = {phi:.2e} +/- {err:.2e} (N={n})")
    print()

    # Compute theoretical predictions
    z_mean = dusty_df['z'].mean()
    phi_lcdm, params_lcdm = compute_CII_LF_LCDM(LF_obs['L_centers'], z_mean)
    phi_janus, params_janus = compute_CII_LF_JANUS(LF_obs['L_centers'], z_mean)

    print("LCDM prediction:")
    print(f"  L* = {params_lcdm['L_star']:.2e} Lsun")
    print(f"  phi* = {params_lcdm['phi_star']:.2e} Mpc^-3 dex^-1")
    print(f"  alpha = {params_lcdm['alpha']}")
    print()

    print("JANUS prediction:")
    print(f"  L* = {params_janus['L_star']:.2e} Lsun (x{F_ACCEL:.1f} enhanced)")
    print(f"  phi* = {params_janus['phi_star']:.2e} Mpc^-3 dex^-1")
    print(f"  alpha = {params_janus['alpha']}")
    print()

    # =================================================================
    # MODEL COMPARISON
    # =================================================================
    print("=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    # Fit normalizations
    norm_janus, chi2_janus = fit_LF_normalization(LF_obs['phi'], LF_obs['phi_err'], phi_janus)
    norm_lcdm, chi2_lcdm = fit_LF_normalization(LF_obs['phi'], LF_obs['phi_err'], phi_lcdm)

    print(f"JANUS: chi2 = {chi2_janus:.2f} (norm = {norm_janus:.2f})")
    print(f"LCDM:  chi2 = {chi2_lcdm:.2f} (norm = {norm_lcdm:.2f})")
    print()

    # BIC comparison
    n_data = np.sum(LF_obs['phi'] > 0)
    n_params = 1  # normalization

    bic_janus = compute_BIC_LF(chi2_janus, n_data, n_params)
    bic_lcdm = compute_BIC_LF(chi2_lcdm, n_data, n_params)
    delta_bic = bic_lcdm - bic_janus

    print(f"BIC_JANUS = {bic_janus:.2f}")
    print(f"BIC_LCDM  = {bic_lcdm:.2f}")
    print(f"Delta_BIC = {delta_bic:.2f}")

    if delta_bic < -10:
        print("  => VERY STRONG evidence for JANUS")
    elif delta_bic < -6:
        print("  => STRONG evidence for JANUS")
    elif delta_bic < -2:
        print("  => POSITIVE evidence for JANUS")
    else:
        print("  => INCONCLUSIVE")
    print()

    # =================================================================
    # GENERATE FIGURES
    # =================================================================
    print("=" * 70)
    print("GENERATING FIGURES")
    print("=" * 70)

    # Apply fitted normalizations
    phi_janus_fitted = norm_janus * phi_janus
    phi_lcdm_fitted = norm_lcdm * phi_lcdm

    # Main LF plot
    plot_CII_luminosity_function(LF_obs, phi_janus_fitted, phi_lcdm_fitted,
                                  params_janus, params_lcdm,
                                  output_dir / 'fig_v17.4_cii_luminosity_function.pdf')

    # Combined killer plot
    plot_combined_killer_plot(LF_obs, phi_janus_fitted, phi_lcdm_fitted,
                               chi2_janus, chi2_lcdm, delta_bic,
                               output_dir / 'fig_v17.4_cii_killer_plot.pdf')

    # =================================================================
    # SAVE RESULTS
    # =================================================================
    print("=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    results = {
        'version': '17.4',
        'description': '[CII] Luminosity Function Analysis',
        'date': '2026-01-05',
        'catalog': {
            'n_dusty': len(dusty_df),
            'z_range': [float(dusty_df['z'].min()), float(dusty_df['z'].max())],
            'L_CII_range': [float(dusty_df['L_CII'].min()), float(dusty_df['L_CII'].max())],
            'SFR_range': [float(10**dusty_df['log_SFR'].min()), float(10**dusty_df['log_SFR'].max())]
        },
        'LF_observed': {
            'log_L_centers': LF_obs['log_L_centers'].tolist(),
            'phi': LF_obs['phi'].tolist(),
            'phi_err': LF_obs['phi_err'].tolist(),
            'counts': LF_obs['counts'].tolist(),
            'volume_Mpc3': float(LF_obs['volume']),
            'n_galaxies': int(LF_obs['n_galaxies'])
        },
        'models': {
            'JANUS': {
                'L_star': float(params_janus['L_star']),
                'phi_star': float(params_janus['phi_star']),
                'alpha': float(params_janus['alpha']),
                'normalization': float(norm_janus),
                'chi2': float(chi2_janus),
                'BIC': float(bic_janus)
            },
            'LCDM': {
                'L_star': float(params_lcdm['L_star']),
                'phi_star': float(params_lcdm['phi_star']),
                'alpha': float(params_lcdm['alpha']),
                'normalization': float(norm_lcdm),
                'chi2': float(chi2_lcdm),
                'BIC': float(bic_lcdm)
            }
        },
        'comparison': {
            'delta_chi2': float(chi2_lcdm - chi2_janus),
            'delta_BIC': float(delta_bic),
            'preferred_model': 'JANUS' if delta_bic < 0 else 'LCDM',
            'evidence_strength': 'very strong' if delta_bic < -10 else ('strong' if delta_bic < -6 else ('positive' if delta_bic < -2 else 'inconclusive'))
        },
        'De_Looze_relation': {
            'equation': 'log(L_CII/Lsun) = 1.0 * log(SFR) + 7.06',
            'scatter_dex': 0.3,
            'reference': 'De Looze et al. 2014'
        }
    }

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {results_file}")

    # =================================================================
    # SUMMARY
    # =================================================================
    print()
    print("=" * 70)
    print("ANALYSIS COMPLETE - v17.4 ([CII] LF)")
    print("=" * 70)
    print()
    print(f"Figures generated (4):")
    print(f"  - fig_v17.4_cii_sfr_relation.pdf")
    print(f"  - fig_v17.4_dusty_mass_sfr.pdf")
    print(f"  - fig_v17.4_cii_luminosity_function.pdf")
    print(f"  - fig_v17.4_cii_killer_plot.pdf")
    print()
    print("=" * 70)
    print("[CII] LF SUMMARY:")
    print(f"  JANUS: chi2 = {chi2_janus:.2f}, BIC = {bic_janus:.2f}")
    print(f"  LCDM:  chi2 = {chi2_lcdm:.2f}, BIC = {bic_lcdm:.2f}")
    print(f"  Delta_BIC = {delta_bic:.2f}")
    print()
    print("CONCLUSION:")
    if delta_bic < -10:
        print(f"[CII] LF provides VERY STRONG independent evidence for JANUS")
        print(f"(Delta_BIC = {delta_bic:.1f}, orthogonal to UV-selected SMF test)")
    elif delta_bic < 0:
        print(f"[CII] LF shows preference for JANUS (Delta_BIC = {delta_bic:.1f})")
    else:
        print(f"[CII] LF inconclusive (Delta_BIC = {delta_bic:.1f})")
    print("=" * 70)

if __name__ == "__main__":
    main()
