#!/usr/bin/env python3
"""
BIC Calculator Utility for JANUS-Z

Computes Bayesian Information Criterion (BIC) for model comparison.

Usage:
    python BIC_calculator.py --chi2_model1 149547 --chi2_model2 29109 --k1 1 --k2 1 --n 236

    Or import as module:
    from BIC_calculator import compute_BIC, compare_models

Author: Patrick Guerin (pg@gfo.bzh)
Date: 2026-01-04
Version: 1.0 (v18)
"""

import argparse
import numpy as np
from typing import Tuple, Dict

def compute_BIC(chi2: float, k: int, n: int) -> float:
    """
    Compute Bayesian Information Criterion

    BIC = χ² + k ln(n)

    where:
    - χ² : Chi-square statistic
    - k   : Number of free parameters
    - n   : Number of data points (bins)

    Args:
        chi2: Chi-square value
        k: Number of free parameters
        n: Number of data points

    Returns:
        BIC value

    Example:
        >>> compute_BIC(chi2=149547, k=1, n=236)
        149552.47
    """
    return chi2 + k * np.log(n)

def delta_BIC(bic1: float, bic2: float) -> float:
    """
    Compute ΔBIC = BIC₁ - BIC₂

    Interpretation (Kass & Raftery 1995):
    |ΔBIC| < 2   : Weak evidence
    2 < |ΔBIC| < 6   : Positive evidence
    6 < |ΔBIC| < 10  : Strong evidence
    |ΔBIC| > 10      : Very strong evidence

    Args:
        bic1: BIC for model 1
        bic2: BIC for model 2

    Returns:
        ΔBIC (negative favors model 2)
    """
    return bic1 - bic2

def interpret_delta_BIC(delta_bic: float) -> str:
    """
    Interpret ΔBIC according to Kass & Raftery (1995) scale

    Args:
        delta_bic: ΔBIC value

    Returns:
        Interpretation string
    """
    abs_delta = abs(delta_bic)

    if abs_delta < 2:
        strength = "Weak"
    elif abs_delta < 6:
        strength = "Positive"
    elif abs_delta < 10:
        strength = "Strong"
    else:
        strength = "Very strong"

    if delta_bic < 0:
        favor = "model 2"
    else:
        favor = "model 1"

    return f"{strength} evidence for {favor}"

def compare_models(
    chi2_model1: float,
    chi2_model2: float,
    k1: int,
    k2: int,
    n: int,
    model1_name: str = "Model 1",
    model2_name: str = "Model 2"
) -> Dict[str, float]:
    """
    Complete model comparison using BIC

    Args:
        chi2_model1: Chi-square for model 1
        chi2_model2: Chi-square for model 2
        k1: Number of parameters in model 1
        k2: Number of parameters in model 2
        n: Number of data points
        model1_name: Name of model 1 (e.g., "JANUS")
        model2_name: Name of model 2 (e.g., "ΛCDM")

    Returns:
        Dictionary with comparison results

    Example:
        >>> results = compare_models(149547, 29109, 1, 1, 236, "JANUS", "ΛCDM")
        >>> print(results['interpretation'])
        'Very strong evidence for ΛCDM'
    """
    bic1 = compute_BIC(chi2_model1, k1, n)
    bic2 = compute_BIC(chi2_model2, k2, n)
    delta = delta_BIC(bic1, bic2)
    interpretation = interpret_delta_BIC(delta)

    # Reduced chi-square
    # Assuming N_dof = n - k (simplified; adjust if needed)
    chi2_red1 = chi2_model1 / (n - k1) if (n - k1) > 0 else np.inf
    chi2_red2 = chi2_model2 / (n - k2) if (n - k2) > 0 else np.inf

    results = {
        'chi2_model1': chi2_model1,
        'chi2_model2': chi2_model2,
        'chi2_red_model1': chi2_red1,
        'chi2_red_model2': chi2_red2,
        'k_model1': k1,
        'k_model2': k2,
        'n_data': n,
        'BIC_model1': bic1,
        'BIC_model2': bic2,
        'Delta_BIC': delta,
        'interpretation': interpretation,
        'model1_name': model1_name,
        'model2_name': model2_name
    }

    return results

def print_comparison(results: Dict[str, float]) -> None:
    """
    Pretty-print model comparison results

    Args:
        results: Dictionary from compare_models()
    """
    print("=" * 70)
    print("BIC MODEL COMPARISON RESULTS")
    print("=" * 70)
    print(f"\n{results['model1_name']}:")
    print(f"  χ² = {results['chi2_model1']:.2f}")
    print(f"  χ²_red = {results['chi2_red_model1']:.2f}")
    print(f"  k (parameters) = {results['k_model1']}")
    print(f"  BIC = {results['BIC_model1']:.2f}")

    print(f"\n{results['model2_name']}:")
    print(f"  χ² = {results['chi2_model2']:.2f}")
    print(f"  χ²_red = {results['chi2_red_model2']:.2f}")
    print(f"  k (parameters) = {results['k_model2']}")
    print(f"  BIC = {results['BIC_model2']:.2f}")

    print(f"\nComparison:")
    print(f"  ΔBIC = {results['Delta_BIC']:.2f}")
    print(f"  Interpretation: {results['interpretation']}")
    print(f"  Data points (n) = {results['n_data']}")

    print("\nKass & Raftery (1995) Scale:")
    print("  |ΔBIC| < 2      : Weak evidence")
    print("  2 < |ΔBIC| < 6  : Positive evidence")
    print("  6 < |ΔBIC| < 10 : Strong evidence")
    print("  |ΔBIC| > 10     : Very strong evidence")
    print("=" * 70)

def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description='Compute BIC for model comparison (JANUS-Z)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # JANUS vs ΛCDM comparison (v18)
  python BIC_calculator.py --chi2_model1 149547 --chi2_model2 29109 --k1 1 --k2 1 --n 236 --name1 JANUS --name2 ΛCDM

  # Quick BIC calculation
  python BIC_calculator.py --chi2 149547 --k 1 --n 236
        """
    )

    # Single model BIC
    parser.add_argument('--chi2', type=float, help='Chi-square value (single model)')
    parser.add_argument('--k', type=int, help='Number of parameters (single model)')
    parser.add_argument('--n', type=int, help='Number of data points')

    # Model comparison
    parser.add_argument('--chi2_model1', type=float, help='Chi-square for model 1')
    parser.add_argument('--chi2_model2', type=float, help='Chi-square for model 2')
    parser.add_argument('--k1', type=int, help='Parameters in model 1')
    parser.add_argument('--k2', type=int, help='Parameters in model 2')
    parser.add_argument('--name1', type=str, default='Model 1', help='Name of model 1')
    parser.add_argument('--name2', type=str, default='Model 2', help='Name of model 2')

    args = parser.parse_args()

    # Check if single model or comparison
    if args.chi2 is not None and args.k is not None and args.n is not None:
        # Single model BIC
        bic = compute_BIC(args.chi2, args.k, args.n)
        print(f"BIC = {bic:.2f}")
        print(f"  (χ² = {args.chi2:.2f}, k = {args.k}, n = {args.n})")

    elif all([args.chi2_model1, args.chi2_model2, args.k1, args.k2, args.n]):
        # Model comparison
        results = compare_models(
            args.chi2_model1,
            args.chi2_model2,
            args.k1,
            args.k2,
            args.n,
            args.name1,
            args.name2
        )
        print_comparison(results)

    else:
        parser.print_help()
        print("\nError: Provide either (--chi2, --k, --n) for single BIC or all comparison arguments")
        return 1

    return 0

if __name__ == '__main__':
    exit(main())
