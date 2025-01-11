import sys
import os
import numpy as np
import math
from scipy.stats import norm
import matplotlib.pyplot as plt
import logging
from typing import Tuple, Callable

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def B_HW(t, T, alpha):
    """Bond scaling factor for constant alpha in the Hull-White/Vasicek model."""
    if T < t:
        raise ValueError("T must be greater than or equal to t")
    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    
    dt = T - t
    if dt == 0:
        return 0.0
        
    # Handle very large alpha to avoid numerical issues
    if alpha > 1e2:
        return 1.0 / alpha
        
    # Use higher-order Taylor series for small alpha
    if abs(alpha) < 1e-8:  
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt
        alpha2 = alpha * alpha
        alpha3 = alpha2 * alpha
        return dt * (1.0 - alpha * dt / 2.0 + alpha2 * dt2 / 6.0 - alpha3 * dt3 / 24.0)
        
    return (1.0 - np.exp(-alpha * dt)) / alpha

def integrated_variance(
    tgrid,      # array of time points, e.g. [0, ..., T]
    sigmaS,     # piecewise constant vol array matching intervals in tgrid
    alpha,      # mean reversion speed of short rate
    sigma_r,    # function sigma_r(t)
    B_func,     # function for B(t,T), e.g. B_HW
    rho         # correlation
):
    """
    Compute the integral of
       [sigma_S(t)^2 + 2 rho sigma_S(t)*B(t,T)*sigma_r(t) + (B(t,T)*sigma_r(t))^2]
    over t in [0, tgrid[-1]], using piecewise constant sigmaS(t) on sub-intervals.
    We'll do simple trapezoidal integration on each sub-interval [tgrid[i], tgrid[i+1]].
    """
    n = len(tgrid) - 1
    total = 0.0

    for i in range(n):
        t0 = tgrid[i]
        t1 = tgrid[i+1]
        sS = sigmaS[i]

        def integrand(t):
            Br = B_func(t, tgrid[-1], alpha)
            sr = sigma_r(t)
            return sS**2 + 2*rho*sS*Br*sr + (Br*sr)**2

        val0 = integrand(t0)
        val1 = integrand(t1)
        total += 0.5*(val0 + val1)*(t1 - t0)

    return total

def implied_vol(
    tgrid,
    sigmaS,     # piecewise constant vol array
    alpha,
    sigma_r,
    B_func,
    rho
):
    """Compute BS implied vol with improved numerical stability."""
    T = tgrid[-1]
    if T <= 0:
        raise ValueError("Maturity must be positive")
    
    var_T = integrated_variance(tgrid, sigmaS, alpha, sigma_r, B_func, rho)
    return np.sqrt(var_T / T)

def bs_call_price(F, K, T, vol):
    """Standard Black-Scholes call price with validation."""
    if T < 0:
        raise ValueError("Time to maturity must be non-negative")
    if vol < 0:
        raise ValueError("Volatility must be non-negative")
    if F <= 0 or K <= 0:
        raise ValueError("Forward price and strike must be positive")

    if T == 0.0:
        return max(F - K, 0.0)
    if vol == 0.0:
        return max(F - K, 0.0)

    d1 = (np.log(F/K) + 0.5*vol**2*T) / (vol*np.sqrt(T))
    d2 = d1 - vol*np.sqrt(T)
    return F*norm.cdf(d1) - K*norm.cdf(d2)

def option_price_hw_equity(
    S0,
    K,
    T,
    P0T,
    sigmaS,
    tgrid,
    alpha,
    sigma_r,
    B_func,
    rho
):
    """
    Price a European call under Hull-White + piecewise constant equity vol on [0..T].
    1) Compute the implied vol for maturity T.
    2) Price via standard Black-Scholes on forward F0 = S0 / P0T.
    3) Multiply by P0T to get present value.
    """
    volT = implied_vol(tgrid, sigmaS, alpha, sigma_r, B_func, rho)
    F0 = S0 / P0T
    call_bs = bs_call_price(F0, K, T, volT)
    return P0T * call_bs

def bootstrap_sigmaS(
    maturities,     # sorted array of maturities [T1, T2, ...]
    market_vols,    # corresponding implied vols
    alpha,
    sigma_r,
    B_func,
    rho,
    npts_per_interval=2
):
    """Bootstrap with improved convergence."""
    if len(maturities) != len(market_vols):
        raise ValueError("Maturities and market_vols must have same length")
    if not np.all(np.diff(maturities) > 0):
        raise ValueError("Maturities must be strictly increasing")
    if not np.all(market_vols > 0):
        raise ValueError("Market vols must be positive")

    sigmaS_vals = []
    prev_T = 0.0
    tgrid_full = np.array([0.0])

    for i, T in enumerate(maturities):
        local_tgrid = np.linspace(prev_T, T, npts_per_interval)
        # merge with global tgrid
        if tgrid_full[-1] < prev_T:
            tgrid_full = np.concatenate([tgrid_full, local_tgrid])
        else:
            tgrid_full = np.concatenate([tgrid_full[:-1], local_tgrid])

        # target integrated variance at T
        target_var = market_vols[i]**2 * T

        def leftover_integral(x):
            """
            Construct a piecewise-constant sigmaS function from 0..T:
              - sigmaS_vals[j] for [T_{j-1}, T_j], j < i
              - x for [T_{i-1}, T_i]
            Then integrate.
            """
            seg_times = [0.0]
            seg_times.extend(maturities[:i])
            seg_times.append(T)
            seg_times = np.array(seg_times)

            seg_sigma = []
            for j in range(i):
                seg_sigma.append(sigmaS_vals[j])
            seg_sigma.append(x)

            # Build sub-partitions for integrated_variance
            sub_tgrid = []
            sub_sigma = []
            for k in range(len(seg_times)-1):
                sub_tgrid.append(seg_times[k])
                sub_sigma.append(seg_sigma[k])
            sub_tgrid.append(seg_times[-1])

            sub_tgrid = np.array(sub_tgrid)
            sub_sigma = np.array(sub_sigma)

            return integrated_variance(sub_tgrid, sub_sigma, alpha, sigma_r, B_func, rho)

        def obj(x):
            return leftover_integral(x) - target_var

        # Solve via bisection
        x_low, x_high = 1e-6, 3.0
        x_mid = 0.0
        for _ in range(60):
            x_mid = 0.5*(x_low + x_high)
            f_mid = obj(x_mid)
            if abs(f_mid) < 1e-12:
                break
            f_low = obj(x_low)
            if f_low*f_mid < 0:
                x_high = x_mid
            else:
                x_low = x_mid

        sigmaS_vals.append(x_mid)
        prev_T = T

    return tgrid_full, sigmaS_vals

def extend_piecewise_constant(x: np.ndarray, y: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    """Extend piecewise constant function to new x points."""
    y_new = np.zeros_like(x_new)
    for i, xi in enumerate(x_new):
        idx = np.searchsorted(x, xi, side='right') - 1
        idx = max(0, idx)  # Handle points before first maturity
        idx = min(idx, len(y) - 1)  # Handle points after last maturity
        y_new[i] = y[idx]
    return y_new

def verify_calibration(
    maturities: np.ndarray,
    market_vols: np.ndarray,
    alpha: float,
    sigma_r: Callable[[float], float],
    rho: float,
    num_test_points: int = 100  # Increased for smoother plots
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Verify calibration by bootstrapping stock vols and recomputing implied vols."""
    logger.info("Starting calibration verification...")
    
    # Bootstrap stock volatilities
    _, sigmaS_vals = bootstrap_sigmaS(
        maturities, market_vols,
        alpha, sigma_r, B_HW, rho
    )
    
    # Create dense grid for testing (include points slightly beyond last maturity)
    test_maturities = np.linspace(0.0, maturities[-1] * 1.1, num_test_points)
    
    # Extend piecewise constant stock vols to test points
    stock_vols = extend_piecewise_constant(maturities, np.array(sigmaS_vals), test_maturities)
    
    # Recompute implied vols at test points
    recomputed_vols = np.zeros_like(test_maturities)
    for i, T in enumerate(test_maturities):
        if T > 0:  # Skip t=0 point
            test_tgrid = np.array([0.0, T])
            test_sigmaS = np.array([stock_vols[i]])
            recomputed_vols[i] = implied_vol(
                test_tgrid, test_sigmaS,
                alpha, sigma_r, B_HW, rho
            )
    
    # Linear interpolation of market vols for comparison
    interp_market_vols = np.interp(test_maturities, maturities, market_vols)
    
    logger.info("Calibration verification completed")
    return test_maturities, interp_market_vols, stock_vols, recomputed_vols

def plot_calibration_results(
    maturities: np.ndarray,
    market_vols: np.ndarray,
    test_maturities: np.ndarray,
    interp_market_vols: np.ndarray,
    stock_vols: np.ndarray,
    recomputed_vols: np.ndarray,
    alpha: float,
    sigma_r: Callable[[float], float],
    rho: float,
) -> None:
    """Plot calibration results in two separate subplots."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Volatility Term Structures
    ax1.plot(test_maturities, interp_market_vols, 'k--', label='Market Implied Vol')
    ax1.plot(test_maturities, stock_vols, 'b-', label='Stock Vol', linewidth=2)
    ax1.plot(maturities, market_vols, 'ko', markersize=8, markerfacecolor='white')
    
    ax1.set_xlabel('Maturity (years)')
    ax1.set_ylabel('Volatility')
    ax1.set_title('Market vs Stock Volatility')
    ax1.grid(True)
    ax1.legend(loc='upper left')
    
    # Plot 2: Calibration Check
    # Market vols
    ax2.plot(maturities, market_vols, 'ko', label='Market Implied Vols', 
             markersize=8, markerfacecolor='white')
    ax2.plot(test_maturities, interp_market_vols, 'k--', alpha=0.5, 
             label='Market (interpolated)')
    
    # Model implied vols at market maturities
    model_vols_at_market = np.array([
        implied_vol(
            np.array([0.0, T]),
            np.array([extend_piecewise_constant(maturities, stock_vols, np.array([T]))[0]]),
            alpha, sigma_r, B_HW, rho
        ) for T in maturities
    ])
    
    # Plot model vols with same style as market
    ax2.plot(maturities, model_vols_at_market, 'ro', 
             label='Model Implied Vols', markersize=8, markerfacecolor='white')
    ax2.plot(test_maturities, recomputed_vols, 'r--', alpha=0.5,
             label='Model (interpolated)')
    
    ax2.set_xlabel('Maturity (years)')
    ax2.set_ylabel('Implied Volatility')
    ax2.set_title('Calibration Verification')
    ax2.grid(True)
    ax2.legend(loc='upper left')
    
    # Add calibration error
    max_error = np.max(np.abs(model_vols_at_market - market_vols))
    ax2.text(0.02, 0.98, f'Max Error: {max_error:.2e}', 
             transform=ax2.transAxes,
             verticalalignment='top')
    
    plt.tight_layout()
    plt.show()

def main():
    """Main function to demonstrate the Hull-White + BS hybrid model."""
    try:
        logger.info("Starting Hull-White + BS hybrid model calculation")
        
        # Market data with realistic smile/skew effects
        maturities = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0])
        market_vols = np.array([0.22, 0.21, 0.20, 0.22, 0.23, 0.25])  # U-shaped curve
        alpha = 0.1
        rho = 0.3  # Correlation for rate-equity effects
        def sigma_r_func(t): return 0.015  # 1.5% rate vol

        # Verify calibration with more test points
        test_mats, interp_vols, stock_vols, recomp_vols = verify_calibration(
            maturities, market_vols,
            alpha, sigma_r_func, rho,
            num_test_points=200  # Increased for smoother curves
        )
        
        # Plot results with model parameters
        plot_calibration_results(
            maturities, market_vols,
            test_mats, interp_vols,
            stock_vols, recomp_vols,
            alpha, sigma_r_func, rho  # Added model parameters
        )
        
        logger.info("Calibration and plotting completed")
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
