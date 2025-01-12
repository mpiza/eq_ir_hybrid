"""
Hull-White Equity Model with Stochastic Interest Rates

This module implements a hybrid equity model combining:
1. Hull-White (extended Vasicek) interest rate model
2. Piecewise constant equity volatility
3. Constant correlation between equity and rates

The model allows for:
- Bootstrapping of equity volatilities from market data
- Pricing of equity options with stochastic rates
- Calibration verification and visualization
"""

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

def B_HW(t: float, T: float, alpha: float) -> float:
    """
    Calculate the Hull-White bond scaling factor B(t,T).
    
    In the Hull-White model, this represents the sensitivity of bond prices
    to the short rate: B(t,T) = (1 - exp(-alpha*(T-t))) / alpha
    
    Args:
        t: Start time
        T: End time
        alpha: Mean reversion speed
        
    Returns:
        float: Bond scaling factor
        
    Raises:
        ValueError: If T < t or alpha < 0
    """
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

def integrate_simpson(f, t0: float, t1: float, n: int = 10) -> float:
    """Enhanced Simpson's rule with very fine grid."""
    # Use much finer grid
    dt = t1 - t0
    n = max(n, int(100 * dt))  # Increased from 50 to 100 points per year
    
    # Split into more subintervals
    n_sub = 10  # Increased from 4 to 10
    sub_results = []
    
    for i in range(n_sub):
        t_start = t0 + (i * dt / n_sub)
        t_end = t0 + ((i + 1) * dt / n_sub)
        
        h = (t_end - t_start) / n
        x = np.linspace(t_start, t_end, n+1)
        y = np.array([f(xi) for xi in x])
        
        result = h/3 * (y[0] + y[-1] + 4*sum(y[1:-1:2]) + 2*sum(y[2:-1:2]))
        sub_results.append(result)
    
    return sum(sub_results)

def integrated_variance(
    tgrid: np.ndarray,
    sigmaS: np.ndarray,
    alpha: float,
    sigma_r: Callable[[float], float],
    B_func: Callable[[float, float, float], float],
    rho: float,
    method: str = "trapezoid"
) -> float:
    """
    Compute the integrated variance for the hybrid equity-rate model.
    
    This function calculates the total integrated instantaneous variance
    of log(S(T)) under the T-forward measure, accounting for:
    - Equity volatility contribution
    - Interest rate volatility contribution
    - Cross-term due to correlation
    
    Args:
        tgrid: Array of time points [0, t1, ..., T]
        sigmaS: Array of piecewise constant equity vols
        alpha: Hull-White mean reversion speed
        sigma_r: Function t -> sigma_r(t) for rate volatility
        B_func: Bond scaling factor function (e.g., B_HW)
        rho: Correlation between equity and rates
        method: Integration method ("trapezoid" or "simpson")
        
    Returns:
        float: Total integrated variance
    """
    T = tgrid[-1]  # Final maturity
    n = len(tgrid) - 1
    total = 0.0

    for i in range(n):
        t0 = tgrid[i]
        t1 = tgrid[i+1]
        sS = sigmaS[i]

        def integrand(t):
            # Ensure t is never greater than T when computing B
            t_bounded = min(t, T)
            Br = B_func(t_bounded, T, alpha)
            sr = sigma_r(t)
            return sS**2 + 2*rho*sS*Br*sr + (Br*sr)**2

        if method == "simpson":
            total += integrate_simpson(integrand, t0, t1)
        else:  # trapezoid
            val0 = integrand(t0)
            val1 = integrand(t1)
            total += 0.5*(val0 + val1)*(t1 - t0)

    return total

def implied_vol(
    tgrid: np.ndarray,
    sigmaS: np.ndarray,
    alpha: float,
    sigma_r: Callable[[float], float],
    B_func: Callable[[float, float, float], float],
    rho: float,
    method: str = "trapezoid"
) -> float:
    """
    Compute Black-Scholes implied volatility for given parameters.
    
    The implied volatility is computed as sqrt(integrated_variance / T),
    where T is the option maturity (last point in tgrid).
    
    Args:
        tgrid: Array of time points [0, t1, ..., T]
        sigmaS: Array of piecewise constant equity vols
        alpha: Hull-White mean reversion speed
        sigma_r: Rate volatility function
        B_func: Bond scaling factor function
        rho: Equity-rate correlation
        method: Integration method ("trapezoid" or "simpson")
        
    Returns:
        float: Black-Scholes implied volatility
        
    Raises:
        ValueError: If maturity T <= 0
    """
    T = tgrid[-1]
    if T <= 0:
        raise ValueError("Maturity must be positive")
    
    var_T = integrated_variance(tgrid, sigmaS, alpha, sigma_r, B_func, rho, method=method)
    return np.sqrt(var_T / T)

def bs_call_price(F: float, K: float, T: float, vol: float) -> float:
    """
    Calculate Black-Scholes call option price on a forward.
    
    Args:
        F: Forward price
        K: Strike price
        T: Time to maturity
        vol: Volatility
        
    Returns:
        float: Call option price
        
    Raises:
        ValueError: If T < 0, vol < 0, F <= 0, or K <= 0
    """
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
    S0: float,
    K: float,
    T: float,
    P0T: float,
    sigmaS: np.ndarray,
    tgrid: np.ndarray,
    alpha: float,
    sigma_r: Callable[[float], float],
    B_func: Callable[[float, float, float], float],
    rho: float,
    method: str = "trapezoid"
) -> float:
    """
    Price a European call under the hybrid Hull-White equity model.
    
    The pricing is done in three steps:
    1. Compute implied vol including rate effects
    2. Transform to forward measure using P0T
    3. Apply Black-Scholes formula
    
    Args:
        S0: Initial stock price
        K: Strike price
        T: Option maturity
        P0T: Zero-coupon bond price P(0,T)
        sigmaS: Array of piecewise constant equity vols
        tgrid: Array of time points
        alpha: Hull-White mean reversion
        sigma_r: Rate volatility function
        B_func: Bond scaling function
        rho: Equity-rate correlation
        method: Integration method ("trapezoid" or "simpson")
        
    Returns:
        float: Call option price
    """
    volT = implied_vol(tgrid, sigmaS, alpha, sigma_r, B_func, rho, method=method)
    F0 = S0 / P0T
    call_bs = bs_call_price(F0, K, T, volT)
    return P0T * call_bs

def bootstrap_sigmaS(
    maturities: np.ndarray,
    market_vols: np.ndarray,
    alpha: float,
    sigma_r: Callable[[float], float],
    B_func: Callable[[float, float, float], float],
    rho: float,
    npts_per_interval: int = 10,  # Increased default
    method: str = "simpson",
    tolerance: float = 1e-14,  # Tighter tolerance
    max_iterations: int = 200  # More iterations
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Enhanced bootstrapping with better convergence for flat volatility structures.
    """
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

        def piecewise_integrated_variance(x):
            """Calculate integrated variance using piecewise-constant volatility function.
            
            Constructs a piecewise-constant sigmaS (volatility) function from 0 to T by:
            - Using known volatilities sigmaS_vals[j] for intervals [T_{j-1}, T_j], j < i
            - Using parameter x as volatility for the interval [T_{i-1}, T_i]
            Then computes the integrated variance over the entire period.
            
            Parameters
            ----------
            x : float
                Volatility value for the current calibration interval
            
            Returns
            -------
            float
                Integrated variance over [0,T] using the constructed piecewise volatility
            """
            # Construct time grid points including 0, known maturities, and final time T
            seg_times = [0.0]
            seg_times.extend(maturities[:i])
            seg_times.append(T)
            seg_times = np.array(seg_times)
            seg_times.sort()  # Ensure times are strictly increasing

            # Build array of volatilities for each segment
            seg_sigma = []
            for j in range(i):
                seg_sigma.append(sigmaS_vals[j])
            seg_sigma.append(x)

            # Construct arrays for integrated_variance calculation
            sub_tgrid = []
            sub_sigma = []
            for k in range(len(seg_times)-1):
                sub_tgrid.append(seg_times[k])
                sub_sigma.append(seg_sigma[k])
            sub_tgrid.append(seg_times[-1])

            sub_tgrid = np.array(sub_tgrid)
            sub_sigma = np.array(sub_sigma)

            
            return integrated_variance(sub_tgrid, sub_sigma, alpha, sigma_r, B_func, rho, method=method)

        def obj(x):
            return piecewise_integrated_variance(x) - target_var

        # Use more sophisticated initial guess
        if i == 0:
            # For first interval, account for rate effects
            def simple_variance(x):
                t = T
                Br = B_func(t, t, alpha)
                sr = sigma_r(t)
                return x**2 + 2*rho*x*Br*sr + (Br*sr)**2
            
            x_init = np.sqrt(market_vols[0]**2 - simple_variance(0))
        else:
            x_init = sigmaS_vals[-1]
        
        # Newton-Raphson with fallback to bisection
        x = x_init
        for iter in range(max_iterations):
            f = obj(x)
            if abs(f) < tolerance:
                break
                
            # Compute numerical derivative
            h = max(1e-7, abs(x) * 1e-7)
            df = (obj(x + h) - f) / h
            
            if abs(df) > 1e-10:  # If derivative is meaningful
                x_new = x - f/df
                if x_new > 0:  # Only accept positive volatility
                    x = x_new
                    continue
            
            # Fallback to bisection if Newton fails
            x_low, x_high = x_init * 0.5, x_init * 2.0
            for _ in range(50):
                x = 0.5 * (x_low + x_high)
                f = obj(x)
                if abs(f) < tolerance:
                    break
                if f > 0:
                    x_high = x
                else:
                    x_low = x
        
        sigmaS_vals.append(x)
        
        # Verify accuracy using the same approach as in optimization
        seg_times = [0.0]
        seg_times.extend(maturities[:i])
        seg_times.append(T)
        seg_times = np.array(seg_times)
        
        test_vols = []
        for j in range(i):
            test_vols.append(sigmaS_vals[j])
        test_vols.append(x)
        
        test_vol = implied_vol(
            np.array(seg_times),
            np.array(test_vols),
            alpha, sigma_r, B_func, rho,
            method=method
        )
        
        error = abs(test_vol - market_vols[i])
        if error > 1e-6:
            logger.warning(f"Large calibration error at T={T}: {error:.2e}")

    print(f"tgrid_full: {tgrid_full}, sigmaS_vals: {sigmaS_vals}")
    return tgrid_full, sigmaS_vals

def extend_piecewise_constant(x: np.ndarray, y: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    """
    Extend a piecewise constant function to new x points.
    
    Args:
        x: Original x points defining intervals
        y: Values for each interval
        x_new: New x points to evaluate at
        
    Returns:
        np.ndarray: Values at x_new points
    """
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
    num_test_points: int = 400,
    npts_per_interval: int = 20,
    method: str = "simpson"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:  # Added sigmaS_vals to return
    """Enhanced verification with finer grid."""
    logger.info("Starting calibration verification...")
    
    # Bootstrap with enhanced accuracy
    _, sigmaS_vals = bootstrap_sigmaS(
        maturities, market_vols,
        alpha, sigma_r, B_HW, rho,
        npts_per_interval=npts_per_interval,
        method=method,
        tolerance=1e-14
    )
    
    # Create test grid properly spanning all intervals
    pieces = []
    # Start from a small positive number instead of 0 to avoid the spike
    eps = 1e-6  # Small positive number
    for i in range(len(maturities)-1):
        pieces.append(np.linspace(maturities[i], maturities[i+1], num_test_points//len(maturities)))
    # Add points from eps (near zero) to first maturity
    pieces.insert(0, np.linspace(eps, maturities[0], num_test_points//len(maturities)))
    
    test_maturities = np.unique(np.sort(np.concatenate(pieces)))
    
    # Extend stock vols to test points
    stock_vols = extend_piecewise_constant(maturities, np.array(sigmaS_vals), test_maturities)
    
    # Recompute implied vols at test points using full history
    recomputed_vols = np.zeros_like(test_maturities)
    for i, T in enumerate(test_maturities):
        if T > 0:  # Changed to handle very small but positive T
            idx = np.searchsorted(maturities, T)
            # Build time grid ensuring strict ordering and proper bounds
            prev_times = maturities[:idx]
            prev_times = prev_times[prev_times < T]
            times = np.array([0.0] + list(prev_times) + [T])
            
            # Build vols array matching the time grid structure
            vols = np.zeros(len(times)-1)  # One less than times since we need vol per interval
            for j in range(len(vols)):
                t_mid = 0.5 * (times[j] + times[j+1])  # Use midpoint to determine vol
                idx_vol = np.searchsorted(maturities, t_mid) - 1
                idx_vol = max(0, min(idx_vol, len(sigmaS_vals)-1))
                vols[j] = sigmaS_vals[idx_vol]
            
            recomputed_vols[i] = implied_vol(
                times,
                vols,
                alpha, sigma_r, B_HW, rho,
                method=method
            )
    
    # For the first point (T ≈ 0), use the limit value
    if len(recomputed_vols) > 0:
        recomputed_vols[0] = recomputed_vols[1]  # Use the next point's value
    
    # Linear interpolation of market vols
    interp_market_vols = np.interp(test_maturities, maturities, market_vols)
    
    logger.info("Calibration verification completed")
    print(f"test_maturities: {test_maturities}, interp_market_vols: {interp_market_vols}, stock_vols: {stock_vols}, recomputed_vols: {recomputed_vols}")
    return test_maturities, interp_market_vols, stock_vols, recomputed_vols, sigmaS_vals  # Added sigmaS_vals

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
    npts_per_interval: int,  # Added parameter
    sigmaS_vals: np.ndarray,  # Added parameter
) -> None:
    """
    Plot calibration results in two panels.
    
    Creates two side-by-side plots:
    1. Market implied vols vs bootstrapped stock vols
    2. Market vs model implied vols with error metrics
    
    Args:
        maturities: Option maturities
        market_vols: Market implied vols
        test_maturities: Dense time grid for plotting
        interp_market_vols: Interpolated market vols
        stock_vols: Bootstrapped stock vols
        recomputed_vols: Model implied vols
        alpha: Hull-White mean reversion
        sigma_r: Rate volatility function
        rho: Equity-rate correlation
        npts_per_interval: Number of points per interval in bootstrapping
        sigmaS_vals: Bootstrapped sigmaS values
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Volatility Term Structures
    ax1.plot(test_maturities, interp_market_vols, 'k--', label='Market Implied Vol')
    ax1.plot(test_maturities, stock_vols, 'b-', label='Stock Vol', linewidth=2)
    ax1.plot(maturities, market_vols, 'ko', markersize=8, markerfacecolor='white')
    
    ax1.set_xlabel('Maturity (years)')
    ax1.set_ylabel('Volatility')
    ax1.set_title('Market vs Stock Volatility')  # Changed from setTitle to set_title
    ax1.grid(True)
    ax1.legend(loc='upper left')
    
    # Plot 2: Volatility Differences
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.2)  # Zero line
    
    # Calculate model vols and differences consistently
    model_vols = np.zeros_like(test_maturities)
    
    # First calculate model vols exactly at market points
    model_vols_at_market = np.zeros_like(maturities)
    for i, T in enumerate(maturities):
        if T > 0:  # Skip T=0
            times = np.array([0.0] + list(maturities[:i]) + [T])
            vols = np.array([sigmaS_vals[0]] + sigmaS_vals[:i] + [sigmaS_vals[min(i, len(sigmaS_vals)-1)]])
            
            model_vols_at_market[i] = implied_vol(
                times, vols,
                alpha, sigma_r, B_HW, rho,
                method="simpson"
            )
    
    # Then calculate model vols for all test points
    for i, T in enumerate(test_maturities):
        if T > 0:  # Skip T=0
            idx = np.searchsorted(maturities, T)
            times = np.array([0.0] + list(maturities[:idx]) + [T])
            times.sort()  # Ensure times are strictly increasing
            vols = np.array([sigmaS_vals[0]] + sigmaS_vals[:idx] + [sigmaS_vals[min(idx, len(sigmaS_vals)-1)]])
            
            model_vols[i] = implied_vol(
                times, vols,
                alpha, sigma_r, B_HW, rho,
                method="simpson"
            )
    
    # Calculate differences in basis points
    diff_at_test = (model_vols - interp_market_vols) * 10000
    diff_at_market = (model_vols_at_market - market_vols) * 10000
    
    # Plot differences
    ax2.plot(test_maturities, diff_at_test, 'r-', label='Model - Market', alpha=0.5)
    ax2.plot(maturities, diff_at_market, 'ro', 
             label='At Market Points', markersize=8, markerfacecolor='white')
    
    ax2.set_xlabel('Maturity (years)')
    ax2.set_ylabel('Difference (bps)')
    ax2.set_title('Calibration Error')
    ax2.grid(True)
    ax2.legend(loc='upper left')
    
    # Add calibration error and model parameters
    max_error = np.max(np.abs(diff_at_market))
    mean_error = np.mean(np.abs(diff_at_market))
    rmse = np.sqrt(np.mean(diff_at_market**2))
    info_text = (
        f'Max Error: {max_error:.1f} bps\n'
        f'Mean Error: {mean_error:.1f} bps\n'
        f'RMSE: {rmse:.1f} bps\n'
        f'α: {alpha:.3f}, ρ: {rho:.2f}'
    )
    ax2.text(0.02, 0.98, info_text, 
             transform=ax2.transAxes,
             verticalalignment='top')
    
    plt.tight_layout()
    plt.show()

def main():
    """
    Demonstrate the Hull-White equity model functionality.
    
    Creates sample market data and shows:
    1. Bootstrapping of stock vols
    2. Calibration verification
    3. Visualization of results
    """
    try:
        logger.info("Starting Hull-White + BS hybrid model calculation")
        
        # Market data with realistic smile/skew effects
        maturities = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0])
        market_vols = np.array([0.22, 0.21, 0.20, 0.22, 0.23, 0.25])  # U-shaped curve
        alpha = 0.1
        rho = 0.3  # Correlation for rate-equity effects
        def sigma_r_func(t): return 0.015  # 1.5% rate vol
        npts_per_interval = 2  # Number of points per interval in bootstrapping

        # Verify calibration with more test points
        test_mats, interp_vols, stock_vols, recomp_vols, sigmaS_vals = verify_calibration(  # Added sigmaS_vals
            maturities, market_vols,
            alpha, sigma_r_func, rho,
            num_test_points=400,  # Increased
            npts_per_interval=20,  # Increased
            method="simpson"
        )
        
        # Plot results with model parameters
        plot_calibration_results(
            maturities, market_vols,
            test_mats, interp_vols,
            stock_vols, recomp_vols,
            alpha, sigma_r_func, rho,  # Added model parameters
            npts_per_interval,  # Added parameter
            sigmaS_vals  # Added parameter
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
