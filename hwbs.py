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
    level=logging.INFO,  # Changed from INFO to DEBUG
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
    if abs(alpha) < 1e-8 and dt != 0:  # Modified condition
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt
        alpha2 = alpha * alpha
        alpha3 = alpha2 * alpha
        return dt * (1.0 - alpha * dt / 2.0 + alpha2 * dt2 / 6.0 - alpha3 * dt3 / 24.0)
        
    return (1.0 - np.exp(-alpha * dt)) / alpha

def integrate_simpson(f, t0: float, t1: float, n: int = 10) -> float:
    """Enhanced Simpson's rule with very fine grid."""
    epsilon = 1e-12  # Added epsilon to handle small integration ranges
    if (t1 - t0) < epsilon:
        return 0.0
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
    n = len(tgrid) - 1
    total = 0.0

    for i in range(n):
        t0 = tgrid[i]
        t1 = tgrid[i+1]
        sS = sigmaS[i]

        def integrand(t):
            epsilon = 1e-12  # Added epsilon to handle numerical precision
            if t > tgrid[-1] + epsilon:
                raise ValueError("t exceeds T beyond acceptable precision")
            t = min(t, tgrid[-1])  # Cap t to T to prevent T < t
            logger.debug(f"Integrand called with t={t}, T={tgrid[-1]}")
            Br = B_func(t, tgrid[-1], alpha)
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
        # Removed previous conditional; now just ensure no duplicate points
        if local_tgrid[0] <= tgrid_full[-1]:
            local_tgrid = local_tgrid[1:]
        tgrid_full = np.concatenate([tgrid_full, local_tgrid])

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
        
        # Verify accuracy
        test_vol = implied_vol(
            np.array([0.0, T]),
            np.array([x]),
            alpha, sigma_r, B_func, rho,
            method=method
        )
        error = abs(test_vol - market_vols[i])
        if error > 1e-6:
            logger.warning(f"Large calibration error at T={T}: {error:.2e}")

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
    num_test_points: int = 400,  # Doubled
    npts_per_interval: int = 20,  # Doubled
    method: str = "simpson"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Enhanced verification with finer grid."""
    logger.info("Starting calibration verification...")
    
    # Use more points for longer maturities
    adjusted_npts = [max(npts_per_interval, int(T * 10)) for T in maturities]
    
    # Bootstrap with enhanced accuracy
    _, sigmaS_vals = bootstrap_sigmaS(
        maturities, market_vols,
        alpha, sigma_r, B_HW, rho,
        npts_per_interval=max(adjusted_npts),
        method=method,
        tolerance=1e-14
    )
    
    # Create denser non-uniform test grid
    test_maturities = np.concatenate([
        np.linspace(0, 0.25, num_test_points // 3),  # More points near zero
        np.linspace(0.25, 1.0, num_test_points // 3),  # More points in middle
        np.linspace(1.0, maturities[-1], num_test_points // 3)  # Points for longer maturities
    ])
    test_maturities = np.unique(np.sort(test_maturities))
    
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
                alpha, sigma_r, B_HW, rho,
                method=method
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
    npts_per_interval: int,  # Added parameter
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
    ax1.legend(loc='lower left')
    
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
    ax2.set_title('Calibration Verification')  # Changed from setTitle to set_title
    ax2.grid(True)
    ax2.legend(loc='lower left')
    
    # Add calibration error and model parameters
    max_error = np.max(np.abs(model_vols_at_market - market_vols))
    info_text = (
        f'Max Error: {max_error:.2e}\n'
        f'Points per interval: {npts_per_interval}\n'
        f'α: {alpha:.3f}, ρ: {rho:.2f}'
    )
    # Move info text to middle right, with white background
    ax2.text(0.98, 0.5, info_text, 
             transform=ax2.transAxes,
             verticalalignment='center',
             horizontalalignment='right',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    
    plt.tight_layout()
    plt.show()

def create_test_data(max_maturity: float, num_points: int, constant_vol: float) -> Tuple[np.ndarray, np.ndarray]:
    """Create test data with constant spacing and volatility."""
    spacing = max_maturity / num_points
    maturities = np.linspace(spacing, max_maturity, num_points)
    market_vols = np.full_like(maturities, constant_vol)
    return maturities, market_vols

def main(maturities: np.ndarray, market_vols: np.ndarray):
    """Demonstrate the Hull-White equity model functionality."""
    try:
        logger.info("Starting Hull-White + BS hybrid model calculation")
        
        alpha = 0.01
        rho = 0.3
        def sigma_r_func(t): return 0.01
        npts_per_interval = 2
        # Verify calibration with more test points
        test_mats, interp_vols, stock_vols, recomp_vols = verify_calibration(
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
            npts_per_interval  # Added parameter
        )
        
        logger.info("Calibration and plotting completed")
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        # Create test data with 5Y max maturity, 20 points, 20% vol
        maturities, market_vols = create_test_data(5.0, 20, 0.20)
        main(maturities, market_vols)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
