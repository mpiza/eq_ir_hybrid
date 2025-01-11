import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, messagebox
import logging
import json
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import multiprocessing
import os
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class OptionParameters:
    """Data class for option parameters with validation."""
    S0: float
    K: float
    T: float
    sigma1: float
    sigma2: float
    r0: float
    a: float
    theta: float
    rho: float

    def __post_init__(self):
        """Validate parameters after initialization."""
        if self.S0 <= 0:
            raise ValueError("Initial stock price must be positive")
        if self.K <= 0:
            raise ValueError("Strike price must be positive")
        if self.T <= 0:
            raise ValueError("Time to maturity must be positive")
        if self.sigma1 < 0 or self.sigma2 < 0:
            raise ValueError("Volatilities must be non-negative")
        if not -1 <= self.rho <= 1:
            raise ValueError("Correlation must be between -1 and 1")

class BlackScholesStochastic:
    """Class implementing Black-Scholes model with stochastic interest rates."""
    
    def __init__(self, params: OptionParameters):
        self.params = params
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        """Additional model-specific parameter validation."""
        try:
            self.params.__post_init__()
        except ValueError as e:
            logger.error(f"Parameter validation failed: {str(e)}")
            raise

    @staticmethod
    @lru_cache(maxsize=128)
    def _B(t: float, T: float, a: float) -> float:
        """Calculate B(t, T, a) with special case handling."""
        if abs(a) < 1e-10:  # Handle near-zero a values
            return T - t
        return (1 - np.exp(-a * (T - t))) / a

    @staticmethod
    @lru_cache(maxsize=128)
    def _c(t: float, a: float, theta: float, r0: float) -> float:
        """Calculate c(t, a, theta, r0) with special case handling."""
        if abs(a) < 1e-10:  # Handle near-zero a values
            return r0 + theta * t
        return r0 * np.exp(-a * t) + theta / a * (1 - np.exp(-a * t))

    def _compute_variance(self, T: float) -> float:
        """Compute variance of log(S(T)) under the forward measure."""
        try:
            integral_B2 = (1 - np.exp(-2 * self.params.a * T)) / (2 * self.params.a)
            variance = (self.params.sigma1**2 * T + 
                       self.params.sigma2**2 * integral_B2 + 
                       2 * self.params.sigma1 * self.params.sigma2 * 
                       self.params.rho * self._B(0, T, self.params.a))
            return variance
        except Exception as e:
            logger.error(f"Error computing variance: {str(e)}")
            raise

    def _forward_price(self, T: float) -> float:
        """Calculate forward price."""
        try:
            integral_B = self._B(0, T, self.params.a)
            integral_c = (self._c(T, self.params.a, self.params.theta, self.params.r0) * T - 
                         self.params.sigma2**2 * integral_B**2 / 2)
            return self.params.S0 * np.exp(integral_c)
        except Exception as e:
            logger.error(f"Error computing forward price: {str(e)}")
            raise

    def calculate_price(self) -> float:
        """Calculate option price with error handling."""
        try:
            F0_T = self._forward_price(self.params.T)
            nu_T = self._compute_variance(self.params.T)
            kappa = self.params.K / F0_T

            d_plus = (-np.log(kappa) + 0.5 * nu_T) / np.sqrt(nu_T)
            d_minus = d_plus - np.sqrt(nu_T)

            return F0_T * norm.cdf(d_plus) - self.params.K * norm.cdf(d_minus)
        except Exception as e:
            logger.error(f"Error calculating option price: {str(e)}")
            raise

    def calculate_greeks(self) -> Dict[str, float]:
        """Calculate option Greeks."""
        try:
            h = 1e-5  # Step size for finite differences
            
            # Delta
            price_up = BlackScholesStochastic(OptionParameters(
                S0=self.params.S0 + h, **{k:v for k,v in self.params.__dict__.items() if k != 'S0'}
            )).calculate_price()
            delta = (price_up - self.calculate_price()) / h
            
            # Gamma
            price_down = BlackScholesStochastic(OptionParameters(
                S0=self.params.S0 - h, **{k:v for k,v in self.params.__dict__.items() if k != 'S0'}
            )).calculate_price()
            gamma = (price_up - 2*self.calculate_price() + price_down) / (h*h)
            
            return {
                'delta': delta,
                'gamma': gamma
            }
        except Exception as e:
            logger.error(f"Error calculating Greeks: {str(e)}")
            raise

class OptionPricePlotter:
    """Class for plotting option prices."""
    
    def __init__(self, base_params: OptionParameters):
        self.base_params = base_params

    def plot_sensitivity(self, x_param: str, x_range: Tuple[float, float], 
                        num_points: int = 100) -> None:
        """Plot option price sensitivity with parallel processing."""
        try:
            x_values = np.linspace(x_range[0], x_range[1], num_points)
            
            with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                y_values = list(executor.map(
                    self._calculate_price_for_param,
                    [(x, x_param) for x in x_values]
                ))

            plt.figure(figsize=(10, 6))
            plt.title(f"Option Price Sensitivity to {x_param}")
            plt.plot(x_values, y_values, label="Option Price")
            plt.xlabel(x_param)
            plt.ylabel("Option Price")
            plt.grid(True)
            plt.legend()
            plt.show()

        except Exception as e:
            logger.error(f"Error in plotting: {str(e)}")
            raise

    def _calculate_price_for_param(self, args: Tuple[float, str]) -> float:
        """Helper method for parallel price calculation."""
        x, x_param = args
        params_dict = self.base_params.__dict__.copy()
        params_dict[x_param] = x
        return BlackScholesStochastic(OptionParameters(**params_dict)).calculate_price()

class OptionPricingGUI:
    """Enhanced GUI for option pricing."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Advanced Option Pricing Tool")
        self.base_dir = Path(__file__).parent
        self.params_file = self.base_dir / 'option_parameters.json'
        self.setup_ui()

    def setup_ui(self):
        """Set up the enhanced user interface."""
        # Create main container with padding
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Parameter input fields
        self.entries = {}
        default_values = self._load_default_parameters()
        
        for i, (key, value) in enumerate(default_values.items()):
            ttk.Label(main_frame, text=key).grid(row=i, column=0, padx=5, pady=2)
            entry = ttk.Entry(main_frame)
            entry.grid(row=i, column=1, padx=5, pady=2)
            entry.insert(0, str(value))
            self.entries[key] = entry

        # X-axis parameter selection
        ttk.Label(main_frame, text="Sensitivity Analysis").grid(
            row=len(default_values), column=0, columnspan=2, pady=10
        )
        
        self.x_param_var = tk.StringVar(self.root)
        self.x_param_var.set("S0")
        x_param_menu = ttk.OptionMenu(
            main_frame, self.x_param_var, "S0", *default_values.keys()
        )
        x_param_menu.grid(row=len(default_values) + 1, column=0, columnspan=2)

        # Range inputs
        range_frame = ttk.Frame(main_frame)
        range_frame.grid(row=len(default_values) + 2, column=0, columnspan=2, pady=5)
        
        ttk.Label(range_frame, text="Range:").grid(row=0, column=0)
        self.range_min = ttk.Entry(range_frame, width=10)
        self.range_min.grid(row=0, column=1, padx=5)
        self.range_min.insert(0, "50")
        
        ttk.Label(range_frame, text="to").grid(row=0, column=2)
        self.range_max = ttk.Entry(range_frame, width=10)
        self.range_max.grid(row=0, column=3, padx=5)
        self.range_max.insert(0, "150")

        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=len(default_values) + 3, column=0, columnspan=2, pady=10)
        
        ttk.Button(button_frame, text="Calculate", command=self.on_calculate).grid(
            row=0, column=0, padx=5
        )
        ttk.Button(button_frame, text="Save Parameters", command=self.save_parameters).grid(
            row=0, column=1, padx=5
        )
        ttk.Button(button_frame, text="Load Parameters", command=self.load_parameters).grid(
            row=0, column=2, padx=5
        )

    @staticmethod
    def _load_default_parameters() -> Dict[str, float]:
        """Load default parameters from configuration."""
        return {
            "S0": 100.0,
            "K": 100.0,
            "T": 1.0,
            "sigma1": 0.2,
            "sigma2": 0.1,
            "r0": 0.03,
            "a": 0.1,
            "theta": 0.05,
            "rho": 0.5
        }

    def _get_parameters(self) -> OptionParameters:
        """Get and validate parameters from UI."""
        try:
            params = {key: float(entry.get()) for key, entry in self.entries.items()}
            return OptionParameters(**params)
        except ValueError as e:
            messagebox.showerror("Invalid Input", str(e))
            raise

    def on_calculate(self):
        """Handle calculation button click."""
        try:
            params = self._get_parameters()
            x_param = self.x_param_var.get()
            x_range = (float(self.range_min.get()), float(self.range_max.get()))
            
            plotter = OptionPricePlotter(params)
            plotter.plot_sensitivity(x_param, x_range)
            
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def save_parameters(self):
        """Save current parameters to file."""
        try:
            params = {key: entry.get() for key, entry in self.entries.items()}
            with open(self.params_file, 'w') as f:
                json.dump(params, f)
            messagebox.showinfo("Success", "Parameters saved successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Could not save parameters: {str(e)}")

    def load_parameters(self):
        """Load parameters from file."""
        try:
            with open(self.params_file, 'r') as f:
                params = json.load(f)
            for key, value in params.items():
                if key in self.entries:
                    self.entries[key].delete(0, tk.END)
                    self.entries[key].insert(0, str(value))
            messagebox.showinfo("Success", "Parameters loaded successfully!")
        except FileNotFoundError:
            messagebox.showerror("Error", "No saved parameters found.")
        except Exception as e:
            messagebox.showerror("Error", f"Could not load parameters: {str(e)}")

    def run(self):
        """Start the GUI."""
        self.root.mainloop()

if __name__ == "__main__":
    try:
        app = OptionPricingGUI()
        app.run()
    except Exception as e:
        logger.critical(f"Application failed to start: {str(e)}")
        raise