import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from typing import Callable, Tuple
import json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from hwbs import verify_calibration, plot_calibration_results

class MarketDataFrame(ttk.LabelFrame):
    """Frame for market data input."""
    def __init__(self, parent):
        super().__init__(parent, text="Market Data")
        self.setup_ui()

    def setup_ui(self):
        # Input mode selection
        ttk.Label(self, text="Input Mode:").grid(row=0, column=0, padx=5, pady=2)
        self.input_mode = tk.StringVar(value="manual")
        mode_frame = ttk.Frame(self)
        mode_frame.grid(row=0, column=1, sticky='w')
        
        ttk.Radiobutton(mode_frame, text="Manual", variable=self.input_mode,
                       value="manual", command=self._toggle_input_mode).pack(side='left')
        ttk.Radiobutton(mode_frame, text="Generate", variable=self.input_mode,
                       value="generate", command=self._toggle_input_mode).pack(side='left')

        # Manual input frame
        self.manual_frame = ttk.Frame(self)
        self.manual_frame.grid(row=1, column=0, columnspan=2, sticky='ew', padx=5, pady=2)
        
        ttk.Label(self.manual_frame, text="Maturities (comma-separated):").grid(row=0, column=0)
        self.maturities_entry = ttk.Entry(self.manual_frame, width=40)
        self.maturities_entry.grid(row=0, column=1)
        self.maturities_entry.insert(0, "0.25, 0.5, 1.0, 2.0, 3.0, 5.0")

        ttk.Label(self.manual_frame, text="Market Vols (comma-separated):").grid(row=1, column=0)
        self.vols_entry = ttk.Entry(self.manual_frame, width=40)
        self.vols_entry.grid(row=1, column=1)
        self.vols_entry.insert(0, "0.22, 0.21, 0.20, 0.22, 0.23, 0.25")

        # Generate frame
        self.generate_frame = ttk.Frame(self)
        self.generate_frame.grid(row=2, column=0, columnspan=2, sticky='ew', padx=5, pady=2)
        
        ttk.Label(self.generate_frame, text="First Maturity:").grid(row=0, column=0)
        self.first_mat_entry = ttk.Entry(self.generate_frame, width=10)
        self.first_mat_entry.grid(row=0, column=1)
        self.first_mat_entry.insert(0, "0.25")
        
        ttk.Label(self.generate_frame, text="Last Maturity:").grid(row=0, column=2, padx=(10,0))
        self.last_mat_entry = ttk.Entry(self.generate_frame, width=10)
        self.last_mat_entry.grid(row=0, column=3)
        self.last_mat_entry.insert(0, "5.0")
        
        ttk.Label(self.generate_frame, text="Number of Points:").grid(row=1, column=0, columnspan=2)
        self.num_points_entry = ttk.Entry(self.generate_frame, width=10)
        self.num_points_entry.grid(row=1, column=2)
        self.num_points_entry.insert(0, "10")
        
        # Initially hide generate frame
        self.generate_frame.grid_remove()

    def _toggle_input_mode(self):
        """Toggle between manual and generate input modes."""
        if self.input_mode.get() == "manual":
            self.generate_frame.grid_remove()
            self.manual_frame.grid()
        else:
            self.manual_frame.grid_remove()
            self.generate_frame.grid()

    def get_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get market data as numpy arrays."""
        try:
            if self.input_mode.get() == "manual":
                # Parse manual input
                maturities = np.array([float(x.strip()) for x in self.maturities_entry.get().split(',')])
                vols = np.array([float(x.strip()) for x in self.vols_entry.get().split(',')])
            else:
                # Generate evenly spaced maturities
                t1 = float(self.first_mat_entry.get())
                t2 = float(self.last_mat_entry.get())
                n = int(self.num_points_entry.get())
                
                if t1 >= t2:
                    raise ValueError("Last maturity must be greater than first maturity")
                if n < 2:
                    raise ValueError("Number of points must be at least 2")
                
                maturities = np.linspace(t1, t2, n)
                vols = np.full_like(maturities, 0.2)  # Flat 20% volatility
            
            if len(maturities) != len(vols):
                raise ValueError("Maturities and volatilities must have same length")
            if not np.all(np.diff(maturities) > 0):
                raise ValueError("Maturities must be strictly increasing")
            if not np.all(vols > 0):
                raise ValueError("Volatilities must be positive")
                
            return maturities, vols
        except Exception as e:
            raise ValueError(f"Invalid market data: {str(e)}")

class ModelParamsFrame(ttk.LabelFrame):
    """Frame for model parameters input."""
    def __init__(self, parent):
        super().__init__(parent, text="Model Parameters")
        self.setup_ui()

    def setup_ui(self):
        # Alpha input
        ttk.Label(self, text="Mean Reversion (α):").grid(row=0, column=0, padx=5, pady=2)
        self.alpha_entry = ttk.Entry(self, width=10)
        self.alpha_entry.grid(row=0, column=1, padx=5, pady=2)
        self.alpha_entry.insert(0, "0.1")

        # Rho input
        ttk.Label(self, text="Correlation (ρ):").grid(row=1, column=0, padx=5, pady=2)
        self.rho_entry = ttk.Entry(self, width=10)
        self.rho_entry.grid(row=1, column=1, padx=5, pady=2)
        self.rho_entry.insert(0, "0.3")

        # Sigma_r function selection
        ttk.Label(self, text="Rate Vol Function:").grid(row=2, column=0, padx=5, pady=2)
        self.sigma_r_var = tk.StringVar(value="constant")
        self.sigma_r_combo = ttk.Combobox(self, 
                                         textvariable=self.sigma_r_var,
                                         values=["constant", "linear", "humped"],
                                         state="readonly",
                                         width=10)
        self.sigma_r_combo.grid(row=2, column=1, padx=5, pady=2)
        
        # Constant vol value
        ttk.Label(self, text="Constant Vol Value:").grid(row=3, column=0, padx=5, pady=2)
        self.const_vol_entry = ttk.Entry(self, width=10)
        self.const_vol_entry.grid(row=3, column=1, padx=5, pady=2)
        self.const_vol_entry.insert(0, "0.015")

    def get_params(self) -> Tuple[float, float, Callable]:
        """Get model parameters."""
        try:
            alpha = float(self.alpha_entry.get())
            rho = float(self.rho_entry.get())
            const_vol = float(self.const_vol_entry.get())
            
            if alpha <= 0:
                raise ValueError("Alpha must be positive")
            if not -1 <= rho <= 1:
                raise ValueError("Rho must be between -1 and 1")
            if const_vol <= 0:
                raise ValueError("Volatility must be positive")
            
            # Create sigma_r function based on selection
            if self.sigma_r_var.get() == "constant":
                def sigma_r(t): return const_vol
            elif self.sigma_r_var.get() == "linear":
                def sigma_r(t): return const_vol * (1 + 0.1 * t)
            else:  # humped
                def sigma_r(t): return const_vol * (1 + np.exp(-(t - 2)**2))
                
            return alpha, rho, sigma_r
        except Exception as e:
            raise ValueError(f"Invalid model parameters: {str(e)}")

class CalibrationApp:
    """Main application class."""
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Hull-White Equity Calibration")
        self.setup_ui()

    def setup_ui(self):
        # Create main frames
        self.market_frame = MarketDataFrame(self.root)
        self.market_frame.pack(padx=10, pady=5, fill="x")

        self.model_frame = ModelParamsFrame(self.root)
        self.model_frame.pack(padx=10, pady=5, fill="x")

        # Buttons
        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(padx=10, pady=5)
        
        ttk.Button(btn_frame, text="Calibrate", command=self.run_calibration).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Save", command=self.save_params).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Load", command=self.load_params).pack(side="left", padx=5)

    def run_calibration(self):
        """Run calibration with current parameters."""
        try:
            # Get parameters
            maturities, market_vols = self.market_frame.get_data()
            alpha, rho, sigma_r = self.model_frame.get_params()

            # Run calibration
            test_mats, interp_vols, stock_vols, recomp_vols = verify_calibration(
                maturities, market_vols, alpha, sigma_r, rho, num_test_points=200
            )

            # Plot results in new window
            plot_window = tk.Toplevel(self.root)
            plot_window.title("Calibration Results")
            
            plot_calibration_results(
                maturities, market_vols,
                test_mats, interp_vols,
                stock_vols, recomp_vols,
                alpha, sigma_r, rho
            )

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def save_params(self):
        """Save current parameters to file."""
        try:
            params = {
                'maturities': self.market_frame.maturities_entry.get(),
                'vols': self.market_frame.vols_entry.get(),
                'alpha': self.model_frame.alpha_entry.get(),
                'rho': self.model_frame.rho_entry.get(),
                'sigma_r_type': self.model_frame.sigma_r_var.get(),
                'const_vol': self.model_frame.const_vol_entry.get()
            }
            
            with open('calibration_params.json', 'w') as f:
                json.dump(params, f)
                
            messagebox.showinfo("Success", "Parameters saved successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Could not save parameters: {str(e)}")

    def load_params(self):
        """Load parameters from file."""
        try:
            with open('calibration_params.json', 'r') as f:
                params = json.load(f)
                
            self.market_frame.maturities_entry.delete(0, tk.END)
            self.market_frame.maturities_entry.insert(0, params['maturities'])
            
            self.market_frame.vols_entry.delete(0, tk.END)
            self.market_frame.vols_entry.insert(0, params['vols'])
            
            self.model_frame.alpha_entry.delete(0, tk.END)
            self.model_frame.alpha_entry.insert(0, params['alpha'])
            
            self.model_frame.rho_entry.delete(0, tk.END)
            self.model_frame.rho_entry.insert(0, params['rho'])
            
            self.model_frame.sigma_r_var.set(params['sigma_r_type'])
            
            self.model_frame.const_vol_entry.delete(0, tk.END)
            self.model_frame.const_vol_entry.insert(0, params['const_vol'])
            
            messagebox.showinfo("Success", "Parameters loaded successfully!")
        except FileNotFoundError:
            messagebox.showwarning("Warning", "No saved parameters found.")
        except Exception as e:
            messagebox.showerror("Error", f"Could not load parameters: {str(e)}")

    def run(self):
        """Start the application."""
        self.root.mainloop()

if __name__ == "__main__":
    app = CalibrationApp()
    app.run()
