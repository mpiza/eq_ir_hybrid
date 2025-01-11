import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import simpledialog

def black_scholes_stochastic_rates(S0, K, T, sigma1, sigma2, r0, a, theta, rho):
    # Helper functions
    def B(t, T, a):
        # Calculate B(t, T, a) with special case when a = 0
        if a == 0:
            return T - t
        return (1 - np.exp(-a * (T - t))) / a

    def c(t, a, theta, r0):
        # Calculate c(t, a, theta, r0) with special case when a = 0
        if a == 0:
            return r0 + theta * t
        return r0 * np.exp(-a * t) + theta / a * (1 - np.exp(-a * t))

    # Variance of log(S(T)) under the forward measure
    def compute_variance(T, sigma1, sigma2, rho, a):
        integral_B2 = (1 - np.exp(-2 * a * T)) / (2 * a)
        variance = sigma1**2 * T + sigma2**2 * integral_B2 + 2 * sigma1 * sigma2 * rho * B(0, T, a)
        return variance

    # Forward price
    def forward_price(S0, T, sigma1, sigma2, rho, a, r0, theta):
        integral_B = B(0, T, a)
        integral_c = c(T, a, theta, r0) * T - sigma2**2 * integral_B**2 / 2
        return S0 * np.exp(integral_c)

    # Parameters
    F0_T = forward_price(S0, T, sigma1, sigma2, rho, a, r0, theta)
    nu_T = compute_variance(T, sigma1, sigma2, rho, a)

    # Moneyness
    kappa = K / F0_T

    # d+ and d-
    d_plus = (-np.log(kappa) + 0.5 * nu_T) / np.sqrt(nu_T)
    d_minus = d_plus - np.sqrt(nu_T)

    # Option price
    C = F0_T * norm.cdf(d_plus) - K * norm.cdf(d_minus)

    return C

def plot_option_price(S0, K, T, sigma1, sigma2, r0, a, theta, rho, x_param, x_range):
    # Generate x values within the specified range
    x_values = np.linspace(x_range[0], x_range[1], 100)
    y_values = []

    # Calculate option price for each x value
    for x in x_values:
        params = {"S0": S0, "K": K, "T": T, "sigma1": sigma1, "sigma2": sigma2, "r0": r0, "a": a, "theta": theta, "rho": rho}
        params[x_param] = x
        y_values.append(black_scholes_stochastic_rates(**params))

    # Plot the option price against the selected x-axis parameter
    plt.figure()
    plt.title(f"Option Price vs {x_param}")
    plt.plot(x_values, y_values, label=f"Option Price")
    plt.xlabel(x_param)
    plt.ylabel("Option Price")
    plt.legend()
    plt.show()

def get_user_input():
    root = tk.Tk()
    root.title("Option Pricing Parameters")

    # Default values for the parameters
    default_values = {
        "S0": 100,
        "K": 100,
        "T": 1.0,
        "sigma1": 0.2,
        "sigma2": 0.1,
        "r0": 0.03,
        "a": 0.1,
        "theta": 0.05,
        "rho": 0.5
    }

    entries = {}
    # Create input fields for each parameter
    for i, (key, value) in enumerate(default_values.items()):
        tk.Label(root, text=key).grid(row=i, column=0)
        entry = tk.Entry(root)
        entry.grid(row=i, column=1)
        entry.insert(0, str(value))
        entries[key] = entry

    # Dropdown menu to select the x-axis parameter
    tk.Label(root, text="X-axis Parameter").grid(row=len(default_values), column=0)
    x_param_var = tk.StringVar(root)
    x_param_var.set("S0")
    x_param_menu = tk.OptionMenu(root, x_param_var, *default_values.keys())
    x_param_menu.grid(row=len(default_values), column=1)

    # Input fields to specify the range of the x-axis parameter
    tk.Label(root, text="X-axis Range (min, max)").grid(row=len(default_values) + 1, column=0)
    x_range_min_entry = tk.Entry(root)
    x_range_min_entry.grid(row=len(default_values) + 1, column=1)
    x_range_min_entry.insert(0, "50")
    x_range_max_entry = tk.Entry(root)
    x_range_max_entry.grid(row[len(default_values) + 1, column=2)
    x_range_max_entry.insert(0, "150")

    def on_submit():
        # Collect values from the input fields
        values = {key: float(entry.get()) for key, entry in entries.items()}
        x_param = x_param_var.get()
        x_range = (float(x_range_min_entry.get()), float(x_range_max_entry.get()))
        root.quit()
        root.destroy()
        return values, x_param, x_range

    # Submit button to finalize the input
    submit_button = tk.Button(root, text="Submit", command=on_submit)
    submit_button.grid(row=len(default_values) + 2, columnspan=3)

    root.mainloop()

    return on_submit()

if __name__ == "__main__":
    # Get user input and plot the option price
    params, x_param, x_range = get_user_input()
    plot_option_price(params["S0"], params["K"], params["T"], params["sigma1"], params["sigma2"], params["r0"], params["a"], params["theta"], params["rho"], x_param, x_range)
