import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("MacOSX")
import matplotlib.pyplot as plt

def infer_pushback_delay_normal_distribution(path_with_early, path_positive_only):
    """
    Using the spreadsheets produced by FAA ASPM, infer parameters of the normal distribution of PSRA
    :param path_with_early: path to the spreadhseet generated when "include early flight" is selected
    :param path_positive_only: path to the spreadsheet generated when "include early flight" is not selecetd
    :return: inferred mu, sigma of underlying delay distribution
    """
    # Read true XLS files; column headers are in the first row
    df_early = pd.read_excel(path_with_early, engine="xlrd")
    df_positive = pd.read_excel(path_positive_only, engine="xlrd")

    # Extract relevant columns
    mu_all = df_early['Average Gate Departure Delay']
    mu_pos = df_positive['Average Gate Departure Delay']

    # Remove any rows with missing data
    valid = mu_all.notna() & mu_pos.notna()
    mu_all = mu_all[valid]
    mu_pos = mu_pos[valid]

    # Estimate fraction of zero or early departures
    p = 1 - (mu_pos / mu_all)

    # Estimate standard deviation from truncated normal mean
    # E[X | X > 0] = μ + σ * φ(−μ/σ) / (1 − Φ(−μ/σ)) = mu_pos
    # Solve this numerically

    from scipy.stats import norm
    from scipy.optimize import minimize

    def objective(params):
        mu, sigma = params
        if sigma <= 0: return np.inf
        trunc_mean = mu + sigma * norm.pdf(-mu/sigma) / (1 - norm.cdf(-mu/sigma))
        return np.mean((trunc_mean - mu_pos)**2)

    # Initial guess from all values
    mu_init = mu_all.mean()
    sigma_init = mu_pos.std()

    result = minimize(objective, [mu_init, sigma_init], bounds=[(None, None), (1e-3, None)])
    mu_est, sigma_est = result.x

    return mu_est, sigma_est


def infer_pushback_delay_lognormal_distribution(path_positive_only):
    """
    Using the positive-only spreadsheet, infer parameters of the lognormal distribution of PSRA delays.
    :param path_positive_only: path to the spreadsheet with positive-only delays
    :return: inferred mu and sigma of the underlying normal for the lognormal distribution
    """
    # Read true XLS file for positive-only delays; column headers are in the first row
    df_positive = pd.read_excel(path_positive_only, engine="xlrd")

    # Extract the column for Average Gate Departure Delay and drop missing values
    delays = df_positive['Average Gate Departure Delay'].dropna()

    # Compute sample mean and variance
    m = delays.mean()
    v = delays.var()

    # Method of moments for lognormal distribution:
    # mean = exp(mu + sigma^2/2) and variance = (exp(sigma^2) - 1) * exp(2*mu + sigma^2)
    sigma2 = np.log(v / (m ** 2) + 1)
    sigma = np.sqrt(sigma2)
    mu = np.log(m) - sigma2 / 2

    return mu, sigma

def plot_inferred_normal_distribution(path_with_early, path_positive_only):
    # Get inferred parameters
    mu_est, sigma_est = infer_pushback_delay_normal_distribution(path_with_early, path_positive_only)

    # Read the positive-only delays from the XLS file
    df_positive = pd.read_excel(path_positive_only, engine="xlrd")
    mu_pos = df_positive['Average Gate Departure Delay'].dropna()

    from scipy.stats import norm
    # Compute the predicted truncated mean for the inferred parameters
    pred_trunc_mean = mu_est + sigma_est * norm.pdf(-mu_est / sigma_est) / (1 - norm.cdf(-mu_est / sigma_est))

    # Compute RMSE between the predicted truncated mean and observed positive-only delays
    rmse = np.sqrt(np.mean((pred_trunc_mean - mu_pos) ** 2))

    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    # Plot histogram of observed positive-only delays
    plt.hist(mu_pos, bins=20, density=True, alpha=0.6, label='Observed Positive-only Delays')

    # Compute the truncated normal PDF for x > 0:
    # f(x) = (1/σ) * φ((x-μ)/σ) / (1 - Φ(-μ/σ))
    x = np.linspace(0, mu_pos.max() + 20, 200)
    density = norm.pdf((x - mu_est) / sigma_est) / (sigma_est * (1 - norm.cdf(-mu_est / sigma_est)))
    plt.plot(x, density, 'r-', label='Inferred Truncated Normal PDF')

    # Plot a vertical line at the predicted truncated mean
    plt.axvline(pred_trunc_mean, color='k', linestyle='--', label=f'Predicted Truncated Mean = {pred_trunc_mean:.2f}')
    plt.xlabel('Average Gate Departure Delay')
    plt.ylabel('Density')
    plt.title('Observed Positive-only Delays vs Inferred Distribution')
    plt.legend()
    plt.savefig("Inferred Distribution.png")

    print(f"RMSE: {rmse:.2f}")
    return rmse


def plot_inferred_lognormal_distribution(path_positive_only):
    # Get inferred parameters (mu and sigma of the underlying normal)
    mu, sigma = infer_pushback_delay_lognormal_distribution(path_positive_only)

    # Read the positive-only delays
    df_positive = pd.read_excel(path_positive_only, engine="xlrd")
    delays = df_positive['Average Gate Departure Delay'].dropna()

    from scipy.stats import lognorm
    # In scipy, the lognormal distribution is parameterized by: shape = sigma, scale = exp(mu)

    # Create a range for x values
    x = np.linspace(0, delays.max() + 20, 200)
    pdf_values = lognorm.pdf(x, s=sigma, scale=np.exp(mu))

    # Plot histogram of observed delays
    plt.figure(figsize=(8, 6))
    counts, bins, _ = plt.hist(delays, bins=20, density=True, alpha=0.6, label='Observed Delays')

    # Overlay the fitted lognormal PDF
    plt.plot(x, pdf_values, 'r-', label='Fitted Lognormal PDF')
    plt.xlabel('Average Gate Departure Delay')
    plt.ylabel('Density')
    plt.title('Observed Positive-only Delays vs Fitted Lognormal Distribution')
    plt.legend()
    plt.savefig("Inferred_Distribution_Lognormal.png")

    # Compute RMSE between histogram densities and fitted PDF at the bin centers
    bin_centers = (bins[:-1] + bins[1:]) / 2
    hist_density, _ = np.histogram(delays, bins=bins, density=True)
    fitted_pdf = lognorm.pdf(bin_centers, s=sigma, scale=np.exp(mu))
    rmse = np.sqrt(np.mean((hist_density - fitted_pdf) ** 2))

    print(f"RMSE: {rmse:.2f}")
    return rmse

def get_delay_bucket_edges():
    return [0, 5, 25, 45, 105, 165, np.inf]

def fit_erlang_delay_distribution_from_counts(counts):
    from scipy.stats import gamma
    from scipy.optimize import minimize_scalar

    total = sum(counts)
    if total == 0:
        return None
    observed_probs = np.array(counts) / total
    edges = get_delay_bucket_edges()

    def loss(params):
        k, theta = params
        if k < 1 or theta <= 0:
            return np.inf
        k = int(round(k))
        cdf_vals = [gamma.cdf(edges[i+1], a=k, scale=theta) - gamma.cdf(edges[i], a=k, scale=theta) for i in range(len(counts))]
        return np.sum((observed_probs - np.array(cdf_vals))**2)

    best_loss = np.inf
    best_params = None
    for k in range(1, 11):
        res = minimize_scalar(lambda theta: loss((k, theta)), bounds=(0.5, 10), method='bounded')
        if res.fun < best_loss:
            best_loss = res.fun
            best_params = (k, res.x)

    return best_params

def group_rows_by_traffic(df, bin_size=10):
    df = df.copy()
    df['traffic_bin'] = (df['Departures For Metric Computation'] // bin_size) * bin_size
    return df.groupby('traffic_bin')

def fit_erlang_distribution_per_traffic_bin(df, bin_size=10):
    grouped = group_rows_by_traffic(df, bin_size)
    fit_results = {}

    bucket_cols = [
        '<20',
        '20-39',
        '40-59',
        '60-119',
        '120-179',
        '≥180',
    ]

    for traffic_bin, group in grouped:
        agg_counts = group[bucket_cols].sum().tolist()
        params = fit_erlang_delay_distribution_from_counts(agg_counts)
        if params:
            fit_results[traffic_bin] = params

    return fit_results

def plot_erlang_fit_accuracy(df, fit_results, bin_size=10, max_plots=3):
    import matplotlib.pyplot as plt
    from scipy.stats import gamma
    from scipy.stats import gaussian_kde
    import numpy as np

    bucket_cols = [
        '<20',
        '20-39',
        '40-59',
        '60-119',
        '120-179',
        '≥180',
    ]
    edges = get_delay_bucket_edges()
    grouped = group_rows_by_traffic(df, bin_size)

    plots_made = 0
    for traffic_bin, group in grouped:
        if traffic_bin not in fit_results:
            continue
        counts = group[bucket_cols].sum().values
        total = counts.sum()
        if total == 0:
            continue

        # Construct midpoints for histogram approximation
        midpoints = [10, 30, 50, 90, 150, 200]
        samples = []
        for midpoint, count in zip(midpoints, counts):
            samples.extend([midpoint] * int(count))
        if not samples:
            continue

        k, theta = fit_results[traffic_bin]
        x = np.linspace(0, max(samples) + 20, 300)
        y = gamma.pdf(x, a=k, scale=theta)

        # Use KDE for empirical approximation
        kde = gaussian_kde(samples)
        y_emp = kde(x)

        plt.figure(figsize=(8, 5))
        plt.plot(x, y_emp, label='Empirical KDE', linestyle='--')
        plt.plot(x, y, label=f'Erlang PDF (k={k}, θ={theta:.2f})')
        plt.title(f"Erlang Fit PDF for Departures {int(traffic_bin)}–{int(traffic_bin+bin_size-1)}")
        plt.xlabel("Taxi Delay (minutes)")
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()
        plt.show()

        plots_made += 1
        if plots_made >= max_plots:
            break

if __name__ == "__main__":
    mu, sigma = infer_pushback_delay_normal_distribution("Airport Analysis 0424 - 0624 Early.xls",
                                                         "Airport Analysis 0424 - 0624.xls")
    print("Inferred parameters:")
    print("mu =", mu, "sigma =", sigma)

    # Plot and report accuracy
    rmse = plot_inferred_normal_distribution("Airport Analysis 0424 - 0624 Early.xls",
                                             "Airport Analysis 0424 - 0624.xls")

    # Lognormal fitting
    mu, sigma = infer_pushback_delay_lognormal_distribution("Airport Analysis 0424 - 0624.xls")
    print("Inferred parameters:")
    print("mu =", mu, "sigma =", sigma)

    # Plot and report accuracy
    rmse = plot_inferred_lognormal_distribution("Airport Analysis 0424 - 0624.xls")

    # Erlang by traffic bins
    taxi_df = pd.read_excel("Taxi Time Analysis 06032024 - 06142024.xls", engine="xlrd")
    traffic_fits = fit_erlang_distribution_per_traffic_bin(taxi_df)
    print("Erlang parameters per traffic bin:")
    for traffic_bin, (k, theta) in sorted(traffic_fits.items()):
        print(f"Departures: {traffic_bin}-{traffic_bin+9}: k={k}, theta={theta:.2f}")

    plot_erlang_fit_accuracy(taxi_df, traffic_fits)