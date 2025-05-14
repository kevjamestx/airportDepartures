import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("MacOSX")
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import skewnorm, gamma, lognorm
from sklearn.metrics import mean_squared_error

# Heavy destinations to classify between Large and Heavy class departures
heavy_dests = ['AKL', 'MEL', 'SYD', 'BNE', 'NAN', 'HNL', 'OGG', 'NRT', 'HND', 'ICN', 'PVG', 'HKG', 'SCL', 'EZE', 'GRU',
               'GIG', 'DUB', 'LHR', 'AMS', 'CDG', 'MAD', 'BCN', 'FRA', 'VCE', 'FCO', 'IST', 'HEL', 'DOH', 'DXB']

def run_analysis(filename):
    # Read and parse the CSV file
    df = pd.read_csv(filename, skiprows=6)
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%y')
    df['Departure Hour'] = pd.to_numeric(df['Departure Hour'], errors='coerce')
    df = df.dropna(subset=['Departure Hour'])
    df['Departure Hour'] = df['Departure Hour'].astype(int)

    # Convert Flight Count to numeric and drop rows with non-numeric values
    df['Flight Count'] = pd.to_numeric(df['Flight Count'], errors='coerce')
    df = df.dropna(subset=['Flight Count'])

    # Create AircraftClass: Heavy if Arrival is in heavy_dests, else Large
    df['AircraftClass'] = np.where(df['Arrival'].isin(heavy_dests), 'Heavy', 'Large')

    # Calculate surface congestion: sum Flight Count per Date and Departure Hour
    congestion = df.groupby(['Date', 'Departure Hour'])['Flight Count'].sum().reset_index()
    congestion.rename(columns={'Flight Count': 'Congestion'}, inplace=True)
    df = pd.merge(df, congestion, on=['Date', 'Departure Hour'], how='left')

    # Clip Average Taxi Out Time at 11.4 minutes and compute extra taxi delay
    df['Average Taxi Out Time'] = df['Average Taxi Out Time'].clip(lower=11.4)
    df['taxi_delay'] = df['Average Taxi Out Time'] - 11.4

    # Adjust Gate Departure Delay: if Gate Departure Delay is 0, compute as (Airport Departure Delay - Taxi Out Delay)
    df['AdjustedGateDelay'] = np.where(df['Gate Departure Delay'] == 0,
                                       df['Airport Departure Delay'] - df['Taxi Out Delay'],
                                       df['Gate Departure Delay'])

    # Filter out outliers using the 1st and 99th percentiles for taxi_delay
    taxi_low, taxi_high = df['taxi_delay'].quantile([0.01, 0.99])
    df_filtered = df[(df['taxi_delay'] >= taxi_low) & (df['taxi_delay'] <= taxi_high)]

    # Filter out outliers for AdjustedGateDelay
    gate_low, gate_high = df_filtered['AdjustedGateDelay'].quantile([0.02, 0.95])
    df_filtered = df_filtered[(df_filtered['AdjustedGateDelay'] >= gate_low) & (df_filtered['AdjustedGateDelay'] <= gate_high)]

    # Fit GLM for taxi_delay using a Gamma family with log link
    # taxi_model = smf.glm(formula='taxi_delay ~ C(AircraftClass) + Congestion',
    #                      data=df_filtered,
    #                      family=sm.families.Gamma(sm.families.links.log()))

    # Univariate analysis instead
    taxi_model = smf.glm(formula='taxi_delay ~ 1',
                         data = df_filtered,
                         family = sm.families.Gamma(sm.families.links.log()))

    taxi_results = taxi_model.fit()
    print("Taxi Delay GLM Summary:")
    print(taxi_results.summary())

    # Fit lognormal distribution to AdjustedGateDelay (after translating data to be positive)
    gate_data = df_filtered['AdjustedGateDelay'].values
    lognorm_params = lognorm.fit(gate_data)  # returns (shape, loc, scale)
    # Report shift as location
    shape, loc, scale = lognorm_params
    print("\nLognormal Parameters for Adjusted Gate Delay (shape, loc, scale):")
    print((shape, loc, scale))

    # Compute RMSE between histogram and fitted PDF
    hist_vals, bin_edges = np.histogram(gate_data, bins=60, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    model_pdf_vals = lognorm.pdf(bin_centers, *lognorm_params)
    rmse = mean_squared_error(hist_vals, model_pdf_vals)
    print("Lognormal Fit RMSE: {:.4f}".format(rmse))

    return taxi_results, lognorm_params, df_filtered

def plot_gate_delay(gate_data, lognorm_params):
    # Plot histogram and KDE overlay for Adjusted Gate Departure Delay
    plt.figure(figsize=(10, 6))
    plt.hist(gate_data, bins=60, density=True, alpha=0.6, color='skyblue', edgecolor='black', label='Histogram')

    # Create a range of x values for the KDE plot
    x_min, x_max = gate_data.min() - 10, gate_data.max() + 10
    x = np.linspace(x_min, x_max, 1000)

    # Compute the PDF of the lognormal distribution with learned parameters
    shape, loc, scale = lognorm_params
    # Shift x values back for the lognormal PDF
    pdf_values = lognorm.pdf(x, shape, loc, scale)

    # Overlay the PDF
    plt.plot(x, pdf_values, 'r-', lw=2, label='Lognormal KDE')

    plt.xlabel('Gate Pushback Delay (minutes)')
    plt.ylabel('Density')
    plt.title('Histogram and Lognormal KDE for Gate Pushback Delay')
    plt.legend()
    plt.savefig("Gate Delay Distribution.png")
    # plt.show()


def plot_taxi_delay(taxi_delay):
    plt.figure(figsize=(10, 6))
    plt.hist(taxi_delay, bins=30, density=True, alpha=0.6, color='lightgreen', edgecolor='black', label='Histogram')

    # Filter out zero values because gamma requires strictly positive data
    positive_taxi_delay = taxi_delay[taxi_delay > 0]

    # Fit a Gamma distribution to the positive taxi_delay data (forcing location = 0)
    shape, loc, scale = gamma.fit(positive_taxi_delay, floc=0)

    # Create x values for the Gamma PDF based on the positive taxi_delay range
    x = np.linspace(positive_taxi_delay.min(), positive_taxi_delay.max(), 1000)
    pdf = gamma.pdf(x, shape, loc=loc, scale=scale)

    # Overlay the fitted Gamma PDF
    plt.plot(x, pdf, 'r-', lw=2, label='Gamma Fit')

    plt.xlabel('Extra Taxi Delay (minutes)')
    plt.ylabel('Density')
    plt.title('Histogram and Gamma Fit for Extra Taxi Delay')
    plt.legend()
    plt.savefig("Taxi Delay Distrbution.png")
    plt.show()

if __name__ == '__main__':
    taxi_results, lognorm_params, df_filtered = run_analysis('City Pair Analysis 0424-0624.csv')
    gate_data = df_filtered['AdjustedGateDelay'].values
    plot_gate_delay(gate_data, lognorm_params)

    # Plot the taxi delay distribution
    # taxi_delay = df_filtered['taxi_delay'].values
    # plot_taxi_delay(taxi_delay)
