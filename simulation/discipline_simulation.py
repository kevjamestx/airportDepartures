import numpy as np
import matplotlib
from datetime import datetime, timedelta
from scipy.stats import skewnorm, lognorm
from simulation.runway import Runway
from simulation.airport import Airport
from simulation.departure import Departure
from visualization.visualizer import *
import re

# matplotlib.use('Agg')


# Function to read departures data from a text file into a DataFrame
def read_txt_to_dataframe(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Remove any leading/trailing whitespace characters and filter out blank lines
    lines = [line.strip() for line in lines if line.strip()]

    # Initialize lists for storing each column of data
    times, flight_numbers, destinations, airlines, aircraft_types = [], [], [], [], []

    # Iterate over the lines and group data into columns
    for i in range(0, len(lines), 6):
        times.append(lines[i])
        flight_numbers.append(lines[i + 1])
        destinations.append(lines[i + 2])
        airlines.append(lines[i + 3])

        # Extract aircraft type without registration number
        aircraft_type_pattern = re.compile(r'(.+?) \(')
        aircraft_match = aircraft_type_pattern.match(lines[i + 4])
        aircraft_type = aircraft_match.group(1) if aircraft_match else lines[i + 4]
        aircraft_types.append(aircraft_type)

    # Create a DataFrame from the lists
    df = pd.DataFrame({
        'STD': times,
        'Flight Number': flight_numbers,
        'Destination': destinations,
        'Airline': airlines,
        'Aircraft Type': aircraft_types,
    })

    # Aircraft classifications for separation standards
    df['Aircraft Class'] = df['Aircraft Type'].map(Departure.aircraft_classification_dict)

    df['STD'] = pd.to_datetime(df['STD'], format='%I:%M %p')
    df['STD'] = df['STD'].apply(lambda x: datetime.combine(datetime.today(), x.time()))  # Combine with today's date

    return df


def discipline_simulation(runway_strs, terminals, discipline, CONFIG, time_bounds=None, verbose=False,
                          filename="../DFW data.txt", actual_pushback_delays = None):

    assert time_bounds is None or (type(time_bounds[0]) is datetime and type(time_bounds[1]) is datetime)

    # Load departures data
    departures_df = read_txt_to_dataframe(filename)
    if time_bounds is not None:
        lower_bound = time_bounds[0].time()
        upper_bound = time_bounds[1].time()
        departures_df = departures_df[(departures_df["STD"].dt.time >= lower_bound) &
                                      (departures_df["STD"].dt.time <= upper_bound)]


    # Apply delays to set actual pushback times
    if actual_pushback_delays is None:
        departures_df['PSRA'] = departures_df['STD'].apply(
            lambda x: x + timedelta(
                minutes=lognorm.rvs(CONFIG['PSRA_shape'], loc=CONFIG['PSRA_loc'], scale=CONFIG['PSRA_scale'])))
    # Actual pushback times provided
    else:
        actual_pushback_delays_timedelta = pd.to_timedelta(actual_pushback_delays, unit="m")
        departures_df['PSRA'] = departures_df['STD'] + actual_pushback_delays_timedelta


    # Instantiate runways
    runways = []
    for runway_str in runway_strs:
        runways.append(Runway(runway_str, CONFIG))

    # Instantiate airport

    airport = Airport(runways, terminals, departures_df, CONFIG)

    eventLog = airport.run_simulation(time_bounds, discipline, verbose)

    return eventLog

if __name__ == "__main__":
    CONFIG = {
        'PSRA_shape': 24.28,
        'PSRA_loc': -7.738,
        'PSRA_scale': 21.23,
        'travel_time_k': 8.533,
        'travel_time_scale': 0.83,
        'service_time_mu': 0.5,
        'service_time_sigma': 0.125,
        'gate_wait_penalty': 2,
        'queue_wait_penalty': 3,
        'alpha': 0
    }

    runways = ["17L", "17R", "18R"]

    terminals = ['A', 'B', 'C', 'D', 'E', 'Cargo', 'Corporate']

    discipline = "priority"

    alpha = 0

    CONFIG['alpha'] = alpha

    result = discipline_simulation(runways, terminals, discipline, CONFIG, filename = '../DFW data.txt')