import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from scipy.stats import erlang
from datetime import datetime, timedelta
import re
import heapq
from runway import Runway
from departure import Departure
from airport import Airport


# Configuration for simulation parameters - easy to modify
CONFIG = {
    'travel_time_k': 3,
    'travel_time_scale': 2,
    'service_time_k': 2,
    'service_time_scale': 1.5,
    'gate_wait_penalty': 3,  # Cost penalty for staying at the gate longer
    'queue_wait_penalty': 2  # Cost penalty for waiting in the takeoff queue
}

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

    # #Aircraft classifications for separation standards
    # df['Aircraft Class'] = df['Aircraft Type'].map(aircraft_classification_dict)

    df['STD'] = pd.to_datetime(df['STD'], format='%I:%M %p')
    df['STD'] = df['STD'].apply(lambda x: datetime.combine(datetime.today(), x.time()))  # Combine with today's date

    # PSRA: Pre Scheduled Random Arrivals
    df['PSRA'] = df['STD'].apply(lambda x: x + timedelta(minutes=np.random.normal(0, 5)))

    return df

if __name__ == "__main__":
    # Load departures data
    departures_df = read_txt_to_dataframe("DFW data.txt")

    # Instantiate runways
    runways = [
        Runway("17L", CONFIG),
        Runway("17C", CONFIG),
        Runway("17R", CONFIG),
        Runway("18L", CONFIG),
        Runway("18R", CONFIG)
    ]

    # Instantiate Terminals
    terminals = ['A', 'B', 'C', 'D', 'E', 'Cargo', 'Corporate']

    # Instantiate airport and run the simulation
    DFW = Airport(runways, terminals, departures_df, CONFIG)
    eventLog = DFW.run_simulation()

    # # Test with different number of servers
    # n = 5
    # DFW = []
    # eventLog = []
    # for i in range(2,n):
    #     DFW.append(Airport(runways[0:i+1], terminals, departures_df, CONFIG))
    #
    # for i, airportInstance in enumerate(DFW):
    #     eventLog.append(DFW[i].run_simulation(verbose=False))

    # Plots

    # Plot throughputs
    eventLog['DEPARTED'] = pd.to_datetime(eventLog['DEPARTED'])

    # Group the data by hour of departure and runway assignment
    departures_per_runway = eventLog.groupby([eventLog['DEPARTED'].dt.hour, 'Runway Assignment']).size().unstack()

    # Plot the number of departures per hour for each runway
    plt.figure(figsize=(10, 6))

    # Iterate over each runway and plot its departures
    for runway in departures_per_runway.columns:
        plt.plot(departures_per_runway.index, departures_per_runway[runway], marker='o', label=runway)

    # Add labels and title
    plt.xlabel('Hour of the Day')
    plt.ylabel('Number of Departures')
    plt.title('Number of Departures per Hour by Runway')
    plt.xticks(range(24))  # Ensure we show all hours of the day
    plt.legend(title="Runway")
    plt.grid(True)
    plt.savefig("Throughout by Runway.png")