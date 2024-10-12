import contextlib
import os
import sys

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from departure import Departure
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class Airport:
    def __init__(self, runways, terminals, departure_schedule, config):
        self.departure_schedule = departure_schedule
        self.runways = runways
        self.terminals = terminals
        self.config = config
        self.departures = []
        self.current_time = min(departure['PSRA'] for _, departure in departure_schedule.iterrows())
        self.eventLog = pd.DataFrame(columns=['READY FOR TAXI', 'IN TAXI', 'IN TAKEOFF QUEUE', 'DEPARTED', 'Runway Assignment'])

    def allocate_runway(self, departure, current_time, verbose):
        # Gate cost linearly dependent on wait time; but never negative
        current_delay = (current_time - departure.scheduled_departure_time).total_seconds()/60
        gate_cost = max(0, self.config['gate_wait_penalty'] * current_delay)
        runway_costs = [(runway, runway.get_wait_time_expectation()) for runway in self.runways]
        min_runway, min_wait_time = min(runway_costs, key=lambda x: x[1])

        # Runway cost depends on expected waiting time in queue
        # TODO: Add in travel time to gate cost
        queue_cost = (min_wait_time.total_seconds() / 60) * self.config['queue_wait_penalty']
        if gate_cost < queue_cost:
            if verbose:
                print(f"Flight {departure.flight_number} chose to stay at gate at {current_time.strftime('%H:%M')}.")
            departure.total_gate_wait += 1  # Increment gate wait by 1 minute
            # Flight can decide to stay at gate or proceed each time step
        else:
            if verbose:
                print(f"Flight {departure.flight_number} proceeds to taxi at {current_time.strftime('%H:%M')}.")
            departure.increment_status(Departure.IN_TAXI, current_time, verbose)
            selected_runway = min_runway
            selected_runway.add_departure(departure, current_time, verbose)

    def run_simulation(self, verbose = True):

        # Initialize departures
        for _, row in self.departure_schedule.iterrows():
            departure = Departure(row['Flight Number'], row['PSRA'], row['STD'], row['Aircraft Type'], self.config)
            self.departures.append(departure)

        # Define simulation start and end times
        start_time = min(departure['STD'] for _, departure in self.departure_schedule.iterrows()) - timedelta(minutes=15)
        end_time = max(departure['PSRA'] for _, departure in self.departure_schedule.iterrows()) + timedelta(minutes=30)

        # Simulation loop; iterate every minute
        current_time = start_time
        while current_time <= end_time:
            # Process each departure
            for departure in self.departures:
                if departure.status == Departure.AT_GATE:
                    if current_time >= departure.actual_departure_time:
                        departure.increment_status(Departure.READY_FOR_TAXI, current_time, verbose)
                if departure.status == Departure.READY_FOR_TAXI:
                    self.allocate_runway(departure, current_time, verbose)

            # Process each runway
            for runway in self.runways:
                runway.process_departures(current_time, verbose)

            current_time += timedelta(minutes=1)

        # Piece together eventlog
        for departure in self.departures:
            self.eventLog.loc[departure.flight_number, : ] = [*departure.eventLog, departure.runway_assignment.id]

        return self.eventLog



