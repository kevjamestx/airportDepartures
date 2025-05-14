from simulation.eventLog import EventLog
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from simulation.departure import Departure
import matplotlib

matplotlib.use('MacOSX')
import matplotlib.pyplot as plt


class Airport:
    def __init__(self, runways, terminals, departure_schedule, config):
        self.virtual_queue = []
        self.departure_schedule = departure_schedule
        self.runways = runways
        self.terminals = terminals
        self.config = config
        self.departures = []
        self.current_time = min(departure['PSRA'] for _, departure in departure_schedule.iterrows())
        self.eventLog = pd.DataFrame(
            columns=['Scheduled Departure Time', 'Flight number', 'Ready for Taxi', 'In Taxi', 'In Takeoff Queue',
                     'Departed', 'Runway Assignment'])

    def allocate_runway(self, departure, current_time, verbose):
        '''
        Allocate runway to departure according to the minimum expected wait + travel time
        '''
        # Gate cost linearly dependent on wait time; but never negative
        current_delay = (current_time - departure.scheduled_departure_time).total_seconds() / 60
        gate_cost = max(0, self.config['gate_wait_penalty'] * current_delay)
        runway_waits = [(runway, runway.get_expected_wait_time()) for runway in self.runways]
        min_runway, min_wait_time = min(runway_waits, key=lambda x: x[1])

        # Runway cost depends on expected waiting time in queue
        # TODO: Add in travel time to taxiway cost
        queue_cost = (min_wait_time.total_seconds() / 60) * self.config['queue_wait_penalty']
        if gate_cost < queue_cost:
            if verbose:
                print(f"Flight {departure.flight_number} chose to stay at gate at {current_time.strftime('%H:%M')}.")
            departure.total_gate_wait += 1  # Increment gate wait by 1 minute
            # Flight can decide to stay at gate or proceed each time step
            return False  # Was not allocated to a runway
        else:
            if verbose:
                print(f"Flight {departure.flight_number} proceeds to taxi at {current_time.strftime('%H:%M')}.")
            departure.increment_status(Departure.IN_TAXI, current_time, verbose)
            selected_runway = min_runway
            selected_runway.add_departure(departure, current_time, verbose)
            return True  # Allocated to a runway and now in the runway queue

    def process_departures(self, current_time, discipline, verbose):
        # Sort list according to priority
        def virtual_queue_priority(queue_member):
            '''
            Those departures with large delays will have a greater value
            '''
            departure, join_time = queue_member
            alpha = self.config['alpha']
            # Higher values to those running later without overly penalizing those who joined early
            if discipline == "priority":
                return (current_time - departure.scheduled_departure_time) + alpha * (current_time - join_time)

            elif discipline in {"FIFO", "FCFS"}:
                return current_time - join_time

            else:
                raise TypeError

        self.virtual_queue.sort(key=virtual_queue_priority, reverse=True)

        for departure, join_time in self.virtual_queue[:]:
            assert departure.status == Departure.READY_FOR_TAXI
            if self.allocate_runway(departure, current_time, verbose):
                self.virtual_queue.pop(0)
            else:
                break  # Don't serve departures further in the queue if the first can't be accommodated

    def run_simulation(self, time_bounds=None, discipline="priority", verbose=True):

        # Initialize departures
        for _, row in self.departure_schedule.iterrows():
            departure = Departure(row['Flight Number'], row['PSRA'], row['STD'], row['Aircraft Type'], self.config)
            self.departures.append(departure)

        # Define simulation start and end times
        if time_bounds is None:
            start_time = min(departure['STD'] for _, departure in self.departure_schedule.iterrows()) - timedelta(
                minutes=15)
            end_time = max(departure['PSRA'] for _, departure in self.departure_schedule.iterrows()) + timedelta(minutes=90)

        else:
            start_time, end_time = time_bounds
            # Allow time for last flights to depart
            end_time = max(departure['PSRA'] for _, departure in self.departure_schedule.iterrows()) + timedelta(minutes=90)

        # Simulation loop; iterate every minute
        current_time = start_time
        while current_time <= end_time:

            # Process each departure
            for departure in self.departures:
                # Increment to Ready for Taxi
                if departure.status == Departure.AT_GATE:
                    if current_time >= departure.actual_departure_time:
                        departure.increment_status(Departure.READY_FOR_TAXI, current_time, verbose)
                        # Add to the virtual queue
                        self.virtual_queue.append((departure, current_time))
                        # Make predictions for actual depart times
                        departure.predict_actual_departure_time(len(self.virtual_queue), [len(rwy.queue) for rwy in self.runways])

            # Allocate runways according to the virtual queue
            self.process_departures(current_time, discipline, verbose)

            # Process each runway
            for runway in self.runways:
                runway.process_departures(current_time, verbose)

            current_time += timedelta(minutes=1)

        # Piece together eventlog
        for departure in self.departures:
            self.eventLog.loc[departure.flight_number, 'Scheduled Departure Time'] = departure.scheduled_departure_time
            self.eventLog.loc[departure.flight_number, 'Flight number'] = departure.flight_number
            self.eventLog.loc[departure.flight_number, 'Ready for Taxi'] = departure.eventLog[0]
            self.eventLog.loc[departure.flight_number, 'In Taxi'] = departure.eventLog[1]
            self.eventLog.loc[departure.flight_number, 'In Takeoff Queue'] = departure.eventLog[2]
            self.eventLog.loc[departure.flight_number, 'Departed'] = departure.eventLog[3]
            self.eventLog.loc[departure.flight_number, 'Runway Assignment'] = departure.runway_assignment.id
            self.eventLog.loc[departure.flight_number, 'Predicted Takeoff Time'] = departure.predicted_takeoff_time


        self.eventLog['Pushback Delay'] = self.eventLog['Ready for Taxi'] - self.eventLog['Scheduled Departure Time']
        self.eventLog['Takeoff Delay'] = self.eventLog['Departed'] - self.eventLog['Scheduled Departure Time']
        self.eventLog['Pushback Control'] = self.eventLog['In Taxi'] - self.eventLog['Ready for Taxi']
        self.eventLog['Ready to Wheels Up'] = self.eventLog['Departed'] - self.eventLog['Ready for Taxi']


        result = EventLog(self.eventLog)
        return result
