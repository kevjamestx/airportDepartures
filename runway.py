import numpy as np
import pandas as pd
from departure import Departure
from scipy.stats import erlang
from datetime import datetime, timedelta


class Runway:
    def __init__(self, id, config):
        self.id = id  # Runway identifier
        self.queue = []  # Departures assigned to this runway
        self.travel_time_dist = erlang(config['travel_time_k'], scale=config['travel_time_scale'])
        self.service_time_dist = erlang(config['service_time_k'], scale=config['service_time_scale'])
        self.available_at = datetime.min  # Next available time
        self.occupancy_log = []  # List to store occupancy intervals (start_time, end_time)

    def __str__(self):
        return self.id

    def add_departure(self, departure, current_time, verbose):
        departure.travel_time = self.get_expected_taxi_time(departure)  # Taxi time
        departure.service_time = self.get_expected_service_time(departure)  # Time to be served by runway
        departure.taxi_start_time = current_time
        departure.taxi_end_time = current_time + departure.travel_time
        departure.service_start_time = None  # Time at which departure is first in line and ready to be served by runway
        departure.service_end_time = None  # Time at which departure is clear of runway
        departure.runway_assignment = self
        self.queue.append(departure)
        if verbose:
            print(f"Flight {departure.flight_number} assigned to runway {self.id} at {current_time.strftime('%H:%M')}.")

    def process_departures(self, current_time, verbose):
        if self.queue:
            self.queue.sort(key=lambda dep: dep.taxi_end_time)  # Sort the queue to reflect when aircraft arrived
            # Update those flights which have physically arrived at the runway
            for departure in self.queue:
                if departure.status == Departure.IN_TAXI and current_time >= departure.taxi_end_time:
                    departure.increment_status(Departure.IN_TAKEOFF_QUEUE, current_time, verbose)
            departure = self.queue[0]
            # Check if the runway is available and the departure has reached the taxiway
            if departure.status == Departure.IN_TAKEOFF_QUEUE and current_time >= self.available_at:
                departure.service_start_time = current_time
                if verbose:
                    print(
                    f"Flight {departure.flight_number} is being served by runway {self.id} at {current_time.strftime('%H:%M')}.")
                # Separation standards
                wake_separation = departure.get_min_separation()
                departed_time = current_time + self.get_expected_service_time(departure)
                self.available_at = departed_time + wake_separation
                departure.increment_status(Departure.DEPARTED, departed_time, verbose)
                self.queue.pop(0)
        else:
            pass  # No departures to process

    def get_wait_time_expectation(self):
        # The sum of the expected service times for all departures in the queue
        # TODO Here implement the consideration of other factors e.g. aircraft type, terminal location.
        return sum([timedelta(minutes=self.service_time_dist.rvs()) for departure in self.queue], timedelta())

    def get_expected_taxi_time(self, departure):
        return timedelta(minutes=self.travel_time_dist.rvs())
        # TODO Here implement expectation of taxi times based on location
        # Should look like:
        # return f(departure.terminalAssignment, self.id)

    def get_expected_service_time(self, departure):
        return timedelta(minutes=self.service_time_dist.rvs())
        # TODO Here implement expectations based on aircraft type
