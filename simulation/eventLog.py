import pickle
import os
import pandas as pd
import numpy as np


class EventLog:
    """
    The class is backed primarily with a pandas DataFrame representing the eventLog.
    Helper functions exist to return data in a standard format for ease of analysis.
    """

    def __init__(self, eventLog: pd.DataFrame):
        """
        :param eventLog: pandas DataFrame with rows as departures and columns as event times
        Expected columns:
            - 'Scheduled Departure Time' (Scheduled Pushback Time) [datetime]
            - 'Flight number' [str]
            - 'Ready for Taxi' (Actual Pushback Time) [datetime]
            - 'In Taxi' [datetime]
            - 'In Takeoff Queue' [datetime]
            - 'Departed' (Actual Takeoff Time) [datetime]
            - 'Runway Assignment' [str]
            - 'Pushback Delay' [timedelta]
            - 'Takeoff Delay' [timedelta]
            - 'Pushback Control' [timedelta]

        Note:
        - All time-related columns should be pandas datetime objects.
        - Scheduled Takeoff Time is not yet present and needs to be collected. For now, 
          it should be calculated as 'Departed' minus takeoff delay.
        """

        expected_columns = {
            'Scheduled Departure Time', 'Flight number', 'Ready for Taxi',
            'In Taxi', 'In Takeoff Queue', 'Departed', 'Runway Assignment',
            'Pushback Delay', 'Takeoff Delay', 'Pushback Control'
        }

        if not expected_columns.issubset(eventLog.columns):
            missing = expected_columns - set(eventLog.columns)
            raise TypeError(f"Missing expected columns: {missing}")

        # Ensure all time-related columns are pandas datetime objects
        datetime_columns = [
            'Scheduled Departure Time', 'Ready for Taxi', 'In Taxi',
            'In Takeoff Queue', 'Departed'
        ]

        for col in datetime_columns:
            if col in eventLog.columns:
                eventLog[col] = pd.to_datetime(eventLog[col], errors='coerce')

        self.eventLog = eventLog

        self.calculate_reorder_potential_metrics()

    # TODO add in methods and refactor airport/runway classes and constructor
    #  to directly add events through this class

    def get_takeoff_delays(self):
        """
        Return Actual Takeoff Times - Scheduled Takeoff Times in minutes for all flights.
        :return: a dictionary {Flight number (str): delay in minutes (int)}
        """
        self.eventLog["Takeoff Delay"] = pd.to_timedelta(self.eventLog["Takeoff Delay"], errors='coerce')
        return self.eventLog.dropna(subset=["Takeoff Delay"]).set_index("Flight number")["Takeoff Delay"] \
            .dt.total_seconds().div(60).astype(int).to_dict()

    def get_max_takeoff_delay(self):
        return max(self.get_takeoff_delays().values())

    def get_mean_takeoff_delay(self):
        return np.mean(list(self.get_takeoff_delays().values()))

    def get_mean_takeoff_delays_nondelayed_flights(self, delay_threshold=5):
        """
        Returns the mean takeoff delay (in minutes) for flights with pushback delay smaller than delay_threshold.
        :param delay_threshold: Minimum pushback delay (in minutes) to include a flight.
        :return: Mean takeoff delay in minutes (float)
        """
        self.eventLog["Pushback Delay"] = pd.to_timedelta(self.eventLog["Pushback Delay"], errors='coerce')
        self.eventLog["Takeoff Delay"] = pd.to_timedelta(self.eventLog["Takeoff Delay"], errors='coerce')

        filtered = self.eventLog[
            self.eventLog["Pushback Delay"].dt.total_seconds().div(60) < delay_threshold
        ]

        return filtered["Takeoff Delay"].dt.total_seconds().div(60).mean()

    def get_max_takeoff_delays_nondelayed_flights(self, delay_threshold=5):
        """
        Returns the mean takeoff delay (in minutes) for flights with pushback delay smaller than delay_threshold.
        :param delay_threshold: Minimum pushback delay (in minutes) to include a flight.
        :return: Mean takeoff delay in minutes (float)
        """
        self.eventLog["Pushback Delay"] = pd.to_timedelta(self.eventLog["Pushback Delay"], errors='coerce')
        self.eventLog["Takeoff Delay"] = pd.to_timedelta(self.eventLog["Takeoff Delay"], errors='coerce')

        filtered = self.eventLog[
            self.eventLog["Pushback Delay"].dt.total_seconds().div(60) < delay_threshold
        ]

        return filtered["Takeoff Delay"].dt.total_seconds().div(60).max()

    def get_mean_takeoff_delays_delayed_flights(self, delay_threshold=5):
        """
        Returns the mean takeoff delay (in minutes) for flights with pushback delay smaller than delay_threshold.
        :param delay_threshold: Minimum pushback delay (in minutes) to include a flight.
        :return: Mean takeoff delay in minutes (float)
        """
        self.eventLog["Pushback Delay"] = pd.to_timedelta(self.eventLog["Pushback Delay"], errors='coerce')
        self.eventLog["Takeoff Delay"] = pd.to_timedelta(self.eventLog["Takeoff Delay"], errors='coerce')

        filtered = self.eventLog[
            self.eventLog["Pushback Delay"].dt.total_seconds().div(60) > delay_threshold
        ]

        return filtered["Takeoff Delay"].dt.total_seconds().div(60).mean()

    def get_pushback_delays(self):
        """
        Return Ready for Taxi Times - Scheduled Pushback Times for all flights.
        :return: a dictionary {Flight number (str): delay in minutes (int)}
        """

        return (self.eventLog.dropna(subset=["Pushback Delay"])
                .set_index("Flight number")["Pushback Delay"]
                .apply(pd.to_timedelta)  # Ensure all values are Timedelta
                .dt.total_seconds().div(60).astype(int)
                .to_dict())

    def get_ready_to_wheels_up(self):
        """
        Return the column representing elapsed time between Ready for Pushback and Wheels Up
        """
        self.eventLog["Ready to Wheels Up"] = pd.to_timedelta(self.eventLog["Ready to Wheels Up"], errors='coerce')
        return self.eventLog.dropna(subset=["Ready to Wheels Up"]).set_index("Flight number")["Ready to Wheels Up"] \
            .dt.total_seconds().div(60).astype(int).to_dict()

    def get_scheduled_takeoff_times(self):
        """
        Return Scheduled Takeoff Times of all flights.
        :return:
        """
        pass  # TODO: Implement when scheduled takeoff times are available

    def calculate_reorder_potential_metrics(self):
        """
        Add a column to the eventLog dataframe titled "Virtual Queue Length".
        This column represents the number of other aircraft in the virtual queue
        at the ready for pushback time of each flight.

        A flight's virtual queue length is determined by counting how many flights:
        - Have a "Ready for Taxi" time earlier than the given flight's "Ready for Taxi" time.
        - Have an "In Taxi" time later than the given flight's "Ready for Taxi" time.
        """

        if not {'Ready for Taxi', 'In Taxi'}.issubset(self.eventLog.columns):
            raise KeyError("The dataframe must contain 'Ready for Taxi' and 'In Taxi' columns.")

        # Ensure the columns are in datetime format
        self.eventLog['Ready for Taxi'] = pd.to_datetime(self.eventLog['Ready for Taxi'])
        self.eventLog['In Taxi'] = pd.to_datetime(self.eventLog['In Taxi'])

        # Compute Virtual Queue Length for each flight
        virtual_queue_lengths = []

        for _, flight in self.eventLog.iterrows():
            ready_time = flight['Ready for Taxi']

            # Count flights that were "Ready for Taxi" before this flight and "In Taxi" after
            queue_length = ((self.eventLog['Ready for Taxi'] < ready_time) &
                            (self.eventLog['In Taxi'] > ready_time)).sum()

            virtual_queue_lengths.append(queue_length)

        # Assign the new column
        self.eventLog['Virtual Queue Length'] = virtual_queue_lengths



    def __repr__(self):
        return f"EventLog({len(self.eventLog)} events)"

    @staticmethod
    def concatenate_eventLogs(eventLogs):
        """
        Concatenates multiple EventLog objects into one.
        :param eventLogs: List of EventLog objects.
        :return: A new EventLog object containing all events from all input logs.
        """
        combined_df = pd.concat([elog.eventLog for elog in eventLogs], ignore_index=True)
        return EventLog(combined_df)

    @staticmethod
    def save_event_logs_to_pickle(event_log_dict, filename, directory="../simulation_results"):
        """
        Saves a dictionary of parameter values mapping to concatenated EventLog objects as a pickle file.

        :param event_log_dict: Dictionary {parameter_value: EventLog}
        :param filename: Name of the output pickle file (str).
        :param directory: Directory where the file will be saved (default: "simulation_results").
        """
        os.makedirs(directory, exist_ok=True)  # Ensure directory exists
        filepath = os.path.join(directory, filename)

        with open(filepath, "wb") as f:
            pickle.dump(event_log_dict, f)

        print(f"Saved EventLog to {filepath}")