import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import os
from scipy import stats
import seaborn as sns
from datetime import datetime, timedelta
from simulation.eventLog import EventLog


def load_event_logs_from_pickle(filename, directory="../simulation_results"):
    """
    Loads a pickle file and reconstructs the dictionary of parameter values mapping to EventLog objects.

    :param filename: Name of the input pickle file (str).
    :param directory: Directory where the file is located (default: "simulation_results").
    :return: Dictionary {parameter_value: EventLog}
    """
    filepath = os.path.join(directory, filename)

    with open(filepath, "rb") as f:
        event_log_dict = pickle.load(f)

    # Ensure all datetime columns are properly converted
    datetime_columns = [
        'Scheduled Departure Time', 'Ready for Taxi', 'In Taxi',
        'In Takeoff Queue', 'Departed'
    ]
    # Ensure all timedelta columns are properly converted
    timedelta_columns = ['Pushback Delay', 'Takeoff Delay', 'Pushback Control', 'Ready to Wheels Up']

    for key, val in event_log_dict.items():
        # Handle format where eventLog is a dictionary containing 'FIFO' and 'priority'
        if isinstance(val, dict):
            for perturbation, eventLog in val.items():
                for discipline in ['FIFO', 'priority']:
                    if discipline in eventLog and isinstance(eventLog[discipline], EventLog):
                        for col in datetime_columns:
                            if col in eventLog[discipline].eventLog.columns:
                                eventLog[discipline].eventLog[col] = pd.to_datetime(eventLog[discipline].eventLog[col],
                                                                                    errors='coerce')
                        for col in timedelta_columns:
                            if col in eventLog[discipline].eventLog.columns:
                                eventLog[discipline].eventLog[col] = pd.to_timedelta(eventLog[discipline].eventLog[col],
                                                                                     errors='coerce')
        # Handle format where eventLog is a direct EventLog object
        elif isinstance(val, EventLog):
            for col in datetime_columns:
                if col in val.eventLog.columns:
                    val.eventLog[col] = pd.to_datetime(val.eventLog[col], errors='coerce')
            for col in timedelta_columns:
                if col in val.eventLog.columns:
                    eventLog.eventLog[col] = pd.to_timedelta(val.eventLog[col], errors='coerce')

    # Handle nested dictionary as for exp 7: {propensity_label: {alpha: [EventLog, ...]}}
    for outer_key, inner_dict in event_log_dict.items():
        if isinstance(inner_dict, dict):
            for alpha_key, logs in inner_dict.items():
                if isinstance(logs, list) and all(isinstance(log, EventLog) for log in logs):
                    for log in logs:
                        for col in datetime_columns:
                            if col in log.eventLog.columns:
                                log.eventLog[col] = pd.to_datetime(log.eventLog[col], errors='coerce')
                        for col in timedelta_columns:
                            if col in log.eventLog.columns:
                                log.eventLog[col] = pd.to_timedelta(log.eventLog[col], errors='coerce')

    print(f"Loaded EventLog from {filepath}")
    return event_log_dict


def draw_simulation_gantt(eventLog, output_name="Gantt Chart.png"):
    """
    Visualizes the timeline of events for each flight in a single simulation.

    The x-axis represents time, and each row corresponds to a flight.
    Events ('Ready for Taxi', 'In Taxi', 'In Takeoff Queue', 'Departed') are marked as points.

    Args:
    event_log (pd.DataFrame): DataFrame containing simulation event logs.
    """

    eventLog = eventLog.eventLog
    event_names = ["Ready for Taxi", "In Taxi", "In Takeoff Queue", "Departed"]
    colors = ["blue", "green", "orange", "red"]  # Distinct colors for each event

    # Ensure time columns are datetime for proper plotting
    for event in event_names:
        eventLog[event] = pd.to_datetime(eventLog[event])

    # Sort by first event time for better visualization
    event_log = eventLog.sort_values(by="Scheduled Departure Time")

    plt.figure(figsize=(12, 8))
    for i, (_, row) in enumerate(event_log.iterrows()):
        flight_number = row["Flight number"]
        start_time = row["Scheduled Departure Time"]
        end_time = row["Departed"]

        # Plot a horizontal bar from 'READY FOR TAXI' to 'DEPARTED'
        plt.hlines(i, start_time, end_time, colors="gray", linewidth=5, alpha=0.7)

        # Add vertical markers at event times
        for event, color in zip(event_names, colors):
            plt.vlines(row[event], i - 0.4, i + 0.4, colors=color, linewidth=2, label=event if i == 0 else "")

    plt.xlabel("Time")
    plt.ylabel("Flights")
    plt.title("Simulation Activity Timeline")
    plt.yticks(range(len(event_log)), event_log["Flight number"])  # Use flight numbers as labels
    plt.legend()
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.savefig("../figures/" + output_name)


def show_pushback_delay_distribution(eventLogs, output_name="Pushback Delay Distribution.png"):
    """
    Visualizes the probability distribution of pushback delays using a histogram
    and kernel density estimate (KDE), aggregating data from multiple event logs.

    Args:
    *event_logs: Variable number of eventLog objects, each with a
                 get_pushback_delays() method returning a dictionary
                 mapping flight numbers to pushback delays in minutes.
    """
    delay_values = []

    # Collect pushback delays from all event logs
    for eventLog in eventLogs:
        pushback_delays = eventLog.get_pushback_delays()
        delay_values.extend(pushback_delays.values())

    plt.figure(figsize=(10, 6))

    # Plot histogram with KDE overlay
    sns.histplot(delay_values, bins=50, kde=True, color="blue", edgecolor="black", alpha=0.6)

    plt.xlabel("Pushback Delay (minutes)")
    plt.ylabel("Frequency")
    plt.title("Probability Distribution of Pushback Delays (Aggregated)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig("../figures/" + output_name)


def show_virtual_queue_length_fluctuations(eventLog, output_name="Virtual Queue Length.png"):
    df = eventLog.eventLog.copy()
    df = df.sort_values("Ready for Taxi")
    df["Ready for Taxi"] = pd.to_datetime(df["Ready for Taxi"])

    # Extract time series of virtual queue lengths
    time_series = df["Ready for Taxi"]
    queue_lengths = df["Virtual Queue Length"]

    plt.figure(figsize=(12, 6))
    plt.plot(time_series, queue_lengths, drawstyle="steps-post")
    plt.xlabel("Time")
    plt.ylabel("Virtual Queue Length")
    plt.title("Virtual Queue Length Over Time")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("../figures/" + output_name)


# Delay vs alpha functions

def plot_max_takeoff_delay_for_alpha(mc_output, output_name="Max Takeoff Delay Statistics vs Alpha.png"):
    """
    Generates a plot showing the average and 95% confidence interval of the max takeoff delay
    across all Monte Carlo runs for different values of alpha.

    Parameters:
        eventLogs (dict): Maps alpha values to lists of EventLogs.
    """
    alphas = []
    mean_delays = []
    ci_delays = []

    for alpha, eventLogs in mc_output.items():
        max_delays = [eventLog.get_max_takeoff_delay() for eventLog in eventLogs]
        mean = np.mean(max_delays)
        std_err = stats.sem(max_delays)
        ci = std_err * stats.t.ppf((1 + 0.95) / 2., len(max_delays) - 1)

        alphas.append(alpha)
        mean_delays.append(mean)
        ci_delays.append(ci)

    plt.figure()
    plt.xlabel('Alpha')
    plt.ylabel('Avg Max Takeoff Delay (minutes)')
    plt.errorbar(alphas, mean_delays, yerr=ci_delays, fmt='o', color='tab:blue', capsize=5, label='Mean ± 95% CI')
    plt.title('Max Takeoff Delay Statistics vs Alpha')
    plt.legend()
    plt.savefig("../figures/" + output_name)


def plot_mean_takeoff_delay_for_alpha(mc_output, output_name="Mean Takeoff Delay Statistics vs Alpha.png"):
    """
    Generates a plot showing the average and 95% confidence interval of the mean takeoff delay
    across all Monte Carlo runs for different values of alpha.

    Parameters:
        eventLogs (dict): Maps alpha values to lists of EventLogs.
    """
    alphas = []
    mean_delays = []
    ci_delays = []

    for alpha, eventLogs in mc_output.items():
        means = [eventLog.get_mean_takeoff_delay() for eventLog in eventLogs]
        mean = np.mean(means)
        std_err = stats.sem(means)
        ci = std_err * stats.t.ppf((1 + 0.95) / 2., len(means) - 1)

        alphas.append(alpha)
        mean_delays.append(mean)
        ci_delays.append(ci)

    plt.figure()
    plt.xlabel('Alpha')
    plt.ylabel('Mean Takeoff Delay (minutes)')
    plt.errorbar(alphas, mean_delays, yerr=ci_delays, fmt='o', color='tab:blue', capsize=5, label='Mean ± 95% CI')
    plt.xscale('log')
    plt.title('Mean Takeoff Delay vs Alpha')
    plt.legend()
    plt.savefig("../figures/" + output_name)


def plot_mean_takeoff_delay_nondelayed_flights_for_alpha(mc_output,
                                                         output_name="Mean Takeoff Delay Statistics vs Alpha.png"):
    """
    Generates a plot showing the average and 95% confidence interval of the mean takeoff delay
    across all Monte Carlo runs for different values of alpha.

    Parameters:
        eventLogs (dict): Maps alpha values to lists of EventLogs.
    """
    alphas = []
    mean_delays = []
    ci_delays = []

    for alpha, eventLogs in mc_output.items():
        means = [eventLog.get_mean_takeoff_delays_nondelayed_flights(delay_threshold=5) for eventLog in eventLogs]
        mean = np.mean(means)
        std_err = stats.sem(means)
        ci = std_err * stats.t.ppf((1 + 0.95) / 2., len(means) - 1)

        alphas.append(alpha)
        mean_delays.append(mean)
        ci_delays.append(ci)

    plt.figure()
    plt.xlabel('Alpha')
    plt.ylabel('Mean Takeoff Delay of Nondelayed Flights (minutes)')
    plt.errorbar(alphas, mean_delays, yerr=ci_delays, fmt='o', color='tab:blue', capsize=5, label='Mean ± 95% CI')
    plt.xscale('log')
    plt.title('Mean Takeoff Delay vs Alpha')
    plt.legend()
    plt.savefig("../figures/" + output_name)


def plot_max_takeoff_delay_nondelayed_flights_for_alpha(mc_output,
                                                        output_name="Mean Takeoff Delay Statistics vs Alpha.png"):
    """
    Generates a plot showing the average and 95% confidence interval of the mean takeoff delay
    across all Monte Carlo runs for different values of alpha.

    Parameters:
        eventLogs (dict): Maps alpha values to lists of EventLogs.
    """
    alphas = []
    mean_delays = []
    ci_delays = []

    for alpha, eventLogs in mc_output.items():
        maxes = [eventLog.get_max_takeoff_delays_nondelayed_flights(delay_threshold=5) for eventLog in eventLogs]
        mean = np.mean(maxes)
        std_err = stats.sem(maxes)
        ci = std_err * stats.t.ppf((1 + 0.95) / 2., len(maxes) - 1)

        alphas.append(alpha)
        mean_delays.append(mean)
        ci_delays.append(ci)

    plt.figure()
    plt.xlabel('Alpha')
    plt.ylabel('Max Takeoff Delay of Nondelayed Flights (minutes)')
    plt.errorbar(alphas, mean_delays, yerr=ci_delays, fmt='o', color='tab:blue', capsize=5, label='Mean ± 95% CI')
    plt.xscale('log')
    plt.title('Max Takeoff Delay vs Alpha')
    plt.legend()
    plt.savefig("../figures/" + output_name)


def plot_mean_takeoff_delay_delayed_flights_for_alpha(mc_output,
                                                      output_name="Mean Takeoff Delay Statistics vs Alpha.png"):
    """
    Generates a plot showing the average and 95% confidence interval of the mean takeoff delay
    across all Monte Carlo runs for different values of alpha.

    Parameters:
        eventLogs (dict): Maps alpha values to lists of EventLogs.
    """
    alphas = []
    mean_delays = []
    ci_delays = []

    for alpha, eventLogs in mc_output.items():
        means = [eventLog.get_mean_takeoff_delays_delayed_flights(delay_threshold=5) for eventLog in eventLogs]
        mean = np.mean(means)
        std_err = stats.sem(means)
        ci = std_err * stats.t.ppf((1 + 0.95) / 2., len(means) - 1)

        alphas.append(alpha)
        mean_delays.append(mean)
        ci_delays.append(ci)

    plt.figure()
    plt.xlabel('Alpha')
    plt.ylabel('Mean Takeoff Delay of Pushback Delayed Flights (minutes)')
    plt.errorbar(alphas, mean_delays, yerr=ci_delays, fmt='o', color='tab:blue', capsize=5, label='Mean ± 95% CI')
    plt.xscale('log')
    plt.title('Mean Takeoff Delay vs Alpha')
    plt.legend()
    plt.savefig("../figures/" + output_name)


# TO vs Pushback Delay functions
def plot_max_takeoff_delay_vs_pushback_delay(fifo_event_logs, priority_event_logs,
                                             output_name="TO delay vs pushback delay.png"):
    """
    Generates a plot where the x-axis is discretized pushback delay (in minutes),
    and the y-axis is the maximum takeoff delay among flights with that pushback delay.
    One line corresponds to FIFO event logs, and the other to priority queue event logs.

    Args:
    fifo_event_logs (list of pd.DataFrame): List of event logs under FIFO discipline.
    priority_event_logs (list of pd.DataFrame): List of event logs under priority queue discipline.
    """

    def compute_max_takeoff_delay(event_logs):
        """Computes max takeoff delay for each discretized minute of pushback delay."""
        combined_data = pd.concat(event_logs, ignore_index=True)

        # Convert to timedelta
        combined_data["Pushback Delay"] = pd.to_timedelta(combined_data["Pushback Delay"])
        combined_data["Takeoff Delay"] = pd.to_timedelta(combined_data["Takeoff Delay"])

        # Discretize pushback delay into minutes
        combined_data["Pushback Delay (min)"] = (combined_data["Pushback Delay"].dt.total_seconds() // 60).astype(int)

        # Get max takeoff delay for each pushback delay bin
        max_takeoff_delays = (
                combined_data.groupby("Pushback Delay (min)")["Takeoff Delay"].max().dt.total_seconds() / 60
        )
        return max_takeoff_delays.sort_index()

    fifo_delays = compute_max_takeoff_delay([eLog.eventLog for eLog in fifo_event_logs])
    priority_delays = compute_max_takeoff_delay([eLog.eventLog for eLog in priority_event_logs])

    plt.figure(figsize=(10, 6))
    plt.plot(fifo_delays.index, fifo_delays.values, label="FIFO", marker="o")
    plt.plot(priority_delays.index, priority_delays.values, label="Priority Queue", marker="s")

    plt.xlim([0, 70])
    plt.ylim([45, 110])
    plt.xlabel("Pushback Delay (minutes)")
    plt.ylabel("Max Takeoff Delay (minutes)")
    plt.title("Max Takeoff Delay vs Pushback Delay")
    plt.legend()
    plt.grid(True)
    plt.savefig("../figures/" + output_name)


def plot_mean_takeoff_delay_vs_pushback_delay(fifo_event_logs, priority_event_logs,
                                              output_name="TO delay vs pushback delay.png",
                                              show_error_bars=False):
    def compute_mean_takeoff_delay(event_logs):
        combined_data = pd.concat(event_logs, ignore_index=True)
        combined_data["Pushback Delay"] = pd.to_timedelta(combined_data["Pushback Delay"])
        combined_data["Takeoff Delay"] = pd.to_timedelta(combined_data["Takeoff Delay"])
        combined_data["Pushback Delay (min)"] = (combined_data["Pushback Delay"].dt.total_seconds() // 60).astype(int)

        mean_takeoff_delays = (
                combined_data.groupby("Pushback Delay (min)")["Takeoff Delay"].mean().dt.total_seconds() / 60
        )
        return mean_takeoff_delays.sort_index(), combined_data

    fifo_delays, fifo_data = compute_mean_takeoff_delay([eLog.eventLog for eLog in fifo_event_logs])
    priority_delays, priority_data = compute_mean_takeoff_delay([eLog.eventLog for eLog in priority_event_logs])

    if show_error_bars:
        fifo_std_err = fifo_data.groupby("Pushback Delay (min)")["Takeoff Delay"].sem().dt.total_seconds() / 60
        priority_std_err = priority_data.groupby("Pushback Delay (min)")["Takeoff Delay"].sem().dt.total_seconds() / 60

    plt.figure(figsize=(10, 6))
    if show_error_bars:
        plt.errorbar(fifo_delays.index, fifo_delays.values, yerr=fifo_std_err.loc[fifo_delays.index],
                     label="FIFO", fmt='o', capsize=5)
        plt.errorbar(priority_delays.index, priority_delays.values, yerr=priority_std_err.loc[priority_delays.index],
                     label="Priority Queue", fmt='s', capsize=5)
    else:
        plt.plot(fifo_delays.index, fifo_delays.values, label="FIFO", marker="o")
        plt.plot(priority_delays.index, priority_delays.values, label="Priority Queue", marker="s")

    plt.xlim([0, 60])
    plt.ylim([0, 110])
    plt.xlabel("Pushback Delay (minutes)")
    plt.ylabel("Mean Takeoff Delay (minutes)")
    plt.title("Mean Takeoff Delay vs Pushback Delay")
    plt.legend()
    plt.grid(True)
    plt.savefig("../figures/" + output_name)


# Below functions are for single perturbation mode experiments

def plot_max_takeoff_delay_vs_pushback_delay_single_flight_perturbations(perturbation_logs, flight_number=None,
                                                                         output_name="TO delay vs pushback delay for single flight.png"):
    """
    Plots the takeoff delay vs. pushback delay for a single flight across multiple perturbations.

    Args:
        perturbation_logs (dict): Dictionary structured as {flight_number: {perturbation: {'FIFO' or 'priority': EventLog}}}.
        flight_number (str, optional): Specific flight number to analyze. If None, chooses an arbitrary flight.
        output_name (str): Filename to save the plot.
    """
    if flight_number is None:
        # Select an arbitrary flight from the keys
        flight_number = list(perturbation_logs.keys())[0]

    fifo_pushback_delays = []
    fifo_takeoff_delays = []
    priority_pushback_delays = []
    priority_takeoff_delays = []

    for perturbation, event_logs in perturbation_logs[flight_number].items():
        if 'FIFO' in event_logs:
            fifo_pushback_delays.append(event_logs['FIFO'].get_pushback_delays()[flight_number])
            fifo_takeoff_delays.append(event_logs['FIFO'].get_takeoff_delays()[flight_number])

        if 'priority' in event_logs:
            priority_pushback_delays.append(event_logs['priority'].get_pushback_delays()[flight_number])
            priority_takeoff_delays.append(event_logs['priority'].get_takeoff_delays()[flight_number])

    plt.figure(figsize=(10, 6))
    plt.plot(fifo_pushback_delays, fifo_takeoff_delays, marker='o', linestyle='-', label="FIFO")
    plt.plot(priority_pushback_delays, priority_takeoff_delays, marker='s', linestyle='-', label="Priority Queue")

    plt.xlabel("Pushback Delay (minutes)")
    plt.ylabel("Takeoff Delay (minutes)")
    plt.title(f"Takeoff Delay vs Pushback Delay for Flight {flight_number}")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig("../figures/" + output_name)


def plot_max_takeoff_delay_vs_pushback_delay_single_flight_perturbations_smoothed(perturbation_logs,
                                                                                  include_mixed_priority=False,
                                                                                  output_name="TO delay vs pushback delay averaged.png"):
    """
    Plots the takeoff delay vs. pushback delay averaged across all flights across multiple perturbations.

    Args:
        perturbation_logs (dict): Dictionary structured as {flight_number: {perturbation: {'FIFO' or 'priority': EventLog}}}.
        output_name (str): Filename to save the plot.
    """
    fifo_data = []
    priority_data = []
    mixed_priority_data = []

    for flight_number, perturbations in perturbation_logs.items():
        for perturbation, event_logs in perturbations.items():
            if 'FIFO' in event_logs:
                pushback_delay = event_logs['FIFO'].get_pushback_delays()[flight_number]
                takeoff_delay = event_logs['FIFO'].get_takeoff_delays()[flight_number]
                fifo_data.append((pushback_delay, takeoff_delay))

            if 'priority' in event_logs:
                pushback_delay = event_logs['priority'].get_pushback_delays()[flight_number]
                takeoff_delay = event_logs['priority'].get_takeoff_delays()[flight_number]
                priority_data.append((pushback_delay, takeoff_delay))

            if 'Mixed priority' in event_logs:
                pushback_delay = event_logs['Mixed priority'].get_pushback_delays()[flight_number]
                takeoff_delay = event_logs['Mixed priority'].get_takeoff_delays()[flight_number]
                mixed_priority_data.append((pushback_delay, takeoff_delay))

    # Convert to DataFrame for aggregation
    fifo_df = pd.DataFrame(fifo_data, columns=["Pushback Delay", "Takeoff Delay"])
    priority_df = pd.DataFrame(priority_data, columns=["Pushback Delay", "Takeoff Delay"])
    mixed_priority_df = pd.DataFrame(mixed_priority_data, columns=["Pushback Delay", "Takeoff Delay"])

    # Group by pushback delay and compute mean takeoff delay and standard error
    fifo_group = fifo_df.groupby("Pushback Delay")["Takeoff Delay"]
    priority_group = priority_df.groupby("Pushback Delay")["Takeoff Delay"]
    mixed_priority_group = mixed_priority_df.groupby("Pushback Delay")["Takeoff Delay"]

    fifo_avg = fifo_group.mean()
    priority_avg = priority_group.mean()
    mixed_priority_avg = mixed_priority_group.mean()

    fifo_std_err = fifo_group.sem()
    priority_std_err = priority_group.sem()
    mixed_priority_std_err = mixed_priority_group.sem()

    plt.figure(figsize=(10, 6))
    confidence_level = 1.96  # for 95% confidence interval
    fifo_ci = confidence_level * fifo_std_err
    priority_ci = confidence_level * priority_std_err
    mixed_priority_ci = confidence_level * mixed_priority_std_err
    plt.errorbar(fifo_avg.index, fifo_avg.values, yerr=fifo_ci.values, fmt='o-', capsize=5, label="FIFO", color="blue")
    # plt.errorbar(mixed_priority_avg.index, mixed_priority_avg.values, yerr=mixed_priority_ci.values, fmt='s-', capsize=5,
    # label=r"Priority Queue ($\alpha = 1$)", color="yellow")
    plt.errorbar(priority_avg.index, priority_avg.values, yerr=priority_ci.values, fmt='s-', capsize=5,
                 label=r"Priority Queue ($\alpha = 0$)", color="red")

    plt.xlabel("Pushback Delay (minutes)")
    plt.ylabel("Average Takeoff Delay (minutes)")
    plt.title("Average Takeoff Delay vs Pushback Delay Across All Flights")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig("../figures/" + output_name)


def plot_effect_of_reordering_on_nonperturbed_flights(perturbation_logs,
                                                      metric="max",
                                                      output_name="TO Delays vs pushback delay on non-perturbed flights.png"):
    fifo_data = []
    priority_data = []

    for perturbed_flight, perturbations in perturbation_logs.items():
        for perturbation, event_logs in perturbations.items():
            if 'priority' in event_logs:
                # Compute affected flights using priority event logs
                priority_event_log_df = event_logs['priority'].eventLog
                perturbed_ready_for_taxi = priority_event_log_df.loc[
                    priority_event_log_df["Flight number"] == perturbed_flight, "Ready for Taxi"
                ].values[0]
                perturbed_in_taxi = priority_event_log_df.loc[
                    priority_event_log_df["Flight number"] == perturbed_flight, "In Taxi"
                ].values[0]

                # Consider only those flights which were specifically affected by the resequencing
                affected_flights = priority_event_log_df[
                    (priority_event_log_df["Flight number"] != perturbed_flight) &
                    (priority_event_log_df["Ready for Taxi"] < perturbed_ready_for_taxi) &
                    (priority_event_log_df["In Taxi"] > perturbed_in_taxi)
                    ]["Flight number"].tolist()

                # Capture takeoff delays for affected flights under both FIFO and priority disciplines
                for discipline in ['FIFO', 'priority']:
                    if discipline in event_logs:
                        event_log_df = event_logs[discipline].eventLog
                        pushback_delay = event_logs[discipline].get_pushback_delays()[perturbed_flight]
                        takeoff_delays = event_log_df[
                                             event_log_df["Flight number"].isin(affected_flights)
                                         ]["Takeoff Delay"].dt.total_seconds() / 60  # Convert to minutes

                        if not takeoff_delays.empty:
                            if metric == "max":
                                if discipline == 'FIFO':
                                    fifo_data.append((pushback_delay, takeoff_delays.max()))
                                else:
                                    priority_data.append((pushback_delay, takeoff_delays.max()))
                            else:
                                if discipline == 'FIFO':
                                    fifo_data.append((pushback_delay, takeoff_delays.mean()))
                                else:
                                    priority_data.append((pushback_delay, takeoff_delays.mean()))

    # Convert to DataFrame for aggregation
    fifo_df = pd.DataFrame(fifo_data, columns=["Pushback Delay", "Takeoff Delay"])
    priority_df = pd.DataFrame(priority_data, columns=["Pushback Delay", "Takeoff Delay"])

    # Filter to include only pushback delays between 0 and 50
    fifo_df = fifo_df[(fifo_df["Pushback Delay"] >= 0) & (fifo_df["Pushback Delay"] <= 90)]
    priority_df = priority_df[(priority_df["Pushback Delay"] >= 0) & (priority_df["Pushback Delay"] <= 90)]

    # Group by pushback delay and compute aggregate metric
    fifo_group = fifo_df.groupby("Pushback Delay")["Takeoff Delay"]
    priority_group = priority_df.groupby("Pushback Delay")["Takeoff Delay"]

    plt.rcParams.update({'axes.titlesize': 18, 'axes.labelsize': 18, 'xtick.labelsize': 18, 'ytick.labelsize': 18,
                         'legend.fontsize': 18})
    plt.figure(figsize=(10, 6))
    if metric == "mean":
        fifo_mean = fifo_group.mean()
        priority_mean = priority_group.mean()
        plt.plot(fifo_mean.index, fifo_mean.values, marker='o', linestyle='-', label="FIFO")
        plt.plot(priority_mean.index, priority_mean.values, marker='s', linestyle='-', label="Priority Queue")

        min_y = min(fifo_mean.min(), priority_mean.min()) - 5
        max_y = max(fifo_mean.max(), priority_mean.max()) + 5
        plt.ylim(min_y, max_y)

    else:
        fifo_agg = fifo_group.max()
        priority_agg = priority_group.max()
        plt.plot(fifo_agg.index, fifo_agg.values, marker='o', linestyle='-', label="FIFO")
        plt.plot(priority_agg.index, priority_agg.values, marker='s', linestyle='-', label="Priority Queue")
        min_y = min(fifo_agg.min(), priority_agg.min()) - 5
        max_y = max(fifo_agg.max(), priority_agg.max()) + 5
        plt.ylim(min_y, max_y)

    plt.xlabel("Pushback Delay (minutes)")
    plt.ylabel(f"{metric.capitalize()} Takeoff Delay (minutes)")
    plt.title(f"{metric.capitalize()} Takeoff Delay vs Pushback Delay for Non-Perturbed Flights")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig("../figures/" + output_name)


# Pareto Plotting
def compute_pareto_frontier(points):
    """Computes the Pareto frontier for a set of points where each point is a tuple (nondelayed_delay, delayed_delay, alpha)."""
    pareto = []
    for point in points:
        dominated = False
        for other in points:
            if (other[0] <= point[0] and other[1] <= point[1]) and ((other[0] < point[0]) or (other[1] < point[1])):
                dominated = True
                break
        if not dominated:
            pareto.append(point)
    pareto.sort(key=lambda x: x[0])
    return pareto


def plot_pareto_frontier_delays(nondelayed_vals, delayed_vals, alphas, output_name):
    """Plots the Pareto frontier for the given nondelayed and delayed flight delays along with all data points."""
    points = list(zip(nondelayed_vals, delayed_vals, alphas))
    pareto_points = compute_pareto_frontier(points)
    plt.rcParams.update({
        'axes.titlesize': 18,
        'axes.labelsize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14
    })
    plt.figure(figsize=(10, 6))
    plt.scatter(nondelayed_vals, delayed_vals, label="All Alphas", color="blue")
    if pareto_points:
        pareto_x = [p[0] for p in pareto_points]
        pareto_y = [p[1] for p in pareto_points]
        plt.plot(pareto_x, pareto_y, marker="D", linestyle="--", color="red", label="Pareto Frontier")
    # Annotate alphas on plot
    # for i, alpha in enumerate(alphas):
    #     plt.annotate(f'{alpha:.2f}', (nondelayed_vals[i], delayed_vals[i]),
    #                  textcoords="offset points", xytext=(0, 5), ha='center')
    plt.xlabel("Mean Nondelayed Takeoff Delay (minutes)")
    plt.ylabel("Mean Delayed Takeoff Delay (minutes)")
    plt.title("Pareto Frontier: Nondelayed vs Delayed Takeoff Delays")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(output_name)


def plot_pareto_frontier_max_nondelayed_vs_mean_delayed(nondelayed_vals_max, delayed_vals_mean, alphas, output_name):
    """Plots the Pareto frontier for max nondelayed takeoff delay vs. mean delayed takeoff delay."""
    points = list(zip(nondelayed_vals_max, delayed_vals_mean, alphas))
    pareto_points = compute_pareto_frontier(points)
    plt.rcParams.update({
        'axes.titlesize': 18,
        'axes.labelsize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14
    })
    plt.figure(figsize=(10, 6))
    plt.scatter(nondelayed_vals_max, delayed_vals_mean, label="All Alphas", color="blue")
    if pareto_points:
        pareto_x = [p[0] for p in pareto_points]
        pareto_y = [p[1] for p in pareto_points]
        plt.plot(pareto_x, pareto_y, marker="D", linestyle="--", color="red", label="Pareto Frontier")
    # Annotate alphas on plot
    # for i, alpha in enumerate(alphas):
    #     plt.annotate(f'{alpha:.2f}', (nondelayed_vals_max[i], delayed_vals_mean[i]),
    #                  textcoords="offset points", xytext=(0, 5), ha='center')
    plt.xlabel("Max Nondelayed Takeoff Delay (minutes)")
    plt.ylabel("Mean Delayed Takeoff Delay (minutes)")
    plt.title("Pareto Frontier: Max Nondelayed vs Mean Delayed Takeoff Delays")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(output_name)





def plot_pushback_delay_distributions(eventLogs):
    delay_props = sorted(eventLogs.keys())
    cmap = plt.get_cmap("viridis")
    plt.figure(figsize=(10, 6))
    for i, dp in enumerate(delay_props):
        sim_results = eventLogs[dp]
        logs = list(sim_results.values()) if isinstance(sim_results, dict) else [sim_results]
        all_pushback_delays = []
        for log in logs:
            effective_log = log["priority"] if (isinstance(log, dict) and "priority" in log) else log
            # Assuming get_pushback_delays() returns a dict mapping flight numbers to delay in minutes
            delays = list(effective_log.get_pushback_delays().values())
            all_pushback_delays.extend(delays)
        color = cmap(i / len(delay_props))
        sns.kdeplot(all_pushback_delays, label=f"Prop = {dp:.2f}", color=color)
    plt.xlabel("Pushback Delay (minutes)")
    plt.ylabel("Density")
    plt.title("Pushback Delay Distributions by Delay Propensity")
    plt.legend(fontsize=8)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig("../figures/Exp_7_Pushback_Delay_Distributions.png")



# Visualize by experiments
def visualize_experiment_1():
    # Plot single flight perturbations
    perturbation_logs = load_event_logs_from_pickle("MC_eventLog_exp_1_v2.pkl")

    plot_max_takeoff_delay_vs_pushback_delay_single_flight_perturbations_smoothed(perturbation_logs,
                                                                                  output_name="Exp 1/TO Delay vs pushback delay for single flight v2.png")

    plot_effect_of_reordering_on_nonperturbed_flights(perturbation_logs, metric="mean",
                                                      output_name="Exp 1/Mean TO Delays vs pushback delay on non-perturbed flights v2.png")

    plot_effect_of_reordering_on_nonperturbed_flights(perturbation_logs, metric="max",
                                                      output_name="Exp 1/Max TO Delays vs pushback delay on non-perturbed flights v2.png")


def visualize_experiment_2():
    # Plot single flight perturbations
    perturbation_logs = load_event_logs_from_pickle("MC_eventLog_exp_2.pkl")

    plot_max_takeoff_delay_vs_pushback_delay_single_flight_perturbations_smoothed(perturbation_logs,
                                                                                  output_name="Exp 2/TO Delay vs "
                                                                                              "pushback delay for "
                                                                                              "single flight DFW.png")

    plot_effect_of_reordering_on_nonperturbed_flights(perturbation_logs, metric="mean",
                                                      output_name="Exp 2/Mean TO Delays vs pushback delay on "
                                                                  "non-perturbed flights DFW.png")

    plot_effect_of_reordering_on_nonperturbed_flights(perturbation_logs, metric="max",
                                                      output_name="Exp 2/Max TO Delays vs pushback delay on "
                                                                  "non-perturbed flights DFW.png")

    # Choose an arbitrary flight to plot
    # plot_max_takeoff_delay_vs_pushback_delay_single_flight_perturbations(perturbation_logs, flight_number='UN6298')


def visualize_experiment_3():
    eventLogs = load_event_logs_from_pickle("MC_eventLog_exp_3.pkl")
    fifo_event_logs = eventLogs[500]
    priority_event_logs = eventLogs[0]
    plot_mean_takeoff_delay_vs_pushback_delay(fifo_event_logs, priority_event_logs, show_error_bars=True,
                                              output_name="Exp 3/TO vs. Pushback.png")
    draw_simulation_gantt(priority_event_logs[1], output_name="Exp 3/Priority Gantt Chart.png")
    draw_simulation_gantt(fifo_event_logs[1], output_name="Exp 3/FIFO Gantt Chart.png")


def visualize_experiment_4():
    eventLogs = load_event_logs_from_pickle("MC_eventLog_exp_4.pkl")
    fifo_event_logs = eventLogs[80]
    priority_event_logs = eventLogs[0]
    # plot_mean_takeoff_delay_vs_pushback_delay(fifo_event_logs, priority_event_logs, show_error_bars= True,
    #                                           output_name="Exp 4/TO vs. Pushback Config 1 v2.png")
    # draw_simulation_gantt(priority_event_logs[1], output_name="Priority Gantt Chart.png")
    # draw_simulation_gantt(fifo_event_logs[1], output_name="FIFO Gantt Chart.png")
    # show_pushback_delay_distribution(fifo_event_logs)

    show_virtual_queue_length_fluctuations(priority_event_logs[0], output_name="Exp 4/Virtual Queue Length.png")


def visualize_experiment_5():
    eventLogs = load_event_logs_from_pickle("MC_eventLog_exp_5.pkl")
    # plot_max_takeoff_delay_for_alpha(eventLogs, output_name="Exp 5/Max TO Delay vs. Alpha.png")
    plot_mean_takeoff_delay_delayed_flights_for_alpha(eventLogs,
                                                      output_name="Exp 5/Mean TO Delay Delayed vs. Alpha.png")
    plot_max_takeoff_delay_nondelayed_flights_for_alpha(eventLogs,
                                                        output_name="Exp 5/Max TO Delay Nondelayed vs. Alpha.png")
    plot_mean_takeoff_delay_nondelayed_flights_for_alpha(eventLogs,
                                                         output_name="Exp 5/Mean TO Delay Nondelayed vs. Alpha.png")

    # --- Pareto Frontier Visualization: Mean vs Mean ---
    alphas = []
    delayed_vals = []
    nondelayed_vals = []
    for alpha, logs in eventLogs.items():
        delayed_val = np.mean([log.get_mean_takeoff_delays_delayed_flights(delay_threshold=5) for log in logs])
        nondelayed_val = np.mean([log.get_mean_takeoff_delays_nondelayed_flights(delay_threshold=5) for log in logs])
        alphas.append(alpha)
        delayed_vals.append(delayed_val)
        nondelayed_vals.append(nondelayed_val)

    plot_pareto_frontier_delays(nondelayed_vals, delayed_vals, alphas, "../figures/Exp 5/Pareto Frontier TO Delays.png")

    # --- Pareto Frontier Visualization: Max Nondelayed vs Mean Delayed ---
    nondelayed_vals_max = []
    delayed_vals_mean = []
    for alpha, logs in eventLogs.items():
        max_nondelayed = np.mean([log.get_max_takeoff_delays_nondelayed_flights(delay_threshold=5) for log in logs])
        mean_delayed = np.mean([log.get_mean_takeoff_delays_delayed_flights(delay_threshold=5) for log in logs])
        nondelayed_vals_max.append(max_nondelayed)
        delayed_vals_mean.append(mean_delayed)

    plot_pareto_frontier_max_nondelayed_vs_mean_delayed(nondelayed_vals_max, delayed_vals_mean, alphas,
                                                        "../figures/Exp 5/Pareto Frontier Max vs Mean TO Delays.png")


def visualize_experiment_6():
    eventLogs = load_event_logs_from_pickle("MC_eventLog_exp_6.pkl")
    FIFO_logs = eventLogs[1000]
    priority_logs = eventLogs[0]
    draw_simulation_gantt(priority_logs[1], output_name="Exp 6/Priority Gantt Chart.png")
    plot_max_takeoff_delay_vs_pushback_delay(FIFO_logs, priority_logs, output_name="Exp 6/Max TO vs. Pushback.png")


def visualize_experiment_7():
    eventLogs_by_propensity = load_event_logs_from_pickle("MC_eventlog_exp_7_v2.pkl")

    for label, alpha_logs in eventLogs_by_propensity.items():
        alphas = []
        delayed_vals = []
        nondelayed_vals = []
        for alpha, logs in alpha_logs.items():
            delayed_val = np.mean([log.get_mean_takeoff_delays_delayed_flights(delay_threshold=5) for log in logs])
            nondelayed_val = np.mean([log.get_mean_takeoff_delays_nondelayed_flights(delay_threshold=5) for log in logs])
            alphas.append(alpha)
            delayed_vals.append(delayed_val)
            nondelayed_vals.append(nondelayed_val)

        plot_pareto_frontier_delays(nondelayed_vals, delayed_vals, alphas,
                                    f"../figures/Exp 7/Pareto Frontier TO Delays - {label}.png")

        nondelayed_vals_max = []
        delayed_vals_mean = []
        for alpha, logs in alpha_logs.items():
            max_nondelayed = np.mean([log.get_max_takeoff_delays_nondelayed_flights(delay_threshold=5) for log in logs])
            mean_delayed = np.mean([log.get_mean_takeoff_delays_delayed_flights(delay_threshold=5) for log in logs])
            nondelayed_vals_max.append(max_nondelayed)
            delayed_vals_mean.append(mean_delayed)

        plot_pareto_frontier_max_nondelayed_vs_mean_delayed(nondelayed_vals_max, delayed_vals_mean, alphas,
                                                            f"../figures/Exp 7/Pareto Frontier Max vs Mean TO Delays - {label}.png")



if __name__ == "__main__":
    visualize_experiment_7()
    # visualize_experiment_6()
