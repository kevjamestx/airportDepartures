import numpy as np
import matplotlib
import multiprocessing
from datetime import timedelta, datetime
import pandas as pd
import re
from simulation.runway import Runway
from simulation.airport import Airport
from eventLog import EventLog
from visualization.visualizer import *
from discipline_simulation import *
import random

matplotlib.use('MACOSX')

def initialize_simulation(seed=None):
    if seed is not None:
        np.random.seed(seed)  # Set seed for NumPy random functions
        random.seed(seed)  # Set seed for built-in Python random functions

# Helper function for multiprocessing in random perturbation mode
def run_single_simulation(args):
    """ Helper function to run a single simulation with multiprocessing."""
    alpha, runway_strs, terminals, discipline, CONFIG, time_bounds, verbose, filename = args
    CONFIG_copy = CONFIG.copy()
    CONFIG_copy['alpha'] = alpha
    # print(f"Simulation completed for alpha={alpha}")
    return alpha, discipline_simulation(runway_strs, terminals, discipline, CONFIG_copy, time_bounds, verbose, filename)

# Helper function for multiprocessing in single perturbation mode
def run_simulation_task_single_perturbation(args):
    runways, terminals, CONFIG, time_bounds, flight_index, perturbation, filename = args
    departures_df = read_txt_to_dataframe(filename)
    departures_df = departures_df[(departures_df["STD"] >= time_bounds[0]) &
                                  (departures_df["STD"] <= time_bounds[1])]
    flight_number = departures_df.iloc[flight_index]['Flight Number']
    return flight_number, perturbation, wrap_simulation_with_single_perturbation(
        runways, terminals, CONFIG, time_bounds=time_bounds,
        test_flight_index=flight_index, perturbation_minutes=perturbation, filename=filename)


def wrap_simulation(runway_strs, terminals, discipline, CONFIG, num_runs=500, alphas=np.linspace(0,10,21),
                    time_bounds=None, verbose=False, filename="../DFW data.txt"):
    """Parallelized Monte Carlo simulation wrapper.
       Runs multiple simulations at multiple levels of alpha and reports spread of max takeoff delays"""
    result = {}  # Dictionary mapping alpha to a list of event logs

    # Create a list of tasks for parallel execution
    tasks = [(alpha, runway_strs, terminals, discipline, CONFIG, time_bounds, verbose, filename)
             for alpha in alphas for _ in range(num_runs)]

    # Use multiprocessing to run simulations in parallel
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.map(run_single_simulation, tasks)

    # Collect results
    for alpha, log in results:
        result.setdefault(alpha, []).append(log)

    return result

def wrap_simulation_with_single_perturbation(runway_strs, terminals, CONFIG, test_flight_index=0, perturbation_minutes=5,
                                      time_bounds = None, filename="../Toy Problem Data.txt", seed = 42):
    """
    Runs FIFO and Priority Queue simulations with a perturbation applied to a single flight's pushback time.

    - Uses scheduled STD times and applies delays instead of direct pushback times.
    - Perturbs the selected flight's delay while keeping others unchanged.
    - Runs FIFO and Priority Queue simulations for comparison.

    :param runway_strs: List of runway identifiers
    :param terminals: List of terminal identifiers
    :param CONFIG: Configuration dictionary
    :param test_flight_index: Index of the flight to perturb in the departure schedule
    :param perturbation_minutes: Amount to perturb the selected flight's pushback time
    :param filename: Path to departure schedule file
    :return: Dictionary with results from FIFO and Priority queue simulations
    """

    departures_df = read_txt_to_dataframe(filename)
    if time_bounds is not None:
        departures_df = departures_df[(departures_df["STD"] >= time_bounds[0]) &
                                      (departures_df["STD"] <= time_bounds[1])]



    # Initialize delays as zero
    delays = pd.Series([0] * len(departures_df), index=departures_df.index)

    # Perturb a single flight's delay
    if test_flight_index < len(delays):
        delays.iloc[test_flight_index] += perturbation_minutes
        perturbed_flight = departures_df.iloc[test_flight_index]['Flight Number']
        # print(f"Perturbing {perturbed_flight}: +{perturbation_minutes} min delay")

    results = {}

    initialize_simulation(seed = seed)
    results['FIFO'] = discipline_simulation(runway_strs, terminals, "FIFO", CONFIG, time_bounds = time_bounds,
                                            filename=filename, actual_pushback_delays=delays)

    CONFIG_copy = CONFIG.copy()
    CONFIG_copy['alpha'] = 1
    results['Mixed priority'] = discipline_simulation(runway_strs, terminals, "FIFO", CONFIG_copy, time_bounds = time_bounds,
                                            filename=filename, actual_pushback_delays=delays)

    initialize_simulation(seed = seed)
    results['priority'] = discipline_simulation(runway_strs, terminals, "priority", CONFIG, time_bounds = time_bounds,
                                                filename=filename, actual_pushback_delays=delays)


    return {perturbed_flight: results}

def run_experiment_1(output_name = "MC_eventlog_exp_1.pkl"):
    """
    Toy Problem.
    Analyze single flight perturbations
    """

    CONFIG = {
        'PSRA_shape': 0.865,
        'PSRA_loc': -9.323,
        'PSRA_scale': 11.227,
        'travel_time_k': 8.533,
        'travel_time_scale': 0.83,
        'service_time_mu': 0.5,
        'service_time_sigma': 0.125,
        'gate_wait_penalty': 2,
        'queue_wait_penalty': 3,
        'alpha': 0
    }

    # Toy Problem

    runways = ["1"]
    terminals = ["A"]
    discipline = "priority"
    filename = "../Toy Problem Data.txt"

    result = {}
    departures_df = read_txt_to_dataframe(filename)
    for flight_index in range(50):
        flight_number = departures_df.iloc[flight_index]['Flight Number']
        for perturbation in range(90):
            sim_result = wrap_simulation_with_single_perturbation(runways, terminals, CONFIG,
                                                                  test_flight_index=flight_index,
                                                                  perturbation_minutes=perturbation,
                                                                  filename=filename)
            result.setdefault(flight_number, {})[perturbation] = sim_result[flight_number]

    EventLog.save_event_logs_to_pickle(result, filename=output_name)

def run_experiment_2(output_name = "MC_eventlog_exp_2.pkl"):
    """
    DFW Airport - single flight perturbations
    5 runways
    0800 - 1200 - busy period
    ~350 departures
    """

    CONFIG = {
        'PSRA_shape': 0.865,
        'PSRA_loc': -9.323,
        'PSRA_scale': 11.227,
        'travel_time_k': 8.533,
        'travel_time_scale': 0.83,
        'service_time_mu': 0.5,
        'service_time_sigma': 0.125,
        'gate_wait_penalty': 2,
        'queue_wait_penalty': 3,
        'alpha': 0
    }

    # DFW Airport

    runways = ["17L", "17R", "18R"]
    terminals = ['A', 'B', 'C', 'D', 'E', 'Cargo', 'Corporate']
    discipline = "priority"
    filename = "../DFW Data.txt"

    # Bound from 0800 to 1200
    time_bounds = [datetime.now().replace(hour= 8, minute=0, second=0, microsecond=0),
                   datetime.now().replace(hour=12, minute=0, second=0, microsecond=0)]

    perturbation_times = 90 #Perturb each departure by 1 minute up to 90 minutes

    departures_df = read_txt_to_dataframe(filename)
    departures_df = departures_df[(departures_df["STD"] >= time_bounds[0]) &
                                  (departures_df["STD"] <= time_bounds[1])]
    num_flights = len(departures_df)

    # Prepare tasks for multiprocessing
    tasks = [
        (runways, terminals, CONFIG, time_bounds, flight_index, perturbation, filename)
        for flight_index in range(num_flights)
        for perturbation in range(perturbation_times)
    ]

    # Use multiprocessing to parallelize execution
    results = {}
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        for flight_number, perturbation, sim_result in pool.imap_unordered(run_simulation_task_single_perturbation, tasks):
            results.setdefault(flight_number, {})[perturbation] = sim_result[flight_number]

    # Save results
    EventLog.save_event_logs_to_pickle(results, filename=output_name)

def run_experiment_3(output_name = "MC_eventlog_exp_3.pkl"):
    """
    Toy problem - random perturbations
    1 runway
    1 hour
    50 departures
    """
    CONFIG = {
        'PSRA_shape': 0.865,
        'PSRA_loc': -9.323,
        'PSRA_scale': 11.227,
        'travel_time_k': 8.533,
        'travel_time_scale': 0.83,
        'service_time_mu': 0.5,
        'service_time_sigma': 0.125,
        'gate_wait_penalty': 2,
        'queue_wait_penalty': 3,
        'alpha': 0
    }

    # Toy Problem
    runways = ["1"]
    terminals = ["A"]
    discipline = "priority"
    alphas = [0, 500]
    filename = "../Toy Problem Data.txt"
    num_runs = 2000

    result = wrap_simulation(runways, terminals, discipline, CONFIG, num_runs= num_runs, alphas=alphas, filename=filename)

    EventLog.save_event_logs_to_pickle(result, filename=output_name)

def run_experiment_4(output_name = "MC_eventlog_exp_4.pkl"):
    """
    DFW Airport - random perturbations
    5 runways
    0800 - 1200
    350 Departures
    """

    CONFIG = {
        'PSRA_shape': 0.865,
        'PSRA_loc': -9.323,
        'PSRA_scale': 11.227,
        'travel_time_k': 8.533,
        'travel_time_scale': 0.83,
        'service_time_mu': 0.5,
        'service_time_sigma': 0.125,
        'gate_wait_penalty': 2,
        'queue_wait_penalty': 3,
        'alpha': 0
    }

    # DFW Airport

    runways = ["17L", "17R", "18R"]
    terminals = ['A', 'B', 'C', 'D', 'E', 'Cargo', 'Corporate']
    discipline = "priority"
    filename = "../DFW Data.txt"
    alphas = [0, 80]

    # Bound from 0800 to 1200
    time_bounds = [datetime.now().replace(hour=8, minute=0, second=0, microsecond=0),
                   datetime.now().replace(hour=12, minute=0, second=0, microsecond=0)]

    result = wrap_simulation(runways, terminals, discipline, CONFIG, alphas=alphas, filename=filename, time_bounds=time_bounds)

    EventLog.save_event_logs_to_pickle(result, filename=output_name)

def run_experiment_5(output_name = "MC_eventlog_exp_5.pkl"):
    """
    Alpha Sensitivity Analysis
    DFW Airport - random perturbations
    5 runways
    0800 - 1200
    350 Departures
    """

    CONFIG = {
        'PSRA_shape': 0.865,
        'PSRA_loc': -9.323,
        'PSRA_scale': 11.227,
        'travel_time_k': 8.533,
        'travel_time_scale': 0.83,
        'service_time_mu': 0.5,
        'service_time_sigma': 0.125,
        'gate_wait_penalty': 2,
        'queue_wait_penalty': 3,
        'alpha': 0
    }

    # DFW Airport

    runways = ["17L", "17R", "18R"]
    terminals = ['A', 'B', 'C', 'D', 'E', 'Cargo', 'Corporate']
    discipline = "priority"
    filename = "../DFW Data.txt"
    alphas = [0] + np.logspace(-2, 2, num = 25)

    # Bound from 0800 to 1200
    time_bounds = [datetime.now().replace(hour=8, minute=0, second=0, microsecond=0),
                   datetime.now().replace(hour=12, minute=0, second=0, microsecond=0)]

    result = wrap_simulation(runways, terminals, discipline, CONFIG, alphas=alphas, filename=filename, time_bounds=time_bounds)

    EventLog.save_event_logs_to_pickle(result, filename=output_name)

# Disregard this experiment
def run_experiment_6(output_name = "MC_eventlog_exp_6.pkl"):
    """
    DFW Airport - stress test for large virtual queues
    I.e. make queue wait penalty large - just do FIFO vs. alpha = 0
    5 runways
    0800 - 1200
    """

    CONFIG = {
        'PSRA_shape': 0.865,
        'PSRA_loc': -9.323,
        'PSRA_scale': 11.227,
        'travel_time_k': 8.533,
        'travel_time_scale': 0.83,
        'service_time_mu': 0.5,
        'service_time_sigma': 0.125,
        'gate_wait_penalty': 2,
        'queue_wait_penalty': 3,
        'alpha': 0
    }

    # DFW Airport

    runways = ["17L", "17R", "18R"]
    terminals = ['A', 'B', 'C', 'D', 'E', 'Cargo', 'Corporate']
    discipline = "priority"
    filename = "../DFW Data.txt"
    alphas = [0, 1000]

    # Bound from 0800 to 1200
    time_bounds = [datetime.now().replace(hour=8, minute=0, second=0, microsecond=0),
                   datetime.now().replace(hour=12, minute=0, second=0, microsecond=0)]

    result = wrap_simulation(runways, terminals, discipline, CONFIG, alphas=alphas, filename=filename, time_bounds= time_bounds, num_runs=2000)

    EventLog.save_event_logs_to_pickle(result, filename=output_name)

def run_experiment_7(output_name="MC_eventlog_exp_7.pkl"):
    """
    Delay Propensity Sensitivity Analysis - Experiment 7
    DFW Airport - random perturbations
    3 runways
    0800 - 1200
    350 Departures
    """

    base_CONFIG = {
        'PSRA_shape': 0.865,
        'PSRA_loc': -9.323,
        'PSRA_scale': 11.227,
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
    filename = "../DFW Data.txt"
    alphas = [0] + np.logspace(-2, 2, num = 8)

    # Parameters which correspond to the delay evolution plot
    delay_levels = ['30%', '40%', '50%', '60%', '70%'] #, '80%', '90%']
    locs = [-18.458045, -13.977751, -10.627000, -8.053737, -5.989282]#, -4.180105, -2.503937]
    scales = [11.727, 11.227, 10.627, 10.027, 9.427] #, 8.827, 8.227]

    result = {}

    # Bound from 0800 to 1200
    time_bounds = [datetime.now().replace(hour=8, minute=0, second=0, microsecond=0),
                   datetime.now().replace(hour=12, minute=0, second=0, microsecond=0)]

    for label, loc, scale in zip(delay_levels, locs, scales):
        CONFIG = base_CONFIG.copy()
        CONFIG['PSRA_loc'] = loc
        CONFIG['PSRA_scale'] = scale
        CONFIG['PSRA_shape'] = base_CONFIG['PSRA_shape']
        sim_result = wrap_simulation(runways, terminals, discipline, CONFIG, alphas=alphas,
                                     filename=filename, time_bounds=time_bounds)
        result[label] = sim_result

    EventLog.save_event_logs_to_pickle(result, filename=output_name)




if __name__ == "__main__":
    # try:
    #     run_experiment_1("MC_eventlog_exp_1_v2.pkl")
    #     print("Completed Experiment 1")
    # except Exception as e:
    #     print(f"Error on Experiment 1: {e}")
    #
    # try:
    #     run_experiment_2("MC_eventlog_exp_2_v2.pkl")
    #     print("Completed Experiment 2")
    # except Exception as e:
    #     print(f"Error on Experiment 2: {e}")

    # try:
    #     run_experiment_7("MC_eventlog_exp_7_v2.pkl")
    #     print("Completed Experiment 7")
    # except Exception as e:
    #     print(f"Error on Experiment 7: {e}")

    # try:
    #     run_experiment_3("MC_eventlog_exp_3_v2.pkl")
    #     print("Completed Experiment 3")
    # except Exception as e:
    #     print(f"Error on Experiment 3: {e}")
    #
    # try:
    #     run_experiment_4("MC_eventlog_exp_4_v2.pkl")
    #     print("Completed Experiment 4")
    # except Exception as e:
    #     print(f"Error on Experiment 4: {e}")
    #
    # try:
    #     run_experiment_5("MC_eventlog_exp_5_v2.pkl")
    #     print("Completed Experiment 5")
    # except Exception as e:
    #     print(f"Error on Experiment 5: {e}")

    # try:
    #     run_experiment_6("MC_eventlog_exp_6_v2.pkl")
    #     print("Completed Experiment 6")
    # except Exception as e:
    #     print(f"Error on Experiment 6: {e}")

    run_experiment_7("MC_eventlog_exp_7_v2.pkl")