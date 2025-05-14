from itertools import combinations, product
import time

def optimal_ordering(input_F):
    """
    :param F: List of all flights, where each flight is given by
              (Scheduled Pushback Time, Pushback Delay)
    :return: (List: Optimal ordering, Maximum Takeoff Delay)
    """

    # Sort by Actual Pushback Time (Makes it easier to see deviations from FIFO in the end result)
    F_sorted = sorted(input_F, key=lambda f: (f[0] + f[1], f[0]))
    # Index flights sequentially
    F = {i: val for i, val in enumerate(F_sorted)}

    T = {}

    times = 200  # Range of times

    # Parameters
    runway_service_time = 5  # Expected taxi time - Deterministic for now

    # Base Cases
    for i in range(len(F)):
        for start_time in range(times):
            # Optimal ordering of a single flight is the flight
            # Max takeoff delay is the flight's own takeoff delay

            earliest_start_time = max(start_time, F[i][0] + F[i][1])  # Later of now and actual pushback time
            T[(i,), start_time] = ((i,), earliest_start_time + runway_service_time, earliest_start_time + runway_service_time - F[i][0])

    # Recursive Cases
    # Orders to tabulate topologically sorted
    all_orders = [list(subset) for r in range(2, len(F) + 1) for subset in combinations(F.keys(), r)]

    for order in all_orders:
        for start_time in range(times):

            def argmin_body(fi):
                """
                Define the function to minimize in the Bellman Equation
                :param fi: input flight argument
                :return: Max Takeoff Delay if fi is selected as next in the sequence
                """

                earliest_start_time = max(start_time, F[fi][0] + F[fi][1])

                cost_remaining_subproblem = T.get((tuple(elem for elem in order if elem != fi),
                                                   earliest_start_time + runway_service_time), ((-1,), -1, -1))[2]

                cost_this_flight = earliest_start_time + runway_service_time - F[fi][0]

                return max(cost_this_flight, cost_remaining_subproblem)

            # Argmin call
            fi_argmin = min(order, key=argmin_body)


            earliest_start_time = max(start_time, F[fi_argmin][0] + F[fi_argmin][1])
            next_available_time = earliest_start_time + runway_service_time

            # Leftover ordering subproblem
            remaining_subproblem = T.get(
                (tuple(elem for elem in order if elem != fi_argmin), next_available_time),
                ((-1,), -1, -1))

            # Final Tabulation
            T[tuple(order), start_time] = (fi_argmin,) + remaining_subproblem[0], \
                                          remaining_subproblem[1], \
                                          max(remaining_subproblem[2],
                                              earliest_start_time + runway_service_time - F[fi_argmin][0])

    return T[tuple(F.keys()), 0]

def optimal_ordering_multi_runway(input_F, num_runways=3):
    """
    :param input_F: List of all flights, where each flight is given by
                    (Scheduled Pushback Time, Pushback Delay)
    :param num_runways: Number of runways available.
    :return: Tuple containing the optimal sequence of (flight index, runway index) assignments and
             the maximum takeoff delay.
    """
    runway_service_time = 3
    F = {i: flight for i, flight in enumerate(input_F)}

    from functools import lru_cache

    @lru_cache(maxsize=None)
    def dp(remaining, runway_state):
        # remaining is a frozenset of flight indices; runway_state is a tuple of next available times per runway
        if not remaining:
            return ((), 0)
        best_ordering = None
        best_cost = float('inf')
        for i in remaining:
            scheduled_pushback, pushback_delay = F[i]
            actual_pushback = scheduled_pushback + pushback_delay
            for r in range(num_runways):
                available = runway_state[r]
                earliest_start = max(available, actual_pushback)
                finish = earliest_start + runway_service_time
                delay = finish - scheduled_pushback
                new_runway_state = list(runway_state)
                new_runway_state[r] = finish
                new_remaining = frozenset(remaining - {i})
                sub_ordering, sub_cost = dp(new_remaining, tuple(new_runway_state))
                cost = max(delay, sub_cost)
                if cost < best_cost:
                    best_cost = cost
                    best_ordering = ((i, r),) + sub_ordering
        return (best_ordering, best_cost)

    initial_runway_state = (0,) * num_runways
    remaining = frozenset(F.keys())
    ordering, max_delay = dp(remaining, initial_runway_state)
    return ordering, max_delay

if __name__ == "__main__":

    # flight = (Scheduled Pushback Time, Pushback Delay)
    f0 = (1, 2)
    f1 = (2, 1)
    f2 = (3, 4)
    f3 = (4, 0)
    f4 = (5, 3)
    f5 = (6, 1)
    f6 = (7, 2)
    f7 = (8, 0)
    f8 = (9, 3)
    f9 = (10, 2)
    f10 = (11, 0)
    f11 = (12, 1)
    f12 = (13, 3)
    f13 = (14, 2)
    f14 = (15, 0)


    F = f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14

    start = time.time()
    T = optimal_ordering(F)
    end = time.time()
    print("Time elapsed:", end - start, "seconds")
    # T = optimal_ordering_multi_runway(F)

    print(T)
