from datetime import timedelta


class Departure:
    AT_GATE = 0
    READY_FOR_TAXI = 1
    IN_TAXI = 2
    IN_TAKEOFF_QUEUE = 3
    DEPARTED = 4

    aircraft_classification_dict = {
        'B77L': 'H', 'B763': 'H', 'B190': 'S', 'B38M': 'L', 'A321': 'L', 'A320': 'L',
        '74F': 'H', 'E75L': 'L', 'A20N': 'L', 'B738': 'L', 'A21N': 'L', 'BCS1': 'L',
        'C208': 'S', 'B752': '757', 'A319': 'L', '741': 'H', 'E170': 'L', 'CRJ9': 'L',
        'MD11': 'H', 'CRJ7': 'L', 'B739': 'L', '777': 'H', 'CL35': 'S', 'B737': 'L',
        'BCS3': 'L', 'B788': 'H', 'A306': 'H', 'B789': 'H', 'C28': 'S', 'PC24': 'S',
        'CL60': 'S', 'B772': 'H', 'E135': 'L', 'B77W': 'H', 'LJ60': 'S', 'GLF4': 'S',
        'BE40': 'S', 'B39M': 'L', 'B741': 'H', 'E190': 'L', 'C750': 'S', 'B744': 'H',
        'A343': 'H', 'CNC': 'S', 'C68A': 'S', 'A35K': 'H', 'A388': 'SUPER', '74Y': 'H',
        'ABM': 'S', 'M11': 'H', 'B742': 'H', 'A333': 'H', '763': 'H'
    }

    def __init__(self, flight_number, actual_departure_time, scheduled_departure_time, aircraft, config):
        self.flight_number = flight_number
        self.actual_departure_time = actual_departure_time
        self.scheduled_departure_time = scheduled_departure_time
        self.status = Departure.AT_GATE
        self.aircraft = aircraft
        self.aircraft_class = Departure.aircraft_classification_dict[aircraft]
        self.runway_assignment = None

        # Event times
        self.taxi_start_time = None
        self.taxi_end_time = None
        self.service_start_time = None
        self.service_end_time = None
        self.total_gate_wait = 0  # Minutes waited at gate
        self.terminalAssignment = None

        # For visualization
        self.eventLog = []

    def __str__(self):
        return self.flight_number

    def increment_status(self, new_status, current_time, verbose):
        self.status = new_status
        self.eventLog.append(current_time)
        if verbose:
            print(f"Flight {self.flight_number} status updated to {self.get_status_string()} at {current_time}.")

    def get_min_separation(self):
        # Source: https://www.faa.gov/air_traffic/publications/atpubs/atc_html/chap3_section_9.html

        if self.aircraft_class == "SUPER":
            return timedelta(minutes=3)

        if self.aircraft_class == "H":
            return timedelta(minutes=2)

        if self.aircraft_class == "757":
            return timedelta(minutes=2)

        else:
            return timedelta(minutes = 0)

    def get_status_string(self):
        status_map = {
            Departure.AT_GATE: "AT GATE",
            Departure.READY_FOR_TAXI: "READY FOR TAXI",
            Departure.IN_TAXI: "IN TAXI",
            Departure.IN_TAKEOFF_QUEUE: "IN TAKEOFF QUEUE",
            Departure.DEPARTED: "DEPARTED"
        }
        return status_map.get(self.status, "UNKNOWN")
