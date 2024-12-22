
from abc import ABC, abstractmethod
import numpy as np
import time

class MonteCarlo:
    """ Class handling all logic related to Monte Carlo simulation """

    def __init__(self, path_class, num_simulations) -> None:
        """
        Initialize the MonteCarlo class.

        :param path_class: The child class of MonteCarlo.Path
        :param num_simulations: Number of simulations to run
        """
        # if not issubclass(path_class, MonteCarlo.Path):
        #     raise TypeError(f"{path_class} is not a subclass of MonteCarlo.Path")

        self.path_class = path_class  # The child class that will be used to create paths
        self.num_simulations = num_simulations
        self.paths = []  # Store all simulation paths

        self.run_simulation()

    def run_simulation(self):
        """Run the simulation for the number of paths"""
        start = time.time()
        for i in range(self.num_simulations):
            # Create a new path instance using the provided child class
            self.paths.append(self.path_class())  # Store the path
        print(self.path_class.__name__, f"Simulation complete at {int((time.time() - start) * 1000)} ms")

    def calculate_all_cash_flows(self, **kwargs):
        ret = []
        for path in self.paths:
            ret.append(
                path.calculate_cash_flows(**kwargs)
            )
            
        return np.array(ret)


    class Path(ABC):
        """ Abstract class ensuring that state_variables and cash flow are defined """
 
        def __init__(self) -> None:
            self.state_variables = self.simulate_state_variables()

        @abstractmethod
        def simulate_state_variables(self) -> dict:
            """ Returns a dictionary with key: state_variable name """
            pass

        @abstractmethod
        def calculate_cash_flows(self, **kwargs) -> list:
            """A required function that child classes must define. Needs to modify self.cash_flows and return npv"""
            pass

        def __str__(self) -> str:
            return f"{self.__class__.__name__}(Cash Flows: {self.cash_flows[:5]})"
