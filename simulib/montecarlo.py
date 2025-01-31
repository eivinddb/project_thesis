
from abc import ABC, abstractmethod
import numpy as np

class MonteCarlo:
    """ Class handling all logic related to Monte Carlo simulation """
    paths = []  # Store all simulation paths

    def __init__(self, path_class, num_simulations) -> None:
        """
        Initialize the MonteCarlo class.

        :param path_class: The child class of MonteCarlo.Path
        :param num_simulations: Number of simulations to run
        """
        if not issubclass(path_class, MonteCarlo.Path):
            raise TypeError(f"{path_class} is not a subclass of MonteCarlo.Path")

        self.path_class = path_class  # The child class that will be used to create paths
        self.num_simulations = num_simulations

    def run_simulation(self):
        """Run the simulation for the number of paths"""
        for i in range(self.num_simulations):
            # Create a new path instance using the provided child class
            self.paths.append(self.path_class())  # Store the path

    class Path(ABC):
        """ Abstract class ensuring that state_variables and cash flow are defined """
 
        def __init__(self) -> None:
            self.state_variables = self.simulate_state_variables()
            self.cash_flows = self.calculate_cash_flows()

        @abstractmethod
        def simulate_state_variables(self) -> dict:
            """ Returns a dictionary with key: state_variable name """
            pass

        @abstractmethod
        def calculate_cash_flows(self) -> list:
            """A required function that child classes must define."""
            pass

        def __str__(self) -> str:
            return f"{self.__class__.__name__}(Cash Flows: {self.calculate_cash_flows()[:5]})"
