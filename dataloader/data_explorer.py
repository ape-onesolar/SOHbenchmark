import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat


class MITBatteryDataExplorer:
    def __init__(self, data_root="data/MIT", output_dir="output", plots_dir="plots"):
        """
        Initialize the battery data explorer

        :param data_root: Root directory of MIT battery dataset
        :param output_dir: Directory to save CSV outputs
        :param plots_dir: Directory to save plot images
        """
        self.data_root = data_root
        self.output_dir = output_dir
        self.plots_dir = plots_dir

        # Create output directories if they don't exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)

        self.battery_data = {}
        self.summary_data = []

    def load_charge_data(self):
        """
        Load charge cycle data from .mat files
        """
        charge_path = os.path.join(self.data_root, "charge")
        charge_files = os.listdir(charge_path)

        for file in charge_files:
            full_path = os.path.join(charge_path, file)
            mat = loadmat(full_path)
            battery = mat["battery"]

            # Process each battery in the dataset
            for battery_idx in range(battery.shape[1]):
                battery_cycles = battery[0, battery_idx][0]

                # Extract cycle-level data
                cycle_data = []
                capacities = []

                for cycle_idx in range(battery_cycles.shape[1]):
                    cycle = battery_cycles[0, cycle_idx]

                    # Extract relevant features
                    time = cycle["relative_time_min"]
                    current = cycle["current_A"]
                    voltage = cycle["voltage_V"]
                    temperature = cycle["temperature_C"]
                    capacity = cycle["capacity"][0]

                    # Combine features
                    cycle_features = {
                        "battery_id": battery_idx + 1,
                        "cycle_type": "charge",
                        "cycle_idx": cycle_idx,
                        "time_mean": np.mean(time),
                        "time_max": np.max(time),
                        "current_mean": np.mean(current),
                        "current_max": np.max(current),
                        "voltage_mean": np.mean(voltage),
                        "voltage_max": np.max(voltage),
                        "temperature_mean": np.mean(temperature),
                        "temperature_max": np.max(temperature),
                        "capacity": capacity,
                    }

                    cycle_data.append(cycle_features)
                    capacities.append(capacity)

                # Store battery data
                self.battery_data[f"battery_{battery_idx+1}_charge"] = {"cycles": cycle_data, "capacities": capacities}

                # Add to summary data
                self.summary_data.extend(cycle_data)

        print(f"Loaded charge data for {len(self.battery_data)} battery datasets")

    def load_partial_charge_data(self):
        """
        Load partial charge cycle data from .mat files
        """
        partial_path = os.path.join(self.data_root, "partial_charge")
        partial_files = os.listdir(partial_path)

        for file in partial_files:
            full_path = os.path.join(partial_path, file)
            mat = loadmat(full_path)
            battery = mat["battery"]

            # Process each battery in the dataset
            for battery_idx in range(battery.shape[1]):
                battery_cycles = battery[0, battery_idx][0]

                # Extract cycle-level data
                cycle_data = []
                capacities = []

                for cycle_idx in range(battery_cycles.shape[1]):
                    cycle = battery_cycles[0, cycle_idx]

                    # Extract relevant features
                    time = cycle["relative_time_min"]
                    current = cycle["current_A"]
                    voltage = cycle["voltage_V"]
                    temperature = cycle["temperature_C"]
                    capacity = cycle["capacity"][0]

                    # Combine features
                    cycle_features = {
                        "battery_id": battery_idx + 1,
                        "cycle_type": "partial_charge",
                        "cycle_idx": cycle_idx,
                        "time_mean": np.mean(time),
                        "time_max": np.max(time),
                        "current_mean": np.mean(current),
                        "current_max": np.max(current),
                        "voltage_mean": np.mean(voltage),
                        "voltage_max": np.max(voltage),
                        "temperature_mean": np.mean(temperature),
                        "temperature_max": np.max(temperature),
                        "capacity": capacity,
                    }

                    cycle_data.append(cycle_features)
                    capacities.append(capacity)

                # Store battery data
                self.battery_data[f"battery_{battery_idx+1}_partial_charge"] = {
                    "cycles": cycle_data,
                    "capacities": capacities,
                }

                # Add to summary data
                self.summary_data.extend(cycle_data)

        print(f"Loaded partial charge data for {len(self.battery_data)} battery datasets")

    def summarize_dataset(self):
        """
        Provide a summary of the loaded battery datasets and save to CSV
        """
        print("\n--- Battery Dataset Summary ---")

        # Convert summary data to DataFrame
        summary_df = pd.DataFrame(self.summary_data)

        # Extract capacities and flatten
        all_capacities = summary_df["capacity"].apply(lambda x: x[0] if isinstance(x, np.ndarray) else x).values

        # Overall dataset statistics
        total_batteries = len(self.battery_data)
        total_cycles = len(self.summary_data)

        print(f"Total Batteries: {total_batteries}")
        print(f"Total Cycles: {total_cycles}")
        print(f"Capacity Statistics:")
        print(f"  Mean Capacity: {np.mean(all_capacities):.2f}")
        print(f"  Min Capacity: {np.min(all_capacities):.2f}")
        print(f"  Max Capacity: {np.max(all_capacities):.2f}")
        print(f"  Capacity Std Dev: {np.std(all_capacities):.2f}")

        # Group-level statistics
        print("\nCapacity Statistics by Cycle Type:")

        # Ensure capacities are flattened for groupby
        summary_df["flattened_capacity"] = summary_df["capacity"].apply(
            lambda x: x[0] if isinstance(x, np.ndarray) else x
        )
        
        # drop duplicated capcity column
        summary_df = summary_df.drop(columns=['capacity'])

        # group stats
        grouped_stats = summary_df.groupby("cycle_type")["flattened_capacity"].agg(["mean", "min", "max", "std"])
        print(grouped_stats)

        # Save summary to CSV
        summary_path = os.path.join(self.output_dir, "battery_cycle_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSummary saved to {summary_path}")

        # Save grouped statistics to CSV
        grouped_stats_path = os.path.join(self.output_dir, "battery_cycle_type_summary.csv")
        grouped_stats.to_csv(grouped_stats_path)
        print(f"Grouped statistics saved to {grouped_stats_path}")

    
def main():
    # Initialize explorer
    explorer = MITBatteryDataExplorer()

    # Load datasets
    explorer.load_charge_data()
    explorer.load_partial_charge_data()

    # Summarize dataset and save to CSV
    explorer.summarize_dataset()


if __name__ == "__main__":
    main()
