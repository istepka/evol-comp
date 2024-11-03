import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Constants
DATA_DIR = Path("../data")
RESULTS_DIR = Path("results")
# New directory for exported CSVs
RESULTS_EXPORT_DIR = Path("exported_results")
PLOTS_DIR = Path("plots")
LATEX_TABLE_FILE = "results_table.tex"

# Define Instances and Methods
INSTANCES = ["TSPA", "TSPB"]
METHODS = ["cmls_inter_r", "cmls_intra_r", "cmls_inter_g", "cmls_intra_g"]


# Configure Matplotlib for serif fonts and even larger font sizes
plt.rcParams.update(
    {
        # 'text.usetex': True,
        "font.family": "serif",
        "font.size": 24,
        "axes.titlesize": 28,
        "axes.labelsize": 24,
        "legend.fontsize": 20,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
    }
)

# Number of nodes to select (50% rounded up)


def compute_nodes_to_select(total_nodes: int) -> int:
    return math.ceil(total_nodes / 2)


# Read node data


def read_node_data(instance: str) -> pd.DataFrame:
    file_path = DATA_DIR / f"{instance}.csv"
    if not file_path.exists():
        raise FileNotFoundError(
            f"Data file for instance {instance} not found at {file_path}"
        )
    df = pd.read_csv(file_path, header=None, names=["x", "y", "cost"], sep=";")
    logging.info(f"Loaded data for instance {instance} with {len(df)} nodes.")
    return df


# Parse solution file with enhanced handling for float average costs


def parse_solution_file(file_path: Path) -> Dict[str, Any]:
    result = {}
    try:
        with open(file_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if line.startswith("Average cost:"):
                # Handle float average cost
                try:
                    parts = line.split(":")[1].split(";")
                    average_cost_str = parts[0].strip()
                    result["average_cost"] = float(average_cost_str)
                    logging.debug(
                        f"Parsed average_cost: {result['average_cost']} from {file_path}"
                    )
                except (IndexError, ValueError) as e:
                    logging.error(f"Error parsing average_cost in {file_path}: {e}")
                    result["average_cost"] = None

            elif line.startswith("Worst cost:"):
                # Handle float worst cost and list of node indices
                try:
                    parts = line.split(";")
                    worst_cost_str = parts[0].split(":")[1].strip()
                    # Using float to accommodate potential floats
                    result["worst_cost"] = float(worst_cost_str)
                    worst_solution_str = parts[1].strip()
                    result["worst_solution"] = [
                        int(idx) for idx in worst_solution_str.split()
                    ]
                    logging.debug(
                        f"Parsed worst_cost: {result['worst_cost']} and worst_solution: {result['worst_solution']} from {file_path}"
                    )
                except (IndexError, ValueError) as e:
                    logging.error(
                        f"Error parsing worst_cost or worst_solution in {file_path}: {e}"
                    )
                    result["worst_cost"] = None
                    result["worst_solution"] = []

            elif line.startswith("Best cost:"):
                # Handle float best cost and list of node indices
                try:
                    parts = line.split(";")
                    best_cost_str = parts[0].split(":")[1].strip()
                    # Using float to accommodate potential floats
                    result["best_cost"] = float(best_cost_str)
                    best_solution_str = parts[1].strip()
                    result["best_solution"] = [
                        int(idx) for idx in best_solution_str.split()
                    ]
                    logging.debug(
                        f"Parsed best_cost: {result['best_cost']} and best_solution: {result['best_solution']} from {file_path}"
                    )
                except (IndexError, ValueError) as e:
                    logging.error(
                        f"Error parsing best_cost or best_solution in {file_path}: {e}"
                    )
                    result["best_cost"] = None
                    result["best_solution"] = []
            elif line.startswith("Execution time:"):
                # Handle float execution time
                try:
                    parts = line.split(":")[1].strip()
                    result["execution_time"] = float(parts)
                    logging.debug(
                        f"Parsed execution_time: {result['execution_time']} from {file_path}"
                    )
                except (IndexError, ValueError) as e:
                    logging.error(f"Error parsing execution_time in {file_path}: {e}")
                    result["execution_time"] = None

    except Exception as e:
        logging.error(f"Failed to parse solution file {file_path}: {e}")

    return result


# Collect all results


def collect_results(
    instances: List[str], methods: List[str]
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    all_results = {
        instance: {method: {} for method in methods} for instance in instances
    }

    for method in methods:
        for instance in instances:
            file_name = f"{method}_each_{instance}.csv_.solution"
            file_path = RESULTS_DIR / file_name
            if file_path.exists():
                parsed = parse_solution_file(file_path)
                all_results[instance][method] = parsed
                logging.info(
                    f"Parsed results for method '{method}' on instance '{instance}'."
                )
            else:
                logging.warning(f"Solution file {file_path} does not exist.")
    return all_results


# Generate LaTeX tables


def generate_latex_tables(
    all_results: Dict[str, Dict[str, Dict[str, Any]]],
    instances: List[str],
    methods: List[str],
    output_file: Path,
):
    with open(output_file, "w") as f:
        f.write("% LaTeX Table of Results\n")
        f.write("\\begin{table}[ht]\n\\centering\n")
        f.write("\\caption{Computational Experiment Results}\n")
        f.write("\\label{tab:results}\n")
        # Instance, Method, Best Cost, Worst Cost, Average Cost
        f.write("\\begin{tabular}{lccccc}\n")
        f.write("\\hline\n")
        f.write(
            "Instance & Method & Best Cost & Worst Cost & Average Cost & Execution Time \\\\\n"
        )
        f.write("\\hline\n")

        for instance in instances:
            for method in methods:
                res = all_results[instance][method]
                if res:
                    best_cost = (
                        f"{res['best_cost']:.2f}"
                        if res.get("best_cost") is not None
                        else "N/A"
                    )
                    worst_cost = (
                        f"{res['worst_cost']:.2f}"
                        if res.get("worst_cost") is not None
                        else "N/A"
                    )
                    average_cost = (
                        f"{res['average_cost']:.2f}"
                        if res.get("average_cost") is not None
                        else "N/A"
                    )
                    execution_time = (
                        f"{res['execution_time']:.2f}"
                        if res.get("execution_time") is not None
                        else "N/A"
                    )
                    f.write(
                        f"{instance} & {method} & {best_cost} & {worst_cost} & {average_cost} & {execution_time} \\\\\n"
                    )
                else:
                    f.write(f"{instance} & {method} & N/A & N/A & N/A & N/A \\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    logging.info(f"LaTeX table generated at {output_file}")


# Generate 2D visualization plots


def visualize_solution(
    instance: str,
    method: str,
    solution: List[int],
    node_data: pd.DataFrame,
    output_dir: Path,
) -> Optional[Path]:
    # Validate the solution list
    num_nodes = len(node_data)
    if not solution:
        logging.warning(
            f"Empty best_solution for {method} on {instance}. Skipping plot."
        )
        return None

    if any(not isinstance(idx, int) for idx in solution):
        logging.warning(
            f"Non-integer node index found in best_solution for {method} on {instance}. Skipping plot."
        )
        return None

    if any(idx < 0 or idx >= num_nodes for idx in solution):
        logging.warning(
            f"Node index out of bounds in best_solution for {method} on {instance}. Skipping plot."
        )
        return None

    # Check if the solution forms a cycle (starts and ends with the same node)
    if solution[0] != solution[-1]:
        logging.warning(
            f"Solution for {method} on {instance} does not form a cycle. Appending the starting node to complete the cycle."
        )
        solution = solution + [solution[0]]

    # Remove duplicate last node if already appended
    if len(solution) >= 2 and solution[-1] == solution[-2]:
        logging.info(
            f"Cycle already complete for {method} on {instance}. Removing duplicate node."
        )
        solution = solution[:-1]

    try:
        # Proceed with plotting
        plt.figure(figsize=(12, 8))
        selected_nodes = node_data.iloc[solution].reset_index(drop=True)
        all_nodes = node_data.reset_index(drop=True)

        # Check for NaN or invalid data
        if selected_nodes[["x", "y", "cost"]].isnull().any().any():
            logging.error(
                f"Selected nodes contain NaN values for {method} on {instance}. Skipping plot."
            )
            return None

        logging.debug(
            f"Selected nodes for {method} on {instance}:\n{selected_nodes.head()}"
        )

        # Plot selected nodes with color representing cost
        try:
            scatter = plt.scatter(
                selected_nodes["x"],
                selected_nodes["y"],
                c=selected_nodes["cost"],
                cmap="viridis",
                label="Selected Nodes",
                edgecolors="red",  # Use 'edgecolors' instead of 'edgecolor'
                linewidths=2.5,
                s=150,
            )
            plt.scatter(
                all_nodes["x"],
                all_nodes["y"],
                c=all_nodes["cost"],
                cmap="viridis",
                label="Unselected Nodes",
                # edgecolors='k',  # Use 'edgecolors' instead of 'edgecolor'
                # linewidths=0.5,
                s=150,
            )
        except Exception as e:
            logging.error(f"Error in scatter plot for {method} on {instance}: {e}")
            return None

        # Add colorbar
        try:
            plt.colorbar(scatter, label="Node Cost")
        except Exception as e:
            logging.error(f"Error adding colorbar for {method} on {instance}: {e}")

        # Draw the Hamiltonian cycle
        plt.plot(
            selected_nodes["x"],
            selected_nodes["y"],
            "r-",
            linewidth=1.5,
            label="Hamiltonian Cycle",
        )

        plt.title(f"Best Solution for {method.capitalize()} on {instance}")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.xticks([0, 1000, 2000, 3000, 4000])
        plt.legend(ncol=3, bbox_to_anchor=(0.5, -0.15), loc="upper center")

        # Adjust layout
        try:
            plt.tight_layout()
        except Exception as e:
            logging.error(f"Error during tight_layout for {method} on {instance}: {e}")

        # Save the plot as PDF
        plot_filename = f"Best_{method}_{instance}.pdf"
        plot_path = output_dir / plot_filename
        logging.warning(plot_path)
        try:
            plt.savefig(plot_path, format="pdf")
            logging.info(f"Plot saved at {plot_path}")
        except Exception as e:
            logging.error(f"Error saving plot for {method} on {instance}: {e}")
        finally:
            plt.close()
        return plot_path

    except Exception as e:
        logging.error(f"Failed to generate plot for {method} on {instance}: {e}")
        plt.close()
        return None


# New Function: Export Best Solution to CSV


def export_best_solution_to_csv(
    instance: str,
    method: str,
    solution: List[int],
    node_data: pd.DataFrame,
    output_dir: Path,
) -> Optional[Path]:
    """
    Exports the best solution cycle to a CSV file with columns: node_idx, x, y, cost.
    Each row corresponds to the next node in the cycle.
    """
    # Validate the solution list
    num_nodes = len(node_data)
    if not solution:
        logging.warning(
            f"Empty best_solution for {method} on {instance}. Skipping CSV export."
        )
        return None

    if any(not isinstance(idx, int) for idx in solution):
        logging.warning(
            f"Non-integer node index found in best_solution for {method} on {instance}. Skipping CSV export."
        )
        return None

    if any(idx < 0 or idx >= num_nodes for idx in solution):
        logging.warning(
            f"Node index out of bounds in best_solution for {method} on {instance}. Skipping CSV export."
        )
        return None

    # Check if the solution forms a cycle (starts and ends with the same node)
    if solution[0] != solution[-1]:
        logging.warning(
            f"Solution for {method} on {instance} does not form a cycle. Appending the starting node to complete the cycle."
        )
        solution = solution + [solution[0]]

    # Remove duplicate last node if already appended
    if len(solution) >= 2 and solution[-1] == solution[-2]:
        logging.info(
            f"Cycle already complete for {method} on {instance}. Removing duplicate node."
        )
        solution = solution[:-1]

    try:
        # Extract node details in the order of the solution
        selected_nodes = node_data.iloc[solution].reset_index(drop=True)
        selected_nodes.insert(0, "node_idx", solution)

        # Check for NaN or invalid data
        if selected_nodes[["node_idx", "x", "y", "cost"]].isnull().any().any():
            logging.error(
                f"Selected nodes contain NaN values for {method} on {instance}. Skipping CSV export."
            )
            return None

        # Define the output filename
        csv_filename = f"{instance}_best_{method}_checker_format.csv"
        csv_path = output_dir / csv_filename

        # Ensure the output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save to CSV
        selected_nodes.to_csv(
            csv_path, columns=["node_idx", "x", "y", "cost"], index=False
        )
        logging.info(f"Best solution CSV exported at {csv_path}")
        return csv_path

    except Exception as e:
        logging.error(
            f"Failed to export best solution CSV for {method} on {instance}: {e}"
        )
        return None


def main():
    # Step 1: Read node data
    node_data_dict = {}
    for instance in INSTANCES:
        try:
            node_data = read_node_data(instance)
            node_data_dict[instance] = node_data
        except FileNotFoundError as e:
            logging.error(e)
            return

    # Step 2: Collect all results
    all_results = collect_results(INSTANCES, METHODS)

    # Step 3: Generate LaTeX tables
    generate_latex_tables(
        all_results.copy(), INSTANCES, METHODS, Path(LATEX_TABLE_FILE)
    )

    # Step 4: Generate plots and export best solutions to CSV
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    # Ensure export directory exists
    RESULTS_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    for instance in INSTANCES:
        for method in METHODS:
            res = all_results[instance][method]
            if res and "best_solution" in res and res["best_solution"]:
                best_solution = res["best_solution"]
                node_data = node_data_dict[instance]

                # Generate plot
                plot_path = visualize_solution(
                    instance, method, best_solution, node_data, PLOTS_DIR
                )
                if plot_path is None:
                    logging.warning(
                        f"Plot not generated for {method} on {instance} due to invalid solution."
                    )

                # Export best solution to CSV
                csv_path = export_best_solution_to_csv(
                    instance, method, best_solution, node_data, RESULTS_EXPORT_DIR
                )
                if csv_path is None:
                    logging.warning(
                        f"CSV export not performed for {method} on {instance} due to invalid solution."
                    )
            else:
                logging.warning(
                    f"No valid best_solution available for {method} on {instance}, skipping plot and CSV export."
                )


if __name__ == "__main__":
    main()
