import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Constants
DATA_DIR = Path('../data')
RESULTS_DIR = Path('results')
# New directory for exported CSVs
RESULTS_EXPORT_DIR = Path('exported_results')
PLOTS_DIR = Path('plots')
LATEX_TABLE_FILE = 'results_table.tex'

# Configure Matplotlib for serif fonts and even larger font sizes
plt.rcParams.update({
    # 'text.usetex': True,
    'font.family': 'serif',
    'font.size': 24,
    'axes.titlesize': 28,
    'axes.labelsize': 24,
    'legend.fontsize': 20,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20
})


def compute_nodes_to_select(total_nodes: int) -> int:
    return math.ceil(total_nodes / 2)

# Read node data


def read_node_data(instance: str) -> pd.DataFrame:
    file_path = DATA_DIR / f'{instance}.csv'
    if not file_path.exists():
        raise FileNotFoundError(
            f"Data file for instance {instance} not found at {file_path}")
    df = pd.read_csv(file_path, header=None, names=[
                     'x', 'y', 'cost'], sep=';')
    logging.info(f"Loaded data for instance {instance} with {len(df)} nodes.")
    return df

# Parse the output.txt file


def parse_output_file(file_path: Path) -> Dict[str, Dict[str, Dict[str, Any]]]:
    all_results = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()
    current_method = None
    current_instance = None
    result_dict = {}
    for line in lines:
        line = line.strip()
        if line.startswith('Name:'):
            if result_dict and current_method and current_instance:
                # Store the previous result_dict
                if current_instance not in all_results:
                    all_results[current_instance] = {}
                all_results[current_instance][current_method] = result_dict
                result_dict = {}
            current_method = line[len('Name:'):].strip()
        elif line.startswith('Problem:'):
            current_instance = line[len('Problem:'):].strip()
        elif line.startswith('Best cost:'):
            result_dict['best_cost'] = float(
                line[len('Best cost:'):].strip())
        elif line.startswith('Worst cost:'):
            result_dict['worst_cost'] = float(
                line[len('Worst cost:'):].strip())
        elif line.startswith('Average cost:'):
            result_dict['average_cost'] = float(
                line[len('Average cost:'):].strip())
        elif line.startswith('Best time:'):
            result_dict['best_time'] = float(
                line[len('Best time:'):].strip())
        elif line.startswith('Worst time:'):
            result_dict['worst_time'] = float(
                line[len('Worst time:'):].strip())
        elif line.startswith('Average time:'):
            result_dict['average_time'] = float(
                line[len('Average time:'):].strip())
        elif line.startswith('Best solution:'):
            solution_str = line[len('Best solution:'):].strip()
            result_dict['best_solution'] = [
                int(s) for s in solution_str.split()]
    # After the last block
    if result_dict and current_method and current_instance:
        if current_instance not in all_results:
            all_results[current_instance] = {}
        all_results[current_instance][current_method] = result_dict
    return all_results

# Generate LaTeX tables


def generate_latex_tables(all_results: Dict[str, Dict[str, Dict[str, Any]]], instances: List[str], methods: List[str], output_file: Path):
    with open(output_file, 'w') as f:
        f.write('% LaTeX Table of Results\n')
        f.write('\\begin{table}[ht]\n\\centering\n')
        f.write('\\caption{Computational Experiment Results}\n')
        f.write('\\label{tab:results}\n')
        # Instance, Method, Best Cost, Worst Cost, Average Cost
        f.write('\\begin{tabular}{lcccc}\n')
        f.write('\\hline\n')
        f.write('Instance & Method & Best Cost & Worst Cost & Average Cost \\\\\n')
        f.write('\\hline\n')

        for instance in instances:
            for method in methods:
                res = all_results[instance][method]
                if res:
                    best_cost = f"{res['best_cost']:.2f}" if res.get(
                        'best_cost') is not None else "N/A"
                    worst_cost = f"{res['worst_cost']:.2f}" if res.get(
                        'worst_cost') is not None else "N/A"
                    average_cost = f"{res['average_cost']:.2f}" if res.get(
                        'average_cost') is not None else "N/A"
                    f.write(
                        f"{instance} & {method} & {best_cost} & {worst_cost} & {average_cost} \\\\\n")
                else:
                    f.write(f"{instance} & {method} & N/A & N/A & N/A \\\\\n")
        f.write('\\hline\n')
        f.write('\\end{tabular}\n')
        f.write('\\end{table}\n')
    logging.info(f"LaTeX table generated at {output_file}")

# Generate 2D visualization plots


def visualize_solution(instance: str, method: str, solution: List[int], node_data: pd.DataFrame, output_dir: Path) -> Optional[Path]:
    # Validate the solution list
    num_nodes = len(node_data)
    if not solution:
        logging.warning(
            f"Empty best_solution for {method} on {instance}. Skipping plot.")
        return None

    if any(not isinstance(idx, int) for idx in solution):
        logging.warning(
            f"Non-integer node index found in best_solution for {method} on {instance}. Skipping plot.")
        return None

    if any(idx < 0 or idx >= num_nodes for idx in solution):
        logging.warning(
            f"Node index out of bounds in best_solution for {method} on {instance}. Skipping plot.")
        return None

    # Check if the solution forms a cycle (starts and ends with the same node)
    if solution[0] != solution[-1]:
        logging.warning(
            f"Solution for {method} on {instance} does not form a cycle. Appending the starting node to complete the cycle.")
        solution = solution + [solution[0]]

    # Remove duplicate last node if already appended
    if len(solution) >= 2 and solution[-1] == solution[-2]:
        logging.info(
            f"Cycle already complete for {method} on {instance}. Removing duplicate node.")
        solution = solution[:-1]

    try:
        # Proceed with plotting
        plt.figure(figsize=(12, 8))
        selected_nodes = node_data.iloc[solution].reset_index(drop=True)
        all_nodes = node_data.reset_index(drop=True)

        # Check for NaN or invalid data
        if selected_nodes[['x', 'y', 'cost']].isnull().any().any():
            logging.error(
                f"Selected nodes contain NaN values for {method} on {instance}. Skipping plot.")
            return None

        logging.debug(
            f"Selected nodes for {method} on {instance}:\n{selected_nodes.head()}")

        # Plot selected nodes with color representing cost
        try:
            scatter = plt.scatter(
                selected_nodes['x'],
                selected_nodes['y'],
                c=selected_nodes['cost'],
                cmap='viridis',
                label='Selected Nodes',
                edgecolors='red',  # Use 'edgecolors' instead of 'edgecolor'
                linewidths=2.5,
                s=150
            )
            plt.scatter(
                all_nodes['x'],
                all_nodes['y'],
                c=all_nodes['cost'],
                cmap='viridis',
                label='Unselected Nodes',
                # edgecolors='k',  # Use 'edgecolors' instead of 'edgecolor'
                # linewidths=0.5,
                s=150
            )
        except Exception as e:
            logging.error(
                f"Error in scatter plot for {method} on {instance}: {e}")
            return None

        # Add colorbar
        try:
            plt.colorbar(scatter, label='Node Cost')
        except Exception as e:
            logging.error(
                f"Error adding colorbar for {method} on {instance}: {e}")

        # Draw the Hamiltonian cycle
        plt.plot(selected_nodes['x'], selected_nodes['y'],
                 'r-', linewidth=1.5, label='Hamiltonian Cycle')

        plt.title(f'Best Solution for {method} on {instance}')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.xticks([0, 1000, 2000, 3000, 4000])
        plt.legend(ncol=3, bbox_to_anchor=(0.5, -0.15), loc='upper center')

        # Adjust layout
        try:
            plt.tight_layout()
        except Exception as e:
            logging.error(
                f"Error during tight_layout for {method} on {instance}: {e}")

        # Save the plot as PDF
        plot_filename = f'Best_{method}_{instance}.pdf'
        plot_path = output_dir / plot_filename
        logging.warning(plot_path)
        try:
            plt.savefig(plot_path, format='pdf')
            logging.info(f"Plot saved at {plot_path}")
        except Exception as e:
            logging.error(f"Error saving plot for {method} on {instance}: {e}")
        finally:
            plt.close()
        return plot_path

    except Exception as e:
        logging.error(
            f"Failed to generate plot for {method} on {instance}: {e}")
        plt.close()
        return None

# Export Best Solution to CSV


def export_best_solution_to_csv(instance: str, method: str, solution: List[int], node_data: pd.DataFrame, output_dir: Path) -> Optional[Path]:
    """
    Exports the best solution cycle to a CSV file with columns: node_idx, x, y, cost.
    Each row corresponds to the next node in the cycle.
    """
    # Validate the solution list
    num_nodes = len(node_data)
    if not solution:
        logging.warning(
            f"Empty best_solution for {method} on {instance}. Skipping CSV export.")
        return None

    if any(not isinstance(idx, int) for idx in solution):
        logging.warning(
            f"Non-integer node index found in best_solution for {method} on {instance}. Skipping CSV export.")
        return None

    if any(idx < 0 or idx >= num_nodes for idx in solution):
        logging.warning(
            f"Node index out of bounds in best_solution for {method} on {instance}. Skipping CSV export.")
        return None

    # Check if the solution forms a cycle (starts and ends with the same node)
    if solution[0] != solution[-1]:
        logging.warning(
            f"Solution for {method} on {instance} does not form a cycle. Appending the starting node to complete the cycle.")
        solution = solution + [solution[0]]

    # Remove duplicate last node if already appended
    if len(solution) >= 2 and solution[-1] == solution[-2]:
        logging.info(
            f"Cycle already complete for {method} on {instance}. Removing duplicate node.")
        solution = solution[:-1]

    try:
        # Extract node details in the order of the solution
        selected_nodes = node_data.iloc[solution].reset_index(drop=True)
        selected_nodes.insert(0, 'node_idx', solution)

        # Check for NaN or invalid data
        if selected_nodes[['node_idx', 'x', 'y', 'cost']].isnull().any().any():
            logging.error(
                f"Selected nodes contain NaN values for {method} on {instance}. Skipping CSV export.")
            return None

        # Define the output filename
        csv_filename = f"{instance}_best_{method}_checker_format.csv"
        csv_path = output_dir / csv_filename

        # Ensure the output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save to CSV
        selected_nodes.to_csv(
            csv_path, columns=['node_idx', 'x', 'y', 'cost'], index=False)
        logging.info(f"Best solution CSV exported at {csv_path}")
        return csv_path

    except Exception as e:
        logging.error(
            f"Failed to export best solution CSV for {method} on {instance}: {e}")
        return None


def main():
    # Step 1: Parse output.txt and collect all results
    output_file = Path('output.txt')
    if not output_file.exists():
        logging.error(f"Output file {output_file} does not exist.")
        return
    all_results = parse_output_file(output_file)

    # Get list of instances and methods
    INSTANCES = list(all_results.keys())
    METHODS = set()
    for instance in all_results:
        METHODS.update(all_results[instance].keys())
    METHODS = list(METHODS)

    # Step 2: Read node data
    node_data_dict = {}
    for instance in INSTANCES:
        try:
            node_data = read_node_data(instance)
            node_data_dict[instance] = node_data
        except FileNotFoundError as e:
            logging.error(e)
            return

    # Step 3: Generate LaTeX tables
    generate_latex_tables(all_results.copy(), INSTANCES,
                          METHODS, Path(LATEX_TABLE_FILE))

    # Step 4: Generate plots and export best solutions to CSV
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    # Ensure export directory exists
    RESULTS_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    for instance in INSTANCES:
        for method in METHODS:
            res = all_results[instance].get(method, {})
            if res and 'best_solution' in res and res['best_solution']:
                best_solution = res['best_solution']
                node_data = node_data_dict[instance]

                # Generate plot
                plot_path = visualize_solution(
                    instance, method, best_solution, node_data, PLOTS_DIR)
                if plot_path is None:
                    logging.warning(
                        f"Plot not generated for {method} on {instance} due to invalid solution.")

                # Export best solution to CSV
                csv_path = export_best_solution_to_csv(
                    instance, method, best_solution, node_data, RESULTS_EXPORT_DIR)
                if csv_path is None:
                    logging.warning(
                        f"CSV export not performed for {method} on {instance} due to invalid solution.")
            else:
                logging.warning(
                    f"No valid best_solution available for {method} on {instance}, skipping plot and CSV export.")


if __name__ == '__main__':
    main()