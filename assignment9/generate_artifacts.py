import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from pathlib import Path
import logging
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Configure Matplotlib for serif fonts and even larger font sizes
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 24,
    'axes.titlesize': 28,
    'axes.labelsize': 24,
    'legend.fontsize': 20,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20
})

def parse_output_text(text: str) -> dict:
    """Parse the output text into a structured format."""
    results = {}
    current_instance = None
    current_result = {}
    current_method = None
    print(text)
    
    lines = text.strip().split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith('Name:'):
            if current_instance and current_method and current_result:
                if current_instance not in results:
                    results[current_instance] = {}
                results[current_instance][current_method] = current_result.copy()
            current_method = line.split(':')[1].strip()
            current_result = {'method': current_method}
        elif line.startswith('Problem instance:'):
            current_instance = line.split(':')[1].strip()
        elif line.startswith('Best cost:'):
            current_result['best_cost'] = float(line.split(':')[1].strip())
        elif line.startswith('Worst cost:'):
            current_result['worst_cost'] = float(line.split(':')[1].strip())
        elif line.startswith('Average cost:'):
            current_result['average_cost'] = float(line.split(':')[1].strip())
        elif line.startswith('Average time:'):
            current_result['average_time'] = float(line.split(':')[1].strip())
        elif line.startswith('Best time:'):
            current_result['best_time'] = float(line.split(':')[1].strip())
        elif line.startswith('Worst time:'):
            current_result['worst_time'] = float(line.split(':')[1].strip())
        elif line.startswith('Best solution:'):
            current_result['best_solution'] = [int(x) for x in line.split(':')[1].strip().split()]
        elif line.startswith('Average iterations number:'):
            current_result['avg_iterations'] = float(line.split(':')[1].strip())
    
    # Add the last instance/method
    if current_instance and current_method and current_result:
        if current_instance not in results:
            results[current_instance] = {}
        results[current_instance][current_method] = current_result
    
    return results

def read_node_data(instance: str) -> pd.DataFrame:
    """Read node data from CSV file."""
    try:
        file_path = f'../data/{instance}.csv'
        df = pd.read_csv(file_path, header=None, names=['x', 'y', 'cost'], sep=';')
        logging.info(f"Loaded data for instance {instance} with {len(df)} nodes.")
        return df
    except Exception as e:
        logging.error(f"Error reading data for instance {instance}: {e}")
        return None

def visualize_solution(instance: str, method: str, solution: list, node_data: pd.DataFrame, output_dir: Path) -> Optional[Path]:
    """Visualize solution with original plotting style."""
    # Validate the solution list
    num_nodes = len(node_data)
    if not solution:
        logging.warning(f"Empty best_solution for {method} on {instance}. Skipping plot.")
        return None

    if any(not isinstance(idx, int) for idx in solution):
        logging.warning(f"Non-integer node index found in best_solution for {method} on {instance}. Skipping plot.")
        return None

    if any(idx < 0 or idx >= num_nodes for idx in solution):
        logging.warning(f"Node index out of bounds in best_solution for {method} on {instance}. Skipping plot.")
        return None

    # Check if the solution forms a cycle (starts and ends with the same node)
    if solution[0] != solution[-1]:
        logging.warning(f"Solution for {method} on {instance} does not form a cycle. Appending the starting node to complete the cycle.")
        solution = solution + [solution[0]]

    # Remove duplicate last node if already appended
    if len(solution) >= 2 and solution[-1] == solution[-2]:
        logging.info(f"Cycle already complete for {method} on {instance}. Removing duplicate node.")
        solution = solution[:-1]

    try:
        # Proceed with plotting
        plt.figure(figsize=(12, 8))
        selected_nodes = node_data.iloc[solution].reset_index(drop=True)
        all_nodes = node_data.reset_index(drop=True)

        # Check for NaN or invalid data
        if selected_nodes[['x', 'y', 'cost']].isnull().any().any():
            logging.error(f"Selected nodes contain NaN values for {method} on {instance}. Skipping plot.")
            return None

        # Plot selected nodes with color representing cost
        scatter = plt.scatter(
            selected_nodes['x'],
            selected_nodes['y'],
            c=selected_nodes['cost'],
            cmap='viridis',
            label='Selected Nodes',
            edgecolors='red',
            linewidths=2.5,
            s=150
        )
        plt.scatter(
            all_nodes['x'],
            all_nodes['y'],
            c=all_nodes['cost'],
            cmap='viridis',
            label='Unselected Nodes',
            s=150
        )

        # Add colorbar
        plt.colorbar(scatter, label='Node Cost')

        # Draw the Hamiltonian cycle
        plt.plot(selected_nodes['x'], selected_nodes['y'],
                 'r-', linewidth=1.5, label='Hamiltonian Cycle')

        plt.title(f'Best Solution for {method} on {instance}')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.xticks([0, 1000, 2000, 3000, 4000])
        plt.legend(ncol=3, bbox_to_anchor=(0.5, -0.15), loc='upper center')

        # Adjust layout
        plt.tight_layout()

        # Save the plot
        plot_filename = f'Best_{method}_{instance}.pdf'
        plot_path = output_dir / plot_filename
        plt.savefig(plot_path, format='pdf', bbox_inches='tight')
        logging.info(f"Plot saved at {plot_path}")
        plt.close()
        return plot_path

    except Exception as e:
        logging.error(f"Failed to generate plot for {method} on {instance}: {e}")
        plt.close()
        return None

def export_best_solution_to_csv(instance: str, method: str, solution: list, node_data: pd.DataFrame, output_dir: Path) -> Optional[Path]:
    """Export the best solution cycle to a CSV file."""
    # Validate the solution list
    num_nodes = len(node_data)
    if not solution:
        logging.warning(f"Empty best_solution for {method} on {instance}. Skipping CSV export.")
        return None

    if any(not isinstance(idx, int) for idx in solution):
        logging.warning(f"Non-integer node index found in best_solution for {method} on {instance}. Skipping CSV export.")
        return None

    if any(idx < 0 or idx >= num_nodes for idx in solution):
        logging.warning(f"Node index out of bounds in best_solution for {method} on {instance}. Skipping CSV export.")
        return None

    # Check if the solution forms a cycle (starts and ends with the same node)
    if solution[0] != solution[-1]:
        logging.warning(f"Solution for {method} on {instance} does not form a cycle. Appending the starting node to complete the cycle.")
        solution = solution + [solution[0]]

    # Remove duplicate last node if already appended
    if len(solution) >= 2 and solution[-1] == solution[-2]:
        logging.info(f"Cycle already complete for {method} on {instance}. Removing duplicate node.")
        solution = solution[:-1]

    try:
        # Extract node details in the order of the solution
        selected_nodes = node_data.iloc[solution].reset_index(drop=True)
        selected_nodes.insert(0, 'node_idx', solution)

        # Check for NaN or invalid data
        if selected_nodes[['node_idx', 'x', 'y', 'cost']].isnull().any().any():
            logging.error(f"Selected nodes contain NaN values for {method} on {instance}. Skipping CSV export.")
            return None

        # Define the output filename and ensure directory exists
        csv_filename = f"{instance}_best_{method}_checker_format.csv"
        csv_path = output_dir / csv_filename
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save to CSV
        selected_nodes.to_csv(csv_path, columns=['node_idx', 'x', 'y', 'cost'], index=False)
        logging.info(f"Best solution CSV exported at {csv_path}")
        return csv_path

    except Exception as e:
        logging.error(f"Failed to export best solution CSV for {method} on {instance}: {e}")
        return None

def main():
    # read in the output text from the new file
    with open('sol.txt', 'r') as f:
        output_text = f.read()
    
    # Create output directories
    plots_dir = Path('plots')
    results_dir = Path('results')
    plots_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse results
    results = parse_output_text(output_text)
    
    # Process each instance and method
    for instance, methods in results.items():
        # Read node data
        node_data = read_node_data(instance)
        if node_data is None:
            continue
            
        for method, result in methods.items():
            # Generate visualization
            plot_path = visualize_solution(
                instance,
                method,
                result['best_solution'],
                node_data,
                plots_dir
            )
            
            # Export solution to CSV
            csv_path = export_best_solution_to_csv(
                instance,
                method,
                result['best_solution'],
                node_data,
                results_dir
            )
            
            # Print summary
            print(f"\nResults for {instance} - {method}:")
            print(f"Best cost: {result['best_cost']}")
            print(f"Average cost: {result['average_cost']}")
            print(f"Worst cost: {result['worst_cost']}")
            print(f"Average time: {result['average_time']}")
            print(f"Best time: {result.get('best_time', 'N/A')}")
            print(f"Worst time: {result.get('worst_time', 'N/A')}")
            print(f"Average iterations: {result.get('avg_iterations', 'N/A')}")

if __name__ == '__main__':
    main()