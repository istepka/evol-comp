#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <tuple>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <numeric>
#include <cassert>
#include <string>
#include <unordered_set>
#include <climits>

/*
## Problem description
We are given three columns of integers with a row for each node. The first two columns contain x
and y coordinates of the node positions in a plane. The third column contains node costs. The goal is
to select exactly 50% of the nodes (if the number of nodes is odd we round the number of nodes to
be selected up) and form a Hamiltonian cycle (closed path) through this set of nodes such that the
sum of the total length of the path plus the total cost of the selected nodes is minimized.
The distances between nodes are calculated as Euclidean distances rounded mathematically to
integer values. The distance matrix should be calculated just after reading an instance and then only
the distance matrix (no nodes coordinates) should be accessed by optimization methods to allow
instances defined only by distance matrices.

TO DO:
Greedy regret heuristics
Implement two methods based on greedy cycle heuristic:
 Greedy 2-regret heuristics.
 Greedy heuristics with a weighted sum criterion – 2-regret + best change of the objective
function. By default use equal weights but you can also experiment with other values. The structure of the report should be the same as previously. Please include in the summary of
results please include results of the previous methods.
*/

// Function to read the CSV file and store data in a vector of tuples (x, y, cost)
std::vector<std::tuple<int, int, int>> readCSV(const std::string &filename)
{
    std::ifstream file(filename);
    std::vector<std::tuple<int, int, int>> data;

    if (!file.is_open())
    {
        std::cerr << "Error opening file!" << std::endl;
        return data; // Return an empty vector if the file can't be opened
    }

    std::string line;
    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        std::string item;
        std::vector<int> row;

        // Split the line by ';' and convert to integers
        while (std::getline(ss, item, ';'))
        {
            row.push_back(std::stoi(item));
        }

        // Ensure there are exactly 3 elements in the row
        if (row.size() == 3)
        {
            data.push_back(std::make_tuple(row[0], row[1], row[2]));
        }
    }

    file.close();
    return data;
}

// Dump to file the average, worst, best of the total costs
bool dumpToFile(const std::string &input_file,
                const std::string &algorithm_type,
                const std::vector<std::vector<int>> &solutions,
                const std::vector<int> &totalCosts,
                const std::string &sn_type,
                std::string additionalString = "")
{
    // Get average, worst, best of the total costs
    float average = std::accumulate(totalCosts.begin(), totalCosts.end(), 0) / static_cast<float>(totalCosts.size());

    int worst_index = std::distance(totalCosts.begin(), std::max_element(totalCosts.begin(), totalCosts.end()));
    int worst_cost = totalCosts[worst_index];
    std::vector<int> worst_path = solutions[worst_index];

    int best_index = std::distance(totalCosts.begin(), std::min_element(totalCosts.begin(), totalCosts.end()));
    int best_cost = totalCosts[best_index];
    std::vector<int> best_path = solutions[best_index];

    std::cout << "Average cost: " << average << std::endl;
    std::cout << "Worst cost: " << worst_cost << std::endl;
    std::cout << "Best cost: " << best_cost << std::endl;

    // create results directory if it doesn't exist
    std::string dir = "./results";
    std::string command = "mkdir -p " + dir;
    system(command.c_str());
    // Save average, worst, best to file
    additionalString = "_" + additionalString;
    std::string filename = "./results/" + algorithm_type + "_" + sn_type + '_' + input_file + additionalString + ".solution";
    std::ofstream file(filename.c_str());

    file << "Average cost: " << average << std::endl;

    file << "Worst cost: " << worst_cost << " ; ";
    for (int i = 0; i < worst_path.size(); i++)
    {
        file << worst_path[i] << " ";
    }
    file << std::endl;

    file << "Best cost: " << best_cost << " ; ";
    for (int i = 0; i < best_path.size(); i++)
    {
        file << best_path[i] << " ";
    }
    file << std::endl;

    file.flush();
    file.close();

    return true;
}

// Get the (euclidean) distance matrix from the data
std::vector<std::vector<int>> getDistanceMatrix(const std::vector<std::tuple<int, int, int>> &data)
{
    int n = data.size();
    std::vector<std::vector<int>> distanceMatrix(n, std::vector<int>(n, 0));

    for (int i = 0; i < n; i++)
    {
        int x1 = std::get<0>(data[i]);
        int y1 = std::get<1>(data[i]);
        for (int j = i + 1; j < n; j++)
        {
            int x2 = std::get<0>(data[j]);
            int y2 = std::get<1>(data[j]);

            int dx = x2 - x1;
            int dy = y2 - y1;

            // Calculate the Euclidean distance
            double dist = std::sqrt(dx * dx + dy * dy);
            // Round the distance to the nearest integer
            int roundedDist = static_cast<int>(std::round(dist));

            distanceMatrix[i][j] = roundedDist;
            distanceMatrix[j][i] = roundedDist;
        }
    }

    return distanceMatrix;
}

// Calculate total cost of a path (distances + cost of nodes)
int calculateTotalCost(const std::vector<int> &path, const std::vector<std::vector<int>> &distanceMatrix, const std::vector<int> &costLookupTable)
{
    int totalCost = 0;
    int pathSize = path.size();

    // Sum distances between consecutive nodes
    for (int i = 0; i < pathSize - 1; i++)
    {
        totalCost += distanceMatrix[path[i]][path[i + 1]];
    }

    // Sum node costs (excluding duplicate nodes if any)
    std::unordered_set<int> uniqueNodes(path.begin(), path.end() - 1);
    for (int node : uniqueNodes)
    {
        totalCost += costLookupTable[node];
    }

    return totalCost;
}

// Random solution
std::vector<int> randomSolution(const int startNode, const std::vector<std::vector<int>> &distanceMatrix, const std::vector<int> &costLookupTable, int k)
{
    int n = distanceMatrix.size();
    std::vector<int> nodes(n);
    for (int i = 0; i < n; i++)
        nodes[i] = i;

    std::random_shuffle(nodes.begin(), nodes.end());

    // Make sure the start node is at the beginning of the nodes list
    auto it = std::find(nodes.begin(), nodes.end(), startNode);
    if (it != nodes.end())
    {
        std::swap(nodes[0], *it);
    }
    else
    {
        nodes.insert(nodes.begin(), startNode);
    }

    // Select exactly k nodes
    std::vector<int> selectedNodes(nodes.begin(), nodes.begin() + k);
    // selectedNodes.push_back(selectedNodes[0]); // Close the cycle

    return selectedNodes;
}

int safeIndex(int index, int size) {
    return (index % size + size) % size; // Ensures positive index within bounds
}

void cmls(std::vector<int> &solution, const std::vector<std::vector<int>> &distanceMatrix, const std::vector<int> &costLookupTable)
{
    int solutionSize = solution.size();
    // std::cout << "Solution size: " << solutionSize << std::endl;
    // std::cout << "Starting cost: " << calculateTotalCost(solution, distanceMatrix, costLookupTable) << std::endl;
    int lastBestDelta = 0;
    int iteration = 0;

    while (true) {
        iteration++;

        int bestDelta = -INT32_MAX;
        std::pair<int, int> bestPair;

        for (int i = 0; i < solutionSize; i++) {
            int current = solution[i];

            // Find 10 nearest neighbors by looking up the distance matrix, additonally add the cost of the node and sort them
            std::vector<std::pair<int, float>> nearestNeighbors;

            for (int j = 0; j < distanceMatrix.size(); j++) {
                if (std::find(solution.begin(), solution.end(), j) != solution.end()) {
                    continue; // Node already in the path
                }

                nearestNeighbors.push_back(std::make_pair(j, distanceMatrix[current][j] + costLookupTable[j]));
            }

            // Sort the nearest neighbors
            std::sort(nearestNeighbors.begin(), nearestNeighbors.end(), [](const std::pair<int, float> &a, const std::pair<int, float> &b) {
                return a.second < b.second;
            });


            for (int j = 0; j < 10; j++) {
                for (int k = -1; k <= 1; k+=2) { // k acts as a multiplier to get the next and next next node both ways

                    int candidate = nearestNeighbors[j].first; // Get the candidate node
                    
                    // TODO: Both sides
                    int ip1_node = solution[safeIndex(i + k, solutionSize)]; // Current +- 1
                    int ip2_node = solution[safeIndex(i + k*2, solutionSize)]; // Current +- 2

                    int newc = distanceMatrix[current][candidate] + distanceMatrix[candidate][ip2_node] + costLookupTable[candidate];
                    int oldc = distanceMatrix[current][ip1_node] + distanceMatrix[ip1_node][ip2_node] + costLookupTable[ip1_node];
                    int delta = oldc - newc; // The lowest the better

                    if (delta > bestDelta) {
                        bestDelta = delta;
                        bestPair = std::make_pair(ip1_node, candidate); // Exchange ip1 and c to add proper candidate edge
                    }
                }

            }
        }

        if (bestDelta <= 0){
            std::cout<<"Best delta is "<<bestDelta<<" so there is no point in going further"<<std::endl;
            break;
        }

        // Insert the candidate edge at ip2
        int swap_location = std::find(solution.begin(), solution.end(), bestPair.first) - solution.begin();
        solution[swap_location] = bestPair.second;


        // if (iteration % 1 == 0) {
        //     std::cout << "Iteration: " << iteration << " Best delta: " << bestDelta << std::endl;
        //     std::cout << "Inserting " << bestPair.second << " at " << swap_location << " to replace " << bestPair.first << std::endl;
        //     int tc = calculateTotalCost(solution, distanceMatrix, costLookupTable);
        //     std::cout << "Total cost after insertion: " << tc << std::endl;
        //     for (auto x : solution) {
        //         std::cout << x << " ";
        //     }
        //     std::cout << std::endl;

        //     // if (iteration % 10 == 0)
        //     //     exit(0);
        // }
    }
}



// Main with arguments
int main(int argc, char *argv[])
{
    if (argc < 4)
    {
        std::cerr << "Usage: " << argv[0] << " <input_file> <algorithm_type> <sn_type>" << std::endl;
        std::cerr << "Algorithm types: random, nn1, nn2, greedy, kregret, w_kregret, kregret_mod, w_kregret_mod, cmls" << std::endl;
        std::cerr << "Starting node types: random, each" << std::endl;
        std::cerr << "Example: " << argv[0] << " input.csv random random" << std::endl;
        return 1;
    }

    std::string input_file = argv[1];
    std::string algorithm_type = argv[2];
    std::string sn_type = argv[3];

    // Read data from the file
    std::vector<std::tuple<int, int, int>> data = readCSV("../data/" + input_file);
    int elements = data.size();
    int k = std::ceil(elements / 2.0);

    // Get the distance matrix
    std::vector<std::vector<int>> distanceMatrix = getDistanceMatrix(data);

    // Get the cost lookup table
    std::vector<int> costLookupTable;
    for (auto &tuple : data)
    {
        costLookupTable.push_back(std::get<2>(tuple));
    }

    // Print the number of nodes
    std::cout << "Number of nodes: " << elements << std::endl;
    std::cout << std::string(80, '-') << std::endl;

    // Generate 200 solutions
    int nSolutions = 200;
    std::vector<std::vector<int>> solutions(nSolutions);
    std::vector<int> totalCosts(nSolutions);

    for (int i = 0; i < nSolutions; i++)
    {
        int startNode;
        if (sn_type == "random")
            startNode = std::rand() % elements;
        else if (sn_type == "each")
            startNode = i % elements;
        else
        {
            std::cerr << "Invalid Starting node type!" << std::endl;
            return 1;
        }

        std::vector<int> path;

        path = randomSolution(startNode, distanceMatrix, costLookupTable, k);

        if (algorithm_type == "cmls")
            cmls(path, distanceMatrix, costLookupTable);
        else  
            std::cerr << "Invalid algorithm type!" << std::endl;

        // Add the start node to the end of the path to close the cycle
        if (path.back() != path.front())
        {
            path.push_back(path.front());
        }

        // Print the path
        for (int node : path)
        {
            std::cout << node << " ";
        }
        std::cout << std::endl;

        int totalCost = calculateTotalCost(path, distanceMatrix, costLookupTable);

        std::cout << "Total cost: " << totalCost << std::endl;

        // ASSERT CORRECTNESS
        // Assert that the selected nodes are exactly k
        assert(path.size() == (k + 1));
        // Assert cycle
        assert(path.front() == path.back());

        // Store the solution and its total cost
        solutions[i] = path;
        totalCosts[i] = totalCost;

        // Clean up the path
        path.clear();
    }

    dumpToFile(input_file, algorithm_type, solutions, totalCosts, sn_type);

    return 0;
}