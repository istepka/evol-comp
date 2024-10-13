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


### Assignment 1
Greedy heuristics

Implement three methods:
- Random solution
- Nearest neighbor considering adding the node only at the end of the current path
- Nearest neighbor considering adding the node at all possible position, i.e. at the end, at the
beginning, or at any place inside the current path
- Greedy cycle adapted to our problem.

For each greedy method generate 200 solutions starting from each node. Generate also 200 random
solutions.
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
bool dumpToFile(const std::string &input_file, const std::string &algorithm_type, const std::vector<std::vector<int>> &solutions, const std::vector<int> &totalCosts, const std::string &sn_type)
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
    std::ofstream file("./results/" + algorithm_type + "_" + sn_type + '_' + input_file + ".solution");

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
    selectedNodes.push_back(selectedNodes[0]); // Close the cycle

    return selectedNodes;
}

// Nearest neighbor considering adding the node only at the end of the current path
std::vector<int> nearestNeighborEnd(const int startNode, const std::vector<std::vector<int>> &distanceMatrix, const std::vector<int> &costLookupTable, int k)
{
    int n = distanceMatrix.size();
    std::vector<int> path;
    std::vector<bool> visited(n, false);

    // Start from the startNode
    path.push_back(startNode);
    visited[startNode] = true;

    int current = startNode;

    while (path.size() < k)
    {
        int next = -1;
        int minEffectiveDistance = INT_MAX;

        for (int i = 0; i < n; i++)
        {
            if (!visited[i])
            {
                int dist = distanceMatrix[current][i];
                int effectiveDistance = dist + costLookupTable[i];
                if (effectiveDistance < minEffectiveDistance)
                {
                    minEffectiveDistance = effectiveDistance;
                    next = i;
                }
            }
        }

        if (next == -1)
        {
            break; // No more nodes to add
        }

        path.push_back(next);
        visited[next] = true;
        current = next;
    }

    // Close the cycle
    path.push_back(startNode);

    return path;
}

// Nearest neighbor considering adding the node at any position
std::vector<int> nearestNeighborAny(const int startNode, const std::vector<std::vector<int>> &distanceMatrix, const std::vector<int> &costLookupTable, int k)
{
    int n = distanceMatrix.size();
    std::vector<int> path;
    std::vector<bool> visited(n, false);

    // Start from the startNode
    path.push_back(startNode);
    visited[startNode] = true;

    // While path.size() < k
    while (path.size() < k)
    {
        int nearest = -1;
        int minEffectiveDistance = INT_MAX;

        // Find the nearest unvisited node to any node in the current path
        for (int node = 0; node < n; node++)
        {
            if (!visited[node])
            {
                for (int p_node : path)
                {
                    int dist = distanceMatrix[p_node][node];
                    int effectiveDistance = dist + costLookupTable[node];
                    if (effectiveDistance < minEffectiveDistance)
                    {
                        minEffectiveDistance = effectiveDistance;
                        nearest = node;
                    }
                }
            }
        }

        if (nearest == -1)
        {
            break; // No more nodes to add
        }

        // Find the best position to insert the nearest node
        int bestPos = -1;
        int minIncrease = INT_MAX;
        int pathSize = path.size();

        for (int i = 0; i < pathSize; i++)
        {
            int current = path[i];
            int next = path[(i + 1) % pathSize]; // wrap around
            int increase = distanceMatrix[current][nearest] + distanceMatrix[nearest][next] - distanceMatrix[current][next];
            if (increase < minIncrease)
            {
                minIncrease = increase;
                bestPos = i + 1;
            }
        }

        // Insert nearest at bestPos
        if (bestPos != -1)
        {
            path.insert(path.begin() + bestPos, nearest);
            visited[nearest] = true;
        }
        else
        {
            // Should not happen
            break;
        }
    }

    // Close the cycle
    path.push_back(path[0]);

    return path;
}

// Greedy cycle adapted to the problem
std::vector<int> greedyCycle(const int startNode, const std::vector<std::vector<int>> &distanceMatrix, const std::vector<int> &costLookupTable, int k)
{
    int n = distanceMatrix.size();
    std::vector<int> path;
    std::vector<bool> visited(n, false);

    // Start from the startNode
    path.push_back(startNode);
    visited[startNode] = true;

    while (path.size() < k)
    {
        int bestNode = -1;
        int bestScore = INT_MAX;

        // For each unselected node
        for (int node = 0; node < n; node++)
        {
            if (!visited[node])
            {
                // Find the minimum increase in distance for insertion
                int minIncrease = INT_MAX;

                for (size_t i = 0; i < path.size(); i++)
                {
                    int current = path[i];
                    int next = path[(i + 1) % path.size()];
                    int increase = distanceMatrix[current][node] + distanceMatrix[node][next] - distanceMatrix[current][next];
                    if (increase < minIncrease)
                    {
                        minIncrease = increase;
                    }
                }

                // Define score as minIncrease + cost of the node
                int score = minIncrease + costLookupTable[node];
                if (score < bestScore)
                {
                    bestScore = score;
                    bestNode = node;
                }
            }
        }

        if (bestNode == -1)
        {
            break; // No more nodes to add
        }

        // Insert the bestNode at the position that minimizes the increase
        int bestIncrease = INT_MAX;
        int bestPos = -1;

        for (size_t i = 0; i < path.size(); i++)
        {
            int current = path[i];
            int next = path[(i + 1) % path.size()];
            int increase = distanceMatrix[current][bestNode] + distanceMatrix[bestNode][next] - distanceMatrix[current][next];
            if (increase < bestIncrease)
            {
                bestIncrease = increase;
                bestPos = i + 1;
            }
        }

        if (bestPos != -1)
        {
            path.insert(path.begin() + bestPos, bestNode);
            visited[bestNode] = true;
        }
        else
        {
            // Append at the end if no better position found
            path.push_back(bestNode);
            visited[bestNode] = true;
        }
    }

    // Close the cycle
    path.push_back(path[0]);

    return path;
}

// Main with arguments
int main(int argc, char *argv[])
{
    if (argc < 4)
    {
        std::cerr << "Usage: " << argv[0] << " <input_file> <algorithm_type> <sn_type>" << std::endl;
        std::cerr << "Algorithm types: random, nn1, nn2, greedy" << std::endl;
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
    std::vector<std::vector<int>> solutions;
    std::vector<int> totalCosts;

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

        if (algorithm_type == "random")
            path = randomSolution(startNode, distanceMatrix, costLookupTable, k);
        else if (algorithm_type == "nn1")
            path = nearestNeighborEnd(startNode, distanceMatrix, costLookupTable, k);
        else if (algorithm_type == "nn2")
            path = nearestNeighborAny(startNode, distanceMatrix, costLookupTable, k);
        else if (algorithm_type == "greedy")
            path = greedyCycle(startNode, distanceMatrix, costLookupTable, k);
        else
        {
            std::cerr << "Invalid algorithm type!" << std::endl;
            return 1;
        }

        int totalCost = calculateTotalCost(path, distanceMatrix, costLookupTable);

        // ASSERT CORRECTNESS
        // Assert that the selected nodes are exactly k
        assert(path.size() == (k + 1));
        // Assert cycle
        assert(path.front() == path.back());
        // Assert start node is at the beginning if not greedy
        assert(path.front() == startNode || (algorithm_type == "greedy"));

        // Store the solution and its total cost
        solutions.push_back(path);
        totalCosts.push_back(totalCost);
    }

    dumpToFile(input_file, algorithm_type, solutions, totalCosts, sn_type);

    return 0;
}