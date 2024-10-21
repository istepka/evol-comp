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
    std::string filename = "./results/" + algorithm_type + "_" + sn_type + '_' + input_file + ".solution";
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
std::vector<int> greedyCycle(int startNode, const std::vector<std::vector<int>> &distanceMatrix, const std::vector<int> &costLookupTable, int k)
{
    int n = distanceMatrix.size();
    std::vector<int> path;
    std::vector<bool> visited(n, false);

    // Start from the startNode
    path.push_back(startNode);
    path.push_back(startNode);
    visited[startNode] = true;

    while (path.size() < k + 1)
    {
        // Go over all unvisited nodes and find the one that minimizes the increase in the objective function
        // While inserted at any position in the path
        int smallestIncrease = INT_MAX;
        int bestNode = -1;
        int bestPosition = -1;
        for (int i = 0; i < n; i++)
        {
            // If node i is in the path then skip
            if (visited[i] == true)
                continue;

            for (int j = 0; j < path.size() - 1; j++)
            {
                // check what happens if we insert here
                int left = path[j];
                int right = path[j + 1];
                int increase = distanceMatrix[left][i] + distanceMatrix[i][right] - distanceMatrix[left][right];

                // Add processing cost
                increase += costLookupTable[i];

                if (increase < smallestIncrease)
                {
                    smallestIncrease = increase;
                    bestNode = i;
                    bestPosition = j + 1; // +1 because we insert between current j and  j + 1
                }
            }
        }

        path.insert(path.begin() + bestPosition, bestNode);
        visited[bestNode] = true;
    }

    return path;
}

bool sortbyCond(const std::pair<int, int> &a,
                const std::pair<int, int> &b)
{
    if (a.first != b.first)
        return (a.first < b.first);
    else
        return (a.second > b.second);
}

std::vector<int> kRegretGreedyCycle(const int startNode, const std::vector<std::vector<int>> &distanceMatrix, const std::vector<int> &costLookupTable, int k)
{
    int n = distanceMatrix.size();
    std::vector<int> path;
    std::vector<bool> visited(n, false);

    // Start from the startNode
    path.push_back(startNode);
    visited[startNode] = true;

    while (path.size() < k)
    {
        // Go over all unvisited nodes and find the one that minimizes the increase in the objective function
        // While inserted at any position in the path
        int AsmallestIncrease[2] = {INT_MAX, INT_MAX};
        int AbestNode[2] = {-1, -1};
        // Select two nodes via normal greedy approach
        for (int i = 0; i < n; i++)
        {

            // If node i is in the path then skip
            if (visited[i] == true)
                continue;

            for (int j = 0; j < path.size(); j++)
            {

                int left = path[j];
                int right = path[(j + 1) % path.size()]; // make sure we calculate it as a cycle

                // check what happens if we insert here
                int increase_A_at_j = distanceMatrix[left][i] + distanceMatrix[i][right] - distanceMatrix[left][right];
                increase_A_at_j += costLookupTable[i];

                for (int k = 0; k < 2; k++)
                {
                    if (increase_A_at_j < AsmallestIncrease[k])
                    {
                        AsmallestIncrease[k] = increase_A_at_j;
                        AbestNode[k] = i;
                        break;
                    }
                }
            }
        }

        float bestRegret = -10000000;
        int bestNode = -1;
        int bestPosition = -1;

        for (int i = 0; i < 2; i++)
        {
            int node = AbestNode[i];
            std::vector<std::pair<float, int>> costs;

            for (int j = 0; j < path.size(); j++)
            {

                int left = path[j];
                int right = path[(j + 1) % path.size()]; // make sure we calculate it as a cycle

                // check what happens if we insert here
                int increase = distanceMatrix[left][node] + distanceMatrix[node][right] - distanceMatrix[left][right];
                increase += costLookupTable[node];

                costs.push_back(std::make_pair((float)increase, j));
            }

            // Sort a vector
            sort(costs.begin(), costs.end(), sortbyCond);

            float regret = costs[1].first - costs[0].first;

            if (regret > bestRegret)
            {
                bestRegret = regret;
                bestNode = node;
                bestPosition = costs[0].second + 1;
            }
        }

        // Insert it into the path
        path.insert(path.begin() + bestPosition, bestNode);
        visited[bestNode] = true;
    }

    path.push_back(startNode);
    return path;
}

std::vector<int> kRegretGreedyCycleWeighted(const int startNode,
                                            const std::vector<std::vector<int>> &distanceMatrix,
                                            const std::vector<int> &costLookupTable,
                                            int k,
                                            float lambdaObjective = 0.5,
                                            float lambdaOption1 = 0.5)
{
    int n = distanceMatrix.size();
    std::vector<int> path;
    std::vector<bool> visited(n, false);

    // Start from the startNode
    path.push_back(startNode);
    visited[startNode] = true;

    while (path.size() < k)
    {
        // Go over all unvisited nodes and find the one that minimizes the increase in the objective function
        // While inserted at any position in the path
        int AsmallestIncrease[2] = {INT_MAX, INT_MAX};
        int AbestNode[2] = {-1, -1};
        // Select two nodes via normal greedy approach
        for (int i = 0; i < n; i++)
        {

            // If node i is in the path then skip
            if (visited[i] == true)
                continue;

            for (int j = 0; j < path.size(); j++)
            {

                int left = path[j];
                int right = path[(j + 1) % path.size()]; // make sure we calculate it as a cycle

                // check what happens if we insert here
                int increase_A_at_j = distanceMatrix[left][i] + distanceMatrix[i][right] - distanceMatrix[left][right];
                increase_A_at_j += costLookupTable[i];

                for (int k = 0; k < 2; k++)
                {
                    if (increase_A_at_j < AsmallestIncrease[k])
                    {
                        AsmallestIncrease[k] = increase_A_at_j;
                        AbestNode[k] = i;
                        break;
                    }
                }
            }
        }

        // While inserted at any position in the path
        float bestScore = -100000000;
        int bestNode = -1;
        int bestPosition = -1;

        for (int i = 0; i < 2; i++)
        {
            int node = AbestNode[i];
            std::vector<std::pair<float, int>> costs;

            for (int j = 0; j < path.size(); j++)
            {

                int left = path[j];
                int right = path[(j + 1) % path.size()]; // make sure we calculate it as a cycle

                // check what happens if we insert here
                int increase = distanceMatrix[left][node] + distanceMatrix[node][right] - distanceMatrix[left][right];
                increase += costLookupTable[node];

                costs.push_back(std::make_pair((float)increase, j));
            }

            // Sort a vector
            sort(costs.begin(), costs.end(), sortbyCond);

            float regret = costs[1].first * (1 - lambdaOption1) - costs[0].first * lambdaOption1;

            float score = -costs[0].first * lambdaObjective + regret * (1 - lambdaObjective);

            if (score > bestScore)
            {
                bestScore = score;
                bestNode = node;
                bestPosition = costs[0].second + 1;
            }
        }

        // std::cout << bestNode << " " << visited[bestNode] << std::endl;
        // Insert it into the path
        path.insert(path.begin() + bestPosition, bestNode);
        visited[bestNode] = true;
    }

    path.push_back(startNode);
    return path;
}

// Main with arguments
int main(int argc, char *argv[])
{
    if (argc < 4)
    {
        std::cerr << "Usage: " << argv[0] << " <input_file> <algorithm_type> <sn_type>" << std::endl;
        std::cerr << "Algorithm types: random, nn1, nn2, greedy, kregret, kregret2" << std::endl;
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
        else if (algorithm_type == "kregret")
            path = kRegretGreedyCycle(startNode, distanceMatrix, costLookupTable, k);
        else if (algorithm_type == "kregret2")
            path = kRegretGreedyCycleWeighted(startNode, distanceMatrix, costLookupTable, k);
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