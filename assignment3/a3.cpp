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
#include <random>
#include <chrono>
#include <iomanip>

// Struct to hold node data
struct Node {
    int x;
    int y;
    int cost;
};

// Function to read the CSV file and store data in a vector of Nodes
std::vector<Node> readCSV(const std::string &filename) {
    std::ifstream file(filename);
    std::vector<Node> data;

    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return data; // Return an empty vector if the file can't be opened
    }

    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string item;
        std::vector<int> row;

        // Split the line by ';' and convert to integers
        while (std::getline(ss, item, ';')) {
            row.push_back(std::stoi(item));
        }

        // Ensure there are exactly 3 elements in the row
        if (row.size() == 3) {
            Node node = {row[0], row[1], row[2]};
            data.push_back(node);
        }
    }

    file.close();
    return data;
}

// Function to calculate the Euclidean distance matrix
std::vector<std::vector<int>> getDistanceMatrix(const std::vector<Node> &data) {
    int n = data.size();
    std::vector<std::vector<int>> distanceMatrix(n, std::vector<int>(n, 0));

    for (int i = 0; i < n; i++) {
        int x1 = data[i].x;
        int y1 = data[i].y;
        for (int j = i + 1; j < n; j++) {
            int x2 = data[j].x;
            int y2 = data[j].y;

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

// Function to calculate the total cost of a path (distances + cost of nodes)
int calculateTotalCost(const std::vector<int> &path, const std::vector<std::vector<int>> &distanceMatrix, const std::vector<int> &costLookupTable) {
    int totalCost = 0;
    int pathSize = path.size();

    // Sum distances between consecutive nodes
    for (int i = 0; i < pathSize - 1; i++) {
        totalCost += distanceMatrix[path[i]][path[i + 1]];
    }

    // Sum node costs (excluding duplicate nodes if any)
    std::unordered_set<int> uniqueNodes(path.begin(), path.end() - 1);
    for (int node : uniqueNodes) {
        totalCost += costLookupTable[node];
    }

    return totalCost;
}

// Function to dump results to a file
bool dumpToFile(const std::string &input_file,
                const std::string &method_type,
                const std::vector<std::vector<int>> &solutions,
                const std::vector<int> &totalCosts) {
    // Get average, worst, best of the total costs
    float average = std::accumulate(totalCosts.begin(), totalCosts.end(), 0.0f) / static_cast<float>(totalCosts.size());

    int worst_index = std::distance(totalCosts.begin(), std::max_element(totalCosts.begin(), totalCosts.end()));
    int worst_cost = totalCosts[worst_index];
    std::vector<int> worst_path = solutions[worst_index];

    int best_index = std::distance(totalCosts.begin(), std::min_element(totalCosts.begin(), totalCosts.end()));
    int best_cost = totalCosts[best_index];
    std::vector<int> best_path = solutions[best_index];

    std::cout << "Average cost: " << average << std::endl;
    std::cout << "Worst cost: " << worst_cost << std::endl;
    std::cout << "Best cost: " << best_cost << std::endl;

    // Create results directory if it doesn't exist
    std::string dir = "./results";
    std::string command = "mkdir -p " + dir;
    system(command.c_str());

    // Prepare filename
    std::string filename = "./results/" + method_type + "_" + input_file + ".solution";
    std::ofstream file(filename.c_str());

    if (!file.is_open()) {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
        return false;
    }

    file << "Average cost: " << average << std::endl;

    file << "Worst cost: " << worst_cost << " ; ";
    for (size_t i = 0; i < worst_path.size(); i++) {
        file << worst_path[i] << " ";
    }
    file << std::endl;

    file << "Best cost: " << best_cost << " ; ";
    for (size_t i = 0; i < best_path.size(); i++) {
        file << best_path[i] << " ";
    }
    file << std::endl;

    file.flush();
    file.close();

    return true;
}

// Function to generate a random starting solution
std::vector<int> generateRandomSolution(const std::vector<std::vector<int>> &distanceMatrix, const std::vector<int> &costLookupTable, int k, std::mt19937 &rng) {
    int n = distanceMatrix.size();
    std::vector<int> nodes(n);
    for (int i = 0; i < n; i++)
        nodes[i] = i;

    std::shuffle(nodes.begin(), nodes.end(), rng);

    // Select exactly k nodes
    std::vector<int> selectedNodes(nodes.begin(), nodes.begin() + k);
    selectedNodes.push_back(selectedNodes[0]); // Close the cycle

    return selectedNodes;
}

// Function to generate a greedy starting solution using a greedy cycle heuristic
std::vector<int> generateGreedySolution(int startNode, const std::vector<std::vector<int>> &distanceMatrix, const std::vector<int> &costLookupTable, int k) {
    int n = distanceMatrix.size();
    std::vector<int> path;
    std::vector<bool> visited(n, false);

    // Start from the startNode
    path.push_back(startNode);
    path.push_back(startNode);
    visited[startNode] = true;

    while (path.size() < static_cast<size_t>(k + 1)) {
        // Go over all unvisited nodes and find the one that minimizes the increase in the objective function
        int smallestIncrease = INT_MAX;
        int bestNode = -1;
        int bestPosition = -1;
        for (int i = 0; i < n; i++) {
            // If node i is in the path then skip
            if (visited[i] == true)
                continue;

            for (int j = 0; j < static_cast<int>(path.size()) - 1; j++) {
                // Check what happens if we insert here
                int left = path[j];
                int right = path[j + 1];
                int increase = distanceMatrix[left][i] + distanceMatrix[i][right] - distanceMatrix[left][right];

                // Add node cost
                increase += costLookupTable[i];

                if (increase < smallestIncrease) {
                    smallestIncrease = increase;
                    bestNode = i;
                    bestPosition = j + 1; // +1 because we insert between current j and j + 1
                }
            }
        }

        if (bestNode != -1) {
            path.insert(path.begin() + bestPosition, bestNode);
            visited[bestNode] = true;
        } else {
            break; // No more nodes to add
        }
    }

    // Close the cycle
    if (!path.empty())
        path.back() = path[0];

    return path;
}

// Function to perform two-nodes exchange within the same route
bool twoNodesExchange(const std::vector<int> &path, int &delta, 
                      const std::vector<std::vector<int>> &distanceMatrix, 
                      const int i, const int j) {
    if (i == j)
        return false; // No swap needed if indices are the same

    int n = path.size() - 1; // Assuming path[0] == path[n] (cycle)

    // Validate indices
    if (i < 0 || i >= n || j < 0 || j >= n)
        return false;

    // Ensure i < j for consistent handling
    int node1 = i;
    int node2 = j;
    if (j < i) std::swap(node1, node2);

    // Check if nodes are adjacent
    bool adjacent = (node1 + 1) % n == node2;

    if (adjacent) {
        // Handle adjacent nodes
        int prev_i = (node1 - 1 + n) % n;
        int next_j = (node2 + 1) % n;

        int old_dist = distanceMatrix[path[prev_i]][path[node1]] +
                       distanceMatrix[path[node1]][path[node2]] +
                       distanceMatrix[path[node2]][path[next_j]];

        int new_dist = distanceMatrix[path[prev_i]][path[node2]] +
                       distanceMatrix[path[node2]][path[node1]] +
                       distanceMatrix[path[node1]][path[next_j]];

        delta = new_dist - old_dist;
    } else {
        // Handle non-adjacent nodes
        int prev_i = (node1 - 1 + n) % n;
        int next_i = (node1 + 1) % n;
        int prev_j = (node2 - 1 + n) % n;
        int next_j = (node2 + 1) % n;

        // If node1 and node2 are not the same and not adjacent
        int old_dist = distanceMatrix[path[prev_i]][path[node1]] +
                       distanceMatrix[path[node1]][path[next_i]] +
                       distanceMatrix[path[prev_j]][path[node2]] +
                       distanceMatrix[path[node2]][path[next_j]];

        int new_dist = distanceMatrix[path[prev_i]][path[node2]] +
                       distanceMatrix[path[node2]][path[next_i]] +
                       distanceMatrix[path[prev_j]][path[node1]] +
                       distanceMatrix[path[node1]][path[next_j]];

        delta = new_dist - old_dist;
    }

    return true;
}

bool twoEdgesExchange(std::vector<int> &path, int &delta, const std::vector<std::vector<int>> &distanceMatrix, int i, int j) {
    if (i == j || std::abs(i - j) <= 1)
        return false;

    int n = path.size() - 1; // excluding the duplicate last node

    if (i < 0 || i >= n || j < 0 || j >= n)
        return false;

    // Ensure i < j
    if (i > j)
        std::swap(i, j);

    // Calculate the change in distance
    int prev_i = (i - 1 + n) % n;
    int next_j = (j + 1) % n;

    int old_dist = distanceMatrix[path[prev_i]][path[i]] + distanceMatrix[path[j]][path[next_j]];
    int new_dist = distanceMatrix[path[prev_i]][path[j]] + distanceMatrix[path[i]][path[next_j]];

    delta = new_dist - old_dist;

    return true;
}

// Function to perform inter-route move: swap a node inside the path with one outside
bool interRouteExchange(std::vector<int> &path, int &delta, const std::vector<std::vector<int>> &distanceMatrix, const std::vector<int> &costLookupTable, int inIdx, int outNode) {
    int n = path.size() - 1; // excluding the duplicate last node
    if (inIdx < 0 || inIdx >= n)
        return false;

    int inNode = path[inIdx];

    // Find positions around the inNode
    int prev = (inIdx - 1 + n) % n;
    int next = (inIdx + 1) % n;

    // Calculate old distances
    int old_dist = distanceMatrix[path[prev]][inNode] + distanceMatrix[inNode][path[next]];

    // Calculate new distances after swap
    int new_dist = distanceMatrix[path[prev]][outNode] + distanceMatrix[outNode][path[next]];

    // Calculate delta
    delta = new_dist - old_dist;

    // Update node costs: subtract inNode's cost and add outNode's cost
    delta += (costLookupTable[outNode] - costLookupTable[inNode]);

    return true;
}

// Function to perform greedy local search
bool greedyLocalSearch(std::vector<int> &path, int &currentCost, const std::string &intraMoveType, const std::vector<std::vector<int>> &distanceMatrix, const std::vector<int> &costLookupTable, std::mt19937 &rng) {
    bool improvement = false;
    int bestDelta = 0;
    std::string moveType = "";
    int pos1 = -1, pos2 = -1;

    // Generate a list of all possible moves
    std::vector<std::tuple<std::string, int, int>> allMoves;

    int n = path.size() - 1; // excluding the duplicate last node

    // Intra-route moves
    if (intraMoveType == "two_nodes" || intraMoveType == "both") {
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                allMoves.push_back(std::make_tuple("two_nodes_exchange", i, j));
            }
        }
    }

    // if (intraMoveType == "two_edges" || intraMoveType == "both") {
    //     for (int i = 0; i < n - 1; i++) {
    //         for (int j = i + 2; j < n; j++) {
    //             allMoves.push_back(std::make_tuple("two_edges_exchange", i, j));
    //         }
    //     }
    // }

    // // Inter-route moves: swap a node in the path with a node not in the path
    // for (int i = 0; i < n; i++) {
    //     for (int outNode = 0; outNode < distanceMatrix.size(); outNode++) {
    //         if (std::find(path.begin(), path.end() -1, outNode) != path.end() -1)
    //             continue; // Node already in the path

    //         allMoves.push_back(std::make_tuple("inter_route_swap", i, outNode));
    //     }
    // }

    // Shuffle the moves to randomize the order
    std::shuffle(allMoves.begin(), allMoves.end(), rng);

    // Iterate through the moves and apply the first improving move
    for (const auto &move : allMoves) {
        std::string mType = std::get<0>(move);
        int mPos1 = std::get<1>(move);
        int mPos2 = std::get<2>(move);
        int delta = 0;
        bool valid = false;

        if (mType == "two_nodes_exchange") {
            valid = twoNodesExchange(path, delta, distanceMatrix, mPos1, mPos2);
        }
        // else if (mType == "two_edges_exchange") {
        //     valid = twoEdgesExchange(path, delta, distanceMatrix, mPos1, mPos2);
        // }
        // else if (mType == "inter_route_swap") {
        //     valid = interRouteExchange(path, delta, distanceMatrix, costLookupTable, mPos1, mPos2);
        // }

        if (valid && delta < 0) { // Improvement found
            // Apply the move
            if (mType == "two_nodes_exchange") {
                std::swap(path[mPos1], path[mPos2]);
            }
            else if (mType == "two_edges_exchange") {
                std::reverse(path.begin() + mPos1 + 1, path.begin() + mPos2 + 1);
            }
            else if (mType == "inter_route_swap") {
                int inNode = path[mPos1];
                int outNode = mPos2;
                path[mPos1] = outNode;
            }

            currentCost += delta;
            std::cout << delta << std::endl;
            std::cout << "currentCost: " << currentCost << std::endl;
            improvement = true;
            break; // Only first improving move is applied
        }
    }

    return improvement;
}

// Function to perform steepest local search
bool steepestLocalSearch(std::vector<int> &path, int &currentCost, const std::string &intraMoveType, const std::vector<std::vector<int>> &distanceMatrix, const std::vector<int> &costLookupTable) {
    bool improvement = false;
    int bestDelta = 0;
    std::string moveType = "";
    int pos1 = -1, pos2 = -1;

    int n = path.size() - 1; // excluding the duplicate last node

    // Initialize best delta as no improvement
    bestDelta = 0;

    // Iterate over all possible intra-route moves
    if (intraMoveType == "two_nodes" || intraMoveType == "both") {
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                int delta = 0;
                if (twoNodesExchange(path, delta, distanceMatrix, i, j)) {
                    if (delta < bestDelta) {
                        bestDelta = delta;
                        moveType = "two_nodes_exchange";
                        pos1 = i;
                        pos2 = j;
                    }
                }
            }
        }
    }

    // if (intraMoveType == "two_edges" || intraMoveType == "both") {
    //     for (int i = 0; i < n - 1; i++) {
    //         for (int j = i + 2; j < n; j++) {
    //             int delta = 0;
    //             std::vector<int> tempPath = path;
    //             if (twoEdgesExchange(tempPath, delta, distanceMatrix, i, j)) {
    //                 if (delta < bestDelta) {
    //                     bestDelta = delta;
    //                     moveType = "two_edges_exchange";
    //                     pos1 = i;
    //                     pos2 = j;
    //                 }
    //             }
    //         }
    //     }
    // }

    // // Iterate over all possible inter-route moves
    // for (int i = 0; i < n; i++) {
    //     for (int outNode = 0; outNode < distanceMatrix.size(); outNode++) {
    //         if (std::find(path.begin(), path.end() -1, outNode) != path.end() -1)
    //             continue; // Node already in the path

    //         int delta = 0;
    //         std::vector<int> tempPath = path;
    //         if (interRouteExchange(tempPath, delta, distanceMatrix, costLookupTable, i, outNode)) {
    //             if (delta < bestDelta) {
    //                 bestDelta = delta;
    //                 moveType = "inter_route_swap";
    //                 pos1 = i;
    //                 pos2 = outNode;
    //             }
    //         }
    //     }
    // }

    // Apply the best move found
    if (bestDelta < 0) { // Improvement found
        if (moveType == "two_nodes_exchange") {
            std::swap(path[pos1], path[pos2]);
        }
        // else if (moveType == "two_edges_exchange") {
        //     std::reverse(path.begin() + pos1 + 1, path.begin() + pos2 + 1);
        // }
        // else if (moveType == "inter_route_swap") {
        //     path[pos1] = pos2;
        // }

        currentCost += bestDelta;
        std::cout << "currentCost: " << currentCost << std::endl;
        improvement = true;
    }

    return improvement;
}

// Function to perform local search based on method type
bool performLocalSearch(std::vector<int> &path, int &currentCost, const std::string &searchType, const std::string &intraMoveType, const std::vector<std::vector<int>> &distanceMatrix, const std::vector<int> &costLookupTable, std::mt19937 &rng) {
    bool improved = false;

    if (searchType == "greedy") {
        improved = greedyLocalSearch(path, currentCost, intraMoveType, distanceMatrix, costLookupTable, rng);
    }
    else if (searchType == "steepest") {
        improved = steepestLocalSearch(path, currentCost, intraMoveType, distanceMatrix, costLookupTable);
    }

    return improved;
}

// Function to run local search until no improvement
std::vector<int> runLocalSearch(const std::vector<int> &initialPath, const std::string &searchType, const std::string &intraMoveType, const std::vector<std::vector<int>> &distanceMatrix, const std::vector<int> &costLookupTable, std::mt19937 &rng, int &finalCost) {
    std::vector<int> currentPath = initialPath;
    int currentCost = calculateTotalCost(currentPath, distanceMatrix, costLookupTable);
    bool improved = true;

    while (improved) {
        improved = performLocalSearch(currentPath, currentCost, searchType, intraMoveType, distanceMatrix, costLookupTable, rng);
    }

    finalCost = currentCost;
    std::cout << "Final cost: " << finalCost << std::endl;
    return currentPath;
}

// Function to perform a single run based on method parameters
std::vector<int> performSingleRun(int runNumber, const std::string &searchType, const std::string &intraMoveType, const std::string &startType, const std::vector<std::vector<int>> &distanceMatrix, const std::vector<int> &costLookupTable, int k, std::mt19937 &rng, int &finalCost) {
    std::vector<int> initialPath;

    if (startType == "random") {
        initialPath = generateRandomSolution(distanceMatrix, costLookupTable, k, rng);
    }
    else if (startType == "greedy") {
        int startNode = runNumber % distanceMatrix.size();
        initialPath = generateGreedySolution(startNode, distanceMatrix, costLookupTable, k);
    }
    else {
        std::cerr << "Invalid starting solution type: " << startType << std::endl;
        exit(1);
    }

    // Run local search
    std::vector<int> finalPath = runLocalSearch(initialPath, searchType, intraMoveType, distanceMatrix, costLookupTable, rng, finalCost);

    return finalPath;
}

// Main function
int main(int argc, char *argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input_file> <method_type>" << std::endl;
        std::cerr << "Method types: greedy_two_nodes_random, greedy_two_nodes_greedy, greedy_two_edges_random, greedy_two_edges_greedy, steepest_two_nodes_random, steepest_two_nodes_greedy, steepest_two_edges_random, steepest_two_edges_greedy" << std::endl;
        std::cerr << "Example: " << argv[0] << " input.csv greedy_two_nodes_random" << std::endl;
        return 1;
    }

    std::string input_file = argv[1];
    std::string method_type = argv[2];

    // Parse method_type to extract searchType, intraMoveType, startType
    std::string searchType, intraMoveType, startType;

    size_t first_underscore = method_type.find('_');
    size_t second_underscore = method_type.find('_', first_underscore + 1);
    size_t third_underscore = method_type.find('_', second_underscore + 1); 

    searchType = method_type.substr(0, first_underscore);
    intraMoveType = method_type.substr(first_underscore + 1, third_underscore - first_underscore - 1);
    startType = method_type.substr(third_underscore + 1);

    std::cout << "Search Type: " << searchType << std::endl;
    std::cout << "Intra Move Type: " << intraMoveType << std::endl;
    std::cout << "Start Type: " << startType << std::endl;

    // Read data from the file
    std::vector<Node> data = readCSV("../data/" + input_file);
    if (data.empty()) {
        std::cerr << "No data read from the file." << std::endl;
        return 1;
    }

    int elements = data.size();
    int k = std::ceil(elements / 2.0);

    // Get the distance matrix
    std::vector<std::vector<int>> distanceMatrix = getDistanceMatrix(data);

    // Get the cost lookup table
    std::vector<int> costLookupTable;
    for (const auto &node : data) {
        costLookupTable.push_back(node.cost);
    }

    // Print the number of nodes
    std::cout << "Number of nodes: " << elements << std::endl;
    std::cout << std::string(80, '-') << std::endl;

    // Initialize random number generator
    std::mt19937 rng(std::random_device{}());

    // Define the number of runs
    int nRuns = 200;

    // Placeholder for solutions and costs
    std::vector<std::vector<int>> solutions(nRuns);
    std::vector<int> totalCosts(nRuns);
    std::vector<double> runTimes(nRuns, 0.0);

    // Start the runs
    for (int run = 0; run < nRuns; run++) {
        auto startTime = std::chrono::high_resolution_clock::now();

        // Perform a single run
        std::vector<int> finalPath = performSingleRun(run, searchType, intraMoveType, startType, distanceMatrix, costLookupTable, k, rng, totalCosts[run]);

        // Calculate elapsed time
        auto endTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = endTime - startTime;
        runTimes[run] = elapsed.count();

        // Store the solution
        solutions[run] = finalPath;
    }

    // Calculate statistics
    double averageCost = std::accumulate(totalCosts.begin(), totalCosts.end(), 0.0) / nRuns;
    int minCost = *std::min_element(totalCosts.begin(), totalCosts.end());
    int maxCost = *std::max_element(totalCosts.begin(), totalCosts.end());

    double averageTime = std::accumulate(runTimes.begin(), runTimes.end(), 0.0) / nRuns;
    double minTime = *std::min_element(runTimes.begin(), runTimes.end());
    double maxTime = *std::max_element(runTimes.begin(), runTimes.end());

    // Display statistics
    std::cout << "Method: " << method_type << std::endl;
    std::cout << "Average Cost: " << averageCost << std::endl;
    std::cout << "Minimum Cost: " << minCost << std::endl;
    std::cout << "Maximum Cost: " << maxCost << std::endl;
    std::cout << "Average Running Time: " << averageTime << " seconds" << std::endl;
    std::cout << "Minimum Running Time: " << minTime << " seconds" << std::endl;
    std::cout << "Maximum Running Time: " << maxTime << " seconds" << std::endl;

    // Dump results to file
    dumpToFile(input_file, method_type, solutions, totalCosts);

    return 0;
}