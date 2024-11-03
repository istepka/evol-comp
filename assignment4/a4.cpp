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
#include <unordered_map>
#include <set>

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

std::pair<std::vector<int>, int> findKnearestNeighbors(
    int current, 
    int k, 
    const std::vector<std::vector<int>> &distanceMatrix, 
    const std::vector<int> &costLookupTable, 
    const std::vector<int> &solution
    ){
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

    // Trim the list to k
    if (nearestNeighbors.size() > k) {
        nearestNeighbors.resize(k);
    }

    
    std::vector<int> nearestNeighborsSimple(k);
    int furthest = nearestNeighbors.back().second;

    for (int i = 0; i < k; i++) {
        nearestNeighborsSimple[i] = nearestNeighbors[i].first;
    }

    return std::make_pair(nearestNeighborsSimple, furthest);
}

void fillOutMapping(std::unordered_map<int, std::set<int>> &mappingTable, const std::vector<int> &nodeList, const int index) {
    for (int node : nodeList) {
        // Check if mappingTable contains the node
        if (mappingTable.find(node) == mappingTable.end()) {
            // Create a new entry as a vector with one element (node)
            mappingTable[node] = std::set<int>{index};
        }
        else
            // Add the node to the vector
            mappingTable[node].insert(index);
    }
}

void cmls(std::vector<int> &solution, const std::vector<std::vector<int>> &distanceMatrix, const std::vector<int> &costLookupTable)
{
    int solutionSize = solution.size();
    int lastBestDelta = 0;
    int iteration = 0;

    std::vector<std::vector<int>> memoizationTable(solutionSize);
    std::vector<int> memoizationTableFurthest(solutionSize);
    std::unordered_map<int, std::set<int>> mappingTable;

    while (true) {
        iteration++;

        int bestDelta = -INT32_MAX;
        std::pair<int, int> bestPair;
        std::pair<int, int> otherTwo;

        for (int i = 0; i < solutionSize; i++) {
            int current = solution[i];

           
            if (memoizationTable[i].empty()) {
                // std::cout<<"Memoization table is empty for node "<<current<<std::endl;

                std::pair<std::vector<int>, int> p = findKnearestNeighbors(current, 10, distanceMatrix, costLookupTable, solution);
                memoizationTable[i] = p.first; 
                memoizationTableFurthest[i] = p.second;
                fillOutMapping(mappingTable, memoizationTable[i], i);
            }

            for (int j = 0; j < 10; j++) {
                for (int k = -1; k <= 1; k+=2) { // k acts as a multiplier to get the next and next next node both ways

                    int candidate = memoizationTable[i][j];
                    
                    // TODO: Both sides
                    int ip1_node = solution[safeIndex(i + k, solutionSize)]; // Current +- 1
                    int ip2_node = solution[safeIndex(i + k*2, solutionSize)]; // Current +- 2

                    int newc = distanceMatrix[current][candidate] + distanceMatrix[candidate][ip2_node] + costLookupTable[candidate];
                    int oldc = distanceMatrix[current][ip1_node] + distanceMatrix[ip1_node][ip2_node] + costLookupTable[ip1_node];
                    int delta = oldc - newc; // The lowest the better

                    if (delta > bestDelta) {
                        bestDelta = delta;
                        bestPair = std::make_pair(ip1_node, candidate); // Exchange ip1 and c to add proper candidate edge
                        otherTwo = std::make_pair(current, ip2_node);
                    }
                }

            }
        }

        if (bestDelta <= 0){
            // std::cout<<"Best delta is "<<bestDelta<<" so there is no point in going further"<<std::endl;
            break;
        }

        // Insert the candidate edge at ip2
        int swap_location = std::find(solution.begin(), solution.end(), bestPair.first) - solution.begin();
        solution[swap_location] = bestPair.second;


        // Now we have to intervene in the mapping table and memoization table
        int toRemove = bestPair.first;
        int toAdd = bestPair.second;


        // Clear memoized stuff based on mapping table for toAdd i.e., we need to recalculate the nearest neighbors when toAdd was one of them
        for (int i : mappingTable[toAdd]) {
            memoizationTable[i].clear();
            memoizationTableFurthest[i] = 0;
        }

        // Remove the old node from the mapping table
        mappingTable.erase(toAdd);


        // Update the memoization table by 
        // 1. Clearing the vector of the old node and triggering a recalculation of the nearest neighbors 
        memoizationTable[swap_location].clear();
        memoizationTableFurthest[swap_location] = 0;

        // 2. Calculating the distance of toRemove node to all others, erasing memoization table entries whenever toRemove is closer than the current furthest
        for (int i = 0; i < solutionSize; i++) {
            if (memoizationTable[i].empty()) {
                continue;
            }

            int current = solution[i];
            int furthest = memoizationTableFurthest[i];

            int newFurthest = distanceMatrix[current][toRemove] + costLookupTable[toRemove];

            if (newFurthest < furthest) {
                memoizationTable[i].clear();
                memoizationTableFurthest[i] = 0;
            }
        }


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


std::vector<int>findKnearestNeighborsWithin(
    int current, 
    int k, 
    const std::vector<std::vector<int>> &distanceMatrix, 
    const std::vector<int> &costLookupTable, 
    const std::vector<int> &solution
    ){
    // Find 10 nearest neighbors by looking up the distance matrix, additonally add the cost of the node and sort them
    std::vector<std::pair<int, float>> nearestNeighbors;

    // Prohibit immediate neighbors
    int prohibited[2] = {solution[safeIndex(current - 1, solution.size())], solution[safeIndex(current + 1, solution.size())]};

    for (int j = 0; j < solution.size(); j++) {
        if (solution[j] == current || solution[j] == prohibited[0] || solution[j] == prohibited[1]) {
            continue; 
        }
        nearestNeighbors.push_back(std::make_pair(solution[j], distanceMatrix[current][solution[j]] + costLookupTable[solution[j]]));
    }

    // Sort the nearest neighbors
    std::sort(nearestNeighbors.begin(), nearestNeighbors.end(), [](const std::pair<int, float> &a, const std::pair<int, float> &b) {
        return a.second < b.second;
    });

    // Trim the list to k
    if (nearestNeighbors.size() > k) {
        nearestNeighbors.resize(k);
    }

    std::vector<int> nearestNeighborsSimple(k);

    for (int i = 0; i < k; i++) {
        nearestNeighborsSimple[i] = nearestNeighbors[i].first;
    }

    // for (auto x : nearestNeighborsSimple) {
    //     std::cout << x << " ";
    // }
    // std::cout << std::endl;


    return nearestNeighborsSimple;
}

void cmls_edge(std::vector<int> &solution, const std::vector<std::vector<int>> &distanceMatrix, const std::vector<int> &costLookupTable)
{
    int solutionSize = solution.size();
    int lastBestDelta = 0;
    int iteration = 0;

    // // Print out the solution
    // for (auto x : solution) {
    //     std::cout << x << " ";
    // }
    // std::cout << std::endl;

    std::unordered_map<int, std::vector<int>> memoizationTable(solutionSize);
    // std::vector<int> memoizationTableFurthest(solutionSize);
    // std::unordered_map<int, std::set<int>> mappingTable;

    while (true) {
        iteration++;

        int bestDelta = -INT32_MAX;
        std::pair<int, int> bestPair;
        std::pair<int, int> otherTwo;
        std::string rotation_type;

        for (int i = 0; i < solutionSize; i++) {
            int current = solution[i];

           
            if (iteration == 1 || memoizationTable[i].empty()) {
                // std::cout<<"Memoization table is empty for node "<<current<<std::endl;

                std::vector<int> p = findKnearestNeighborsWithin(current, 10, distanceMatrix, costLookupTable, solution);
                memoizationTable[solution[i]] = p; 
                // memoizationTableFurthest[i] = p.second;
                // fillOutMapping(mappingTable, memoizationTable[i], i);
            }


            for (int candidate : memoizationTable[solution[i]]) { // j is the index of the node in the solution
                for (int k = -1; k <= 1; k+=2) { // k acts as a multiplier to get the next and next next node both ways
                    
                    // Find j 
                    int j = std::find(solution.begin(), solution.end(), candidate) - solution.begin();

                    // TODO: Both sides
                    int aindex = safeIndex(i + k, solutionSize);
                    int bindex = safeIndex(j + k, solutionSize);
                    int ap1_node = solution[aindex]; // Current +- 1
                    int bp1_node = solution[bindex]; // Current +- 1

                    int newc = distanceMatrix[current][candidate] + distanceMatrix[ap1_node][bp1_node];
                    int oldc = distanceMatrix[current][ap1_node] + distanceMatrix[candidate][bp1_node];
                    int delta = oldc - newc; // The lowest the better

                    if (delta > bestDelta) {
                        bestDelta = delta;
                        bestPair = std::make_pair(i, j); // Exchange ip1 and c to add proper candidate edge
                        otherTwo = std::make_pair(aindex, bindex);
                        rotation_type = k == 1 ? "positive" : "negative";
                    }
                }

            }
        }

        if (bestDelta <= 0){
            // std::cout<<"Best delta is "<<bestDelta<<" so there is no point in going further"<<std::endl;
            break;
        }

        // Insert the candidate edge at ip2
        // If positive then we need to stich the solution in a following way:
        // [0 : a] [b : a1] [b1 : n]
        // If negative then its:
        // [0 : a1] [b1 : a] [b : n]
        // Where a and a1 are smaller and b and b1 are larger
        // We're adding the edge between a and b and a1 and b1
        int a = bestPair.first < bestPair.second ? bestPair.first : bestPair.second;
        int b = bestPair.second > bestPair.first ? bestPair.second : bestPair.first;
        int a1 = a == bestPair.first ? otherTwo.first : otherTwo.second;
        int b1 = b == bestPair.first ? otherTwo.first : otherTwo.second;

        // If a == 0 then we need to move first element to the end, decrement all indices by 1 and swap a and b
        if (a == 0) {
            std::rotate(solution.begin(), solution.begin() + 1, solution.end());
            a = solutionSize - 1;
            a1 = safeIndex(a1 - 1, solutionSize);
            b = safeIndex(b - 1, solutionSize);
            b1 = safeIndex(b1 - 1, solutionSize);

            std::swap(a, b);
            std::swap(a1, b1);
        }
        
        if (b1 == 0) // then we need to move first element to the end, decrement all indices by 1 , no need to swap a and b
        {
            std::rotate(solution.begin(), solution.begin() + 1, solution.end());
            a = safeIndex(a - 1, solutionSize);
            a1 = safeIndex(a1 - 1, solutionSize);
            b = safeIndex(b - 1, solutionSize);
            b1 = solutionSize - 1;
        }

        // std::cout << "Starting rotation: " << rotation_type << std::endl;
        // std::cout << "Indices: a: " << a << " a1: " << a1 << " b: " << b << " b1: " << b1 << std::endl;

        std::vector<int> newSolution;

        if (rotation_type == "positive") {
            // Extract middle chunk
            std::vector<int> middleChunk(solution.begin() + a1, solution.begin() + b + 1); // inclusive

            // Reverse the middle chunk
            std::reverse(middleChunk.begin(), middleChunk.end());

            // Create new vector and insert the chunks
            newSolution.insert(newSolution.begin(), solution.begin(), solution.begin() + a + 1);
            newSolution.insert(newSolution.end(), middleChunk.begin(), middleChunk.end());
            newSolution.insert(newSolution.end(), solution.begin() + b1, solution.end());

        } else {
            // Extract middle chunk
            std::vector<int> middleChunk(solution.begin() + a , solution.begin() + b1 + 1); // inclusive

            // Reverse the middle chunk
            std::reverse(middleChunk.begin(), middleChunk.end());

            // Create new vector and insert the chunks
            newSolution.insert(newSolution.begin(), solution.begin(), solution.begin() + a1 + 1);
            newSolution.insert(newSolution.end(), middleChunk.begin(), middleChunk.end());
            newSolution.insert(newSolution.end(), solution.begin() + b, solution.end());
        }

        assert(newSolution.size() == solution.size());


        solution = newSolution;

        




        // // Clear memoized stuff based on mapping table for toAdd i.e., we need to recalculate the nearest neighbors when toAdd was one of them
        // for (int i : mappingTable[toAdd]) {
        //     memoizationTable[i].clear();
        //     memoizationTableFurthest[i] = 0;
        // }

        // // Remove the old node from the mapping table
        // mappingTable.erase(toAdd);


        // // Update the memoization table by 
        // // 1. Clearing the vector of the old node and triggering a recalculation of the nearest neighbors 
        // memoizationTable[swap_location].clear();
        // memoizationTableFurthest[swap_location] = 0;

        // // 2. Calculating the distance of toRemove node to all others, erasing memoization table entries whenever toRemove is closer than the current furthest
        // for (int i = 0; i < solutionSize; i++) {
        //     if (memoizationTable[i].empty()) {
        //         continue;
        //     }

        //     int current = solution[i];
        //     int furthest = memoizationTableFurthest[i];

        //     int newFurthest = distanceMatrix[current][toRemove] + costLookupTable[toRemove];

        //     if (newFurthest < furthest) {
        //         memoizationTable[i].clear();
        //         memoizationTableFurthest[i] = 0;
        //     }
        // }



        // if (iteration % 1 == 0) {
        //     std::cout << "Iteration: " << iteration << " Best delta: " << bestDelta << std::endl;
        //     std::cout << "Indices: " << a << " " << b << " " << a1 << " " << b1 << std::endl;
        //     for (auto x : solution) {
        //         std::cout << x << " ";
        //     }
        //     std::cout << std::endl;
        //     int tc = calculateTotalCost(solution, distanceMatrix, costLookupTable);
        //     std::cout << "Total cost after insertion: " << tc << std::endl;

        //     // if (iteration % 50 == 0)
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
        if (algorithm_type == "cmls_edge")
            cmls_edge(path, distanceMatrix, costLookupTable);
        else  
            std::cerr << "Invalid algorithm type!" << std::endl;

        // Add the start node to the end of the path to close the cycle
        if (path.back() != path.front())
        {
            path.push_back(path.front());
        }

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