#include <bits/stdc++.h>
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
    float average = std::accumulate(totalCosts.begin(), totalCosts.end(), 0) / totalCosts.size();

    int worst_index = std::distance(totalCosts.begin(), std::max_element(totalCosts.begin(), totalCosts.end()));
    int worst_cost = totalCosts[worst_index];
    std::vector<int> worst_path = solutions[worst_index];

    int best_index = std::distance(totalCosts.begin(), std::min_element(totalCosts.begin(), totalCosts.end()));
    int best_cost = totalCosts[best_index];
    std::vector<int> best_path = solutions[best_index];

    std::cout << "Average cost: " << average << std::endl;
    std::cout << "Worst cost: " << worst_cost << std::endl;
    std::cout << "Best cost: " << best_cost << std::endl;

    // Save average, worst, best to file
    std::ofstream file(algorithm_type + "_" + sn_type + '_' + input_file + ".solution");

    file << "Average cost: " << average << std::endl;

    file << "Worst cost: " << worst_cost << " ; ";
    for (int i = 0; i < worst_path.size(); i++)
    {
        file << worst_path[i] << " ";
    }

    file << "Best cost: " << best_cost << " ; ";
    for (int i = 0; i < best_path.size(); i++)
    {
        file << best_path[i] << " ";
    }

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
        for (int j = 0; j < n; j++)
        {
            int x1 = std::get<0>(data[i]);
            int y1 = std::get<1>(data[i]);
            int x2 = std::get<0>(data[j]);
            int y2 = std::get<1>(data[j]);

            int dx = x2 - x1;
            int dy = y2 - y1;

            // Calculate the euclidean distance
            distanceMatrix[i][j] = std::sqrt(dx * dx + dy * dy);

            // Make sure to mathematically round the distance to the nearest integer
            distanceMatrix[i][j] = std::round(distanceMatrix[i][j]);
        }
    }

    return distanceMatrix;
}

// Calculate total cost of a path (distances + cost of nodes)
int calculateTotalCost(const std::vector<int> &path, const std::vector<std::vector<int>> &distanceMatrix, const std::vector<int> &costLookupTable)
{
    int totalCost = 0;

    for (int i = 0; i < path.size() - 1; i++)
    {
        totalCost += distanceMatrix[path[i]][path[i + 1]];
    }

    totalCost += distanceMatrix[path[path.size() - 1]][path[0]];

    for (int i = 0; i < path.size(); i++)
    {
        totalCost += costLookupTable[path[i]];
    }

    return totalCost;
}

// Random solution
std::vector<int> randomSolution(const int startNode, const std::vector<std::vector<int>> &distanceMatrix, const std::vector<int> &costLookupTable)
{
    int n = distanceMatrix.size();
    std::vector<int> path(n);
    for (int i = 0; i < n; i++)
        path[i] = i;

    std::random_shuffle(path.begin(), path.end());

    // Make sure the start node is at the beginning of the path
    auto it = std::find(path.begin(), path.end(), startNode);
    std::rotate(path.begin(), it, path.end());

    // Select exactly 50% of the nodes with the same node at the beginning and end
    int half = std::ceil(n / 2.0);

    std::vector<int> selectedNodes(path.begin(), path.begin() + half - 1);
    selectedNodes.push_back(selectedNodes[0]);

    return selectedNodes;
}

// Nearest neighbor considering adding the node only at the end of the current path
std::vector<int> nearestNeighborEnd(const int startNode, const std::vector<std::vector<int>> &distanceMatrix, const std::vector<int> &costLookupTable)
{
    int n = distanceMatrix.size();
    std::vector<int> path;
    std::vector<bool> visited(n, false);
    int left_to_visit = std::ceil(n / 2.0) - 1;

    // Start from the first node
    int current = startNode;

    while (left_to_visit > 0)
    {
        path.push_back(current);
        visited[current] = true;

        int next = -1;
        int minDistance = 1e9;

        for (int i = 0; i < n; i++)
        {
            if (!visited[i] && distanceMatrix[current][i] < minDistance)
            {
                next = i;
                minDistance = distanceMatrix[current][i];
            }
        }

        current = next;
        left_to_visit--;
    }

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
        std::cerr << "SN types: random, each" << std::endl;
        std::cerr << "Example: " << argv[0] << " input.csv random" << std::endl;
        return 1;
    }

    std::string input_file = argv[1];
    std::string algorithm_type = argv[2];
    std::string sn_type = argv[3];

    // Read data from the file
    std::vector<std::tuple<int, int, int>> data = readCSV("../data/" + input_file);
    int elements = data.size();
    int half = std::ceil(elements / 2.0);

    // Get the distance matrix
    std::vector<std::vector<int>> distanceMatrix = getDistanceMatrix(data);

    // Get the cost lookup table
    std::vector<int> costLookupTable;
    for (auto &tuple : data)
    {
        costLookupTable.push_back(std::get<2>(tuple));
    }

    // Print the number of nodes and average distance
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
            std::cerr << "Invalid SN type!" << std::endl;
            return 1;
        }

        std::vector<int> path;

        if (algorithm_type == "random")
            path = randomSolution(startNode, distanceMatrix, costLookupTable);
        else if (algorithm_type == "nn1")
            path = nearestNeighborEnd(startNode, distanceMatrix, costLookupTable);
        else if (algorithm_type == "nn2")
            exit(1);
        else if (algorithm_type == "greedy")
            exit(1);
        else
        {
            std::cerr << "Invalid algorithm type!" << std::endl;
            return 1;
        }

        int totalCost = calculateTotalCost(path, distanceMatrix, costLookupTable);

        // ASSERT CORRECTNESS
        // Assert that the selected nodes are exactly 50% of the total nodes
        assert(path.size() == half);
        // Assert cycle
        assert(path.front() == path.back());
        // Assert start node is at the beginning
        assert(path.front() == startNode);

        // Store the solution and its total cost
        solutions.push_back(path);
        totalCosts.push_back(totalCost);
    }

    dumpToFile(input_file, algorithm_type, solutions, totalCosts, sn_type);

    return 0;
}
