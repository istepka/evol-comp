#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <list>
#include <memory>
#include <map>
#include <coroutine>
#include <algorithm>
#include <random>
#include <cassert>
#include <deque>
#include <unordered_set>

using namespace std;

auto rng = std::default_random_engine(42);

struct Result
{
    int bestCost;
    int worstCost;
    int averageCost;
    vector<int> bestSolution;
    vector<int> worstSolution;

    Result(int bc, int wc, int ac, vector<int> bs, vector<int> ws)
        : bestCost(bc), worstCost(wc), averageCost(ac), bestSolution(bs), worstSolution(ws) {}
};

int calculate_cost(vector<int> solution, vector<vector<int>> distances, vector<int> costs)
{
    int cost = 0;
    for (int j = 0; j < solution.size() - 1; j++)
    {
        cost += distances[solution[j]][solution[j + 1]];
        cost += costs[solution[j]];
    }
    cost += distances[solution[solution.size() - 1]][solution[0]];
    return cost;
}

class Algo
{
public:
    vector<vector<int>> distances;
    vector<int> costs;
    string name;
    Algo(vector<vector<int>> distances, vector<int> costs, string name)
        : distances(distances), costs(costs), name(name) {}
    virtual Result solve() = 0;
    int calculate_cost(vector<int> &solution)
    {
        int cost = 0;
        for (int j = 0; j < solution.size() - 1; j++)
        {
            cost += distances[solution[j]][solution[j + 1]];
            cost += costs[solution[j]];
        }
        cost += distances[solution[solution.size() - 1]][solution[0]];
        return cost;
    }
    string get_name()
    {
        return this->name;
    }
};



class Greedy2RegretWieghted: public Algo {
public:
    Greedy2RegretWieghted(vector<vector<int>> distances, vector<int> costs):
        Algo(distances, costs, "Greedy2RegretWieghted"){}
    
    Result solve()
    {
        return Result(0, 0, 0, vector<int>(), vector<int>());
    }

    Result repair(vector<int> partial_solution){
        vector<int> bestSolution;
        vector<int> worstSolution;
        int bestCost = INT32_MAX;
        int worstCost = 0;
        int averageCost = 0;
        int solution_size = distances.size()/2;
        vector<int> current_solution = partial_solution;
        vector<vector<int>> graph;
        int starting_node;
        vector<bool> visited(costs.size());
        for(int i=0; i<partial_solution.size(); i++){
            visited[partial_solution[i]] = true;
        }
        while(current_solution.size() < solution_size){
            
            int smallest_increase = INT32_MAX;
            int insert_index = -1;
            int insert_node = -1;
            int max_score = -INT32_MAX;


            for(int k=0; k<distances.size(); k++){ // dla wszystkich nieodwiedzonych nodeów
                if(visited[k]) continue;
                vector<int> insertion_cost_for_j;
                for(int j=0; j<current_solution.size(); j++){ // dla każdego nodea z cyklu
                    int curr = -distances[current_solution[j == 0 ? current_solution.size() - 1 : j - 1]][current_solution[j]] + distances[current_solution[j == 0 ? current_solution.size() - 1 : j - 1]][k] + distances[k][current_solution[j]] + costs[k];
                    insertion_cost_for_j.push_back(curr);
                }
                int smallest_index = -1;
                int smallest_value = INT32_MAX;
                int second_smallest_value = INT32_MAX;

                for (int h = 0; h < insertion_cost_for_j.size(); h++) {
                    if (insertion_cost_for_j[h] < smallest_value) {
                        second_smallest_value = smallest_value;
                        smallest_value = insertion_cost_for_j[h];
                        smallest_index = h;
                    } else if (insertion_cost_for_j[h] < second_smallest_value) {
                        second_smallest_value = insertion_cost_for_j[h];
                    }
                }
                int regret = second_smallest_value - smallest_value;
                int left_node_idx = smallest_index == 0 ? current_solution.size() -1 : smallest_index -1;
                int insertion_cost = - distances[current_solution[left_node_idx]][current_solution[smallest_index]] + distances[current_solution[left_node_idx]][k] + distances[k][current_solution[smallest_index]] + costs[k];
                int score = regret - insertion_cost;
                if(score> max_score){
                    max_score = score;
                    insert_index = smallest_index;
                    insert_node = k;
                }
            }

            current_solution.insert(current_solution.begin() + insert_index, insert_node);
            visited[insert_node] = true; 
            // curr_graph.push_back(current_solution);
        }
        int current_cost = calculate_cost(current_solution);
        if(current_cost > worstCost){
            worstCost = current_cost;
            worstSolution = current_solution;
        }
        averageCost += current_cost;
        current_solution.clear();
        return Result(bestCost, worstCost, averageCost/distances.size(), bestSolution, worstSolution);
    }
};


class GreedyCycle : public Algo
{
public:
    GreedyCycle(vector<vector<int>> distances, vector<int> costs, int i)
        : Algo(distances, costs, "GreedyCycle") {}

    Result solve()
    {
        return Result(0, 0, 0, vector<int>(), vector<int>());
    }

    Result repair(vector<int> partial_solution)
    {
        vector<int> worstSolution;
        int solution_size = this->distances.size() / 2;
        vector<int> current_solution = partial_solution;
        vector<bool> visited(this->costs.size());
        for (int i = 0; i < current_solution.size(); i++)
        {
            visited[current_solution[i]] = true;
        }
        while (current_solution.size() < solution_size)
        {
            int smallest_increase = INT32_MAX;
            int insert_index = -1;
            int insert_node = -1;

            for (int j = 0; j < current_solution.size(); j++)
            { // Dla każdego nodea z cyklu
                int min_distance = INT32_MAX;
                int min_index = -1;
                for (int k = 0; k < this->distances.size(); k++)
                { // znajdź najbliższy nieodwiedzony node
                    if (visited[k])
                        continue;
                    int curr = -this->distances[current_solution[j == 0 ? current_solution.size() - 1 : j - 1]][current_solution[j]] + this->distances[current_solution[j == 0 ? current_solution.size() - 1 : j - 1]][k] + this->distances[k][current_solution[j]] + this->costs[k];
                    if (curr < min_distance)
                    {
                        min_distance = curr;
                        min_index = k;
                    }
                }
                if (min_distance < smallest_increase)
                {
                    smallest_increase = min_distance;
                    insert_index = j;
                    insert_node = min_index;
                }
            } // koniec
            current_solution.insert(current_solution.begin() + insert_index, insert_node);
            visited[insert_node] = true;
        }
        return Result(0, 0, 0, current_solution, worstSolution);
    }
};

template <typename T>
struct generator
{
    struct promise_type;
    using handle_type = std::coroutine_handle<promise_type>;

    struct promise_type
    {
        T value;
        std::suspend_always initial_suspend() { return {}; }
        std::suspend_always final_suspend() noexcept { return {}; }
        generator get_return_object() { return generator{handle_type::from_promise(*this)}; }
        void unhandled_exception() { std::terminate(); }
        std::suspend_always yield_value(T val)
        {
            value = val;
            return {};
        }
    };

    bool move_next() { return coro ? (coro.resume(), !coro.done()) : false; }
    T current_value() { return coro.promise().value; }

    generator(generator const &) = delete;
    generator(generator &&other) : coro(other.coro) { other.coro = {}; }
    ~generator()
    {
        if (coro)
            coro.destroy();
    }

private:
    generator(handle_type h) : coro(h) {}
    handle_type coro;
};

class LocalSearch : public Algo
{
public:
    vector<bool> visited;
    LocalSearch(vector<vector<int>> distances, vector<int> costs)
        : Algo(distances, costs, "LS")
    {
        visited = vector<bool>(distances.size());
    }

    int fixIndex(int index, int solutionSize)
    {
        if (index < 0)
        {
            return solutionSize + index;
        }
        if (index >= solutionSize)
        {
            return index - solutionSize;
        }
        return index;
    }

    Result solve()
    {
        return Result(0, 0, 0, vector<int>(), vector<int>());
    }

    void localSearch(shared_ptr<vector<int>> solution)
    {
        while (true)
        {
            auto neighbourhoodIterator2 = neighbourhoodGenerator(*solution);
            int bestDelta = INT32_MAX;
            vector<int> bestMove;
            while (neighbourhoodIterator2.move_next())
            {
                vector<int> move = neighbourhoodIterator2.current_value();
                int delta = calculateDelta(*solution, move);
                if (delta < bestDelta)
                {
                    bestDelta = delta;
                    bestMove = move;
                }
            }
            if (bestDelta >= 0)
            {
                break;
            }
            applyMove(solution, bestMove);
        }
    }

    int calculateDelta(vector<int> solution, vector<int> &move)
    {
        int delta;
        if (move.size() == 3)
        {
            // exchange nodes
            int i = move[0];
            int new_node = move[1];
            int old_node = solution[i];
            int oldCost = costs[old_node] + distances[old_node][solution[fixIndex(i + 1, solution.size())]] + distances[old_node][solution[fixIndex(i - 1 + solution.size(), solution.size())]];
            int newCost = costs[new_node] + distances[new_node][solution[fixIndex(i + 1, solution.size())]] + distances[new_node][solution[fixIndex(i - 1 + solution.size(), solution.size())]];
            delta = newCost - oldCost;
        }
        else if (move.size() == 4)
        {
            // edge exchnge
            int edge1_first = solution[move[0]];
            int edge1_second = solution[move[1]];
            int edge2_first = solution[move[2]];
            int edge2_second = solution[move[3]];
            int oldCost = distances[edge1_first][edge1_second] + distances[edge2_first][edge2_second];
            int newCost = distances[edge1_first][edge2_first] + distances[edge1_second][edge2_second];
            delta = newCost - oldCost;
        }
        else
        {
            throw runtime_error("Wrong size of best move");
        }
        return delta;
    }

    void applyMove(shared_ptr<vector<int>> solution, const vector<int> &move)
    { // modifies solution i think usuall  poinnter is enuogh
        if (move.size() == 3)
        {
            int i = move[0];
            int j = move[1];
            (*solution)[i] = j;
            visited[move[2]] = false;
            visited[j] = true;
        }
        else if (move.size() == 4)
        {
            int j = move[1];
            int k = move[2];
            reverse(solution->begin() + j, solution->begin() + k + 1);
        }
    }

    generator<vector<int>> neighbourhoodGenerator(vector<int> &currentSolution)
    {
        auto intraNeighbourhoodIterator = intraEdgesNeighbourhoodGenerator(currentSolution);
        while (intraNeighbourhoodIterator.move_next())
        {
            co_yield intraNeighbourhoodIterator.current_value();
        }
        auto interNeighbourhoodIterator = interNeighbourhoodGenerator(currentSolution);
        while (interNeighbourhoodIterator.move_next())
        {
            co_yield interNeighbourhoodIterator.current_value();
        }
    }

    generator<vector<int>> interNeighbourhoodGenerator(vector<int> &currentSolution)
    {
        vector<int> move = {0, 0};
        for (int i = 0; i < currentSolution.size(); i++)
        {
            int currentnode = currentSolution[i];
            for (int j = 0; j < distances.size(); j++)
            {
                if (!visited[j])
                {
                    co_yield makeInterMove(i, j, currentSolution[i]);
                }
            }
        }
    }

    vector<int> makeInterMove(int currentNodeId, int newNode, int currentNode)
    {
        return {currentNodeId, newNode, currentNode};
    }

    generator<vector<int>> intraEdgesNeighbourhoodGenerator(vector<int> &currentSolution)
    {
        vector<int> temp_vec = {0, 0, 0, 0};
        vector<int> move = vector<int>(temp_vec);
        for (int i = 0; i < currentSolution.size(); i++)
        {
            int node1 = currentSolution[i];
            int node1_next = currentSolution[fixIndex(i + 1, currentSolution.size())];
            for (int j = i + 2; j < currentSolution.size(); j++)
            {
                int node2 = currentSolution[j];
                int node2_next = currentSolution[fixIndex(j + 1, currentSolution.size())];
                co_yield makeIntraMove(i, i + 1, j, fixIndex(j + 1, currentSolution.size()));
            }
        }
    }

    vector<int> makeIntraMove(int edge1_first, int edge1_second, int edge2_first, int edge2_second)
    {
        return {edge1_first, edge1_second, edge2_first, edge2_second};
    }
};

class EvolutionaryAlgorithm : public Algo
{
public:
    int solutionSize;
    int populationSize;
    int maxIterations;
    int maxTime;
    int elitismSize;
    LocalSearch ls;
    bool doLocalSearch;
    map<int, shared_ptr<vector<int>>> population;

    EvolutionaryAlgorithm(vector<vector<int>> distances, vector<int> costs, string name, int population_size, int max_iterations, int max_time, int elitism_size, bool doLocalSearch)
        : Algo(distances, costs, name), populationSize(population_size), maxIterations(max_iterations), maxTime(max_time), elitismSize(elitism_size), ls(distances, costs), doLocalSearch(doLocalSearch)
    {
        this->solutionSize = distances.size() / 2;
        if(doLocalSearch){
            this->name += "_LS";
        }
    }

        Result solve()
    {
        generate_initial_population();
        int iteration = 0;
        clock_t start, end;
        start = clock();
        while (iteration < maxIterations && (double(end - start) / double(CLOCKS_PER_SEC)) < maxTime) {
            iteration++;
            map<int, shared_ptr<vector<int>>> new_population;
            while (new_population.size() < populationSize - elitismSize) {
                pair<shared_ptr<vector<int>>, shared_ptr<vector<int>>> parents = select_parents();
                vector<shared_ptr<vector<int>>> children = crossover(parents.first, parents.second);
                for (auto child: children) {
                    if (doLocalSearch) {
                        ls.localSearch(child);
                    }
                    int cost = calculate_cost(*child);
                    if (new_population.find(cost) == new_population.end() and
                        population.find(cost) == population.end()) {
                        new_population[cost] = child;
                    }
                    if (new_population.size() == populationSize - elitismSize) {
                        break;
                    }
                }
            }
            while (population.size() > elitismSize) {
                population.erase(population.rbegin()->first);
            }
            for (auto it = new_population.begin(); it != new_population.end(); it++) {
                population[it->first] = it->second;
            }
            end = clock();
        }

        //best solution has lowest score
        int bestCost = population.begin()->first;
        vector<int> bestSolution = *population.begin()->second;
        return Result(bestCost, bestCost, bestCost, bestSolution, bestSolution);
    }

    void generate_initial_population()
    {
        while (population.size() < populationSize)
        {
            shared_ptr<vector<int>> solution = make_shared<vector<int>>(solutionSize);
            iota(solution->begin(), solution->end(), 0);
            shuffle(solution->begin(), solution->end(), rng);
            solution->resize(solutionSize);
            for (int i = 0; i < ls.visited.size(); i++){
                ls.visited[i] = false;
            }
            for (int i = 0; i < solution->size(); i++){
                ls.visited[(*solution)[i]] = true;
            }
            ls.localSearch(solution);
            
            int cost = calculate_cost(*solution);
            if (population.find(cost) == population.end())
            {
                population[cost] = solution;
            }
        }
    }

    pair<shared_ptr<vector<int>>, shared_ptr<vector<int>>> select_parents()
    {
        std::uniform_int_distribution<> dist(0, populationSize - 1);
        auto first = std::begin(population);
        std::advance(first, dist(rng));
        auto second = std::begin(population);
        do
        {
            second = std::begin(population);
            std::advance(second, dist(rng));
        } while (first == second);
        return make_pair(first->second, second->second);
    }

    vector<shared_ptr<vector<int>>> crossover(shared_ptr<vector<int>> first_parent, shared_ptr<vector<int>> second_parent)
    {
        // random crossover 1 or 2
        if (rand() % 2 == 0)
        {
            return crossover1(first_parent, second_parent);
        }
        else
        {
            return crossover2(first_parent, second_parent);
        }
    }

    // we do not need common edges
    vector<shared_ptr<vector<int>>> crossover1(shared_ptr<vector<int>> first_parent, shared_ptr<vector<int>> second_parent)
    {
        unordered_set<int> first_nodes;
        unordered_set<int> second_nodes;
        for (int i = 0; i < first_parent->size(); i++)
        {
            first_nodes.insert((*first_parent)[i]);
            second_nodes.insert((*second_parent)[i]);
        }
        unordered_set<int> common_nodes;
        for (int i = 0; i < first_parent->size(); i++)
        {
            if (second_nodes.find((*first_parent)[i]) != second_nodes.end())
            {
                common_nodes.insert((*first_parent)[i]);
            }
        }
        vector<int> firstChild(solutionSize);
        vector<int> secondChild(solutionSize);
        for (int i = 0; i < solutionSize; i++)
        {
            if (common_nodes.find((*first_parent)[i]) != common_nodes.end())
            {
                firstChild[i] = (*first_parent)[i];
            }
            else
            {
                firstChild[i] = -1;
            }
            if (common_nodes.find((*second_parent)[i]) != common_nodes.end())
            {
                secondChild[i] = (*second_parent)[i];
            }
            else
            {
                secondChild[i] = -1;
            }
        }
        vector<int> missingNodes;
        for (int i = 0; i < solutionSize; i++)
        {
            // if i not in common nodes
            if (common_nodes.find(i) == common_nodes.end())
            {
                missingNodes.push_back(i);
            }
        }
        vector<int> firstMissingNodes = missingNodes;
        vector<int> secondMissingNodes = missingNodes;
        shuffle(firstMissingNodes.begin(), firstMissingNodes.end(), rng);
        shuffle(secondMissingNodes.begin(), secondMissingNodes.end(), rng);
        for (int i = 0; i < solutionSize; i++)
        {
            if (firstChild[i] == -1)
            {
                firstChild[i] = firstMissingNodes.back();
                firstMissingNodes.pop_back();
            }
            if (secondChild[i] == -1)
            {
                secondChild[i] = secondMissingNodes.back();
                secondMissingNodes.pop_back();
            }
        }

        vector<shared_ptr<vector<int>>> children;
        if (doLocalSearch)
        {
            shared_ptr<vector<int>> child = make_shared<vector<int>>(firstChild);
            shared_ptr<vector<int>> child2 = make_shared<vector<int>>(secondChild);
            for (int i = 0; i < ls.visited.size(); i++){
                ls.visited[i] = false;
            }
            for (int i = 0; i < child->size(); i++){
                ls.visited[(*child)[i]] = true;
            }
            ls.localSearch(child);
            for (int i = 0; i < ls.visited.size(); i++){
                ls.visited[i] = false;
            }
            for (int i = 0; i < child2->size(); i++){
                ls.visited[(*child2)[i]] = true;
            }
            ls.localSearch(child2);

            children.push_back(child);
            children.push_back(child2);
        }
        else{
            children.push_back(make_shared<vector<int>>(firstChild));
            children.push_back(make_shared<vector<int>>(secondChild));
        }
        return children;
    }

    vector<shared_ptr<vector<int>>> crossover2(shared_ptr<vector<int>> parent1, shared_ptr<vector<int>> parent2)
    {
        GreedyCycle gc = GreedyCycle(distances, costs, rand() % distances.size());
        LocalSearch ls = LocalSearch(distances, costs);
        Greedy2RegretWieghted g2rw = Greedy2RegretWieghted(distances, costs);
        shared_ptr<vector<int>> child = make_shared<vector<int>>(*parent1); // Start with a copy of parent1
        shared_ptr<vector<int>> child2 = make_shared<vector<int>>(*parent2);
        vector<shared_ptr<vector<int>>> children;

        // Create sets for easier search
        unordered_set<int> parent1Nodes(parent1->begin(), parent1->end());
        unordered_set<int> parent2Nodes(parent2->begin(), parent2->end());

        // Remove nodes from child that are not present in parent2
        child->erase(
            remove_if(child->begin(), child->end(), [&parent2Nodes](int node)
                      { return parent2Nodes.find(node) == parent2Nodes.end(); }),
            child->end());
        
        // Remove nodes from child2 that are not present in parent1
        child2->erase(
            remove_if(child2->begin(), child2->end(), [&parent1Nodes](int node)
                      { return parent1Nodes.find(node) == parent1Nodes.end(); }),
            child2->end());

        child = make_shared<vector<int>>(gc.repair(*child).bestSolution); // Repair the child using greedy cycle
        child2 = make_shared<vector<int>>(g2rw.repair(*child2).bestSolution); // Repair the child using g2rw

        if (doLocalSearch)
        {
            for (int i = 0; i < ls.visited.size(); i++){
                ls.visited[i] = false;
            }
            for (int i = 0; i < child->size(); i++){
                ls.visited[(*child)[i]] = true;
            }
            ls.localSearch(child);
            for (int i = 0; i < ls.visited.size(); i++){
                ls.visited[i] = false;
            }
            for (int i = 0; i < child2->size(); i++){
                ls.visited[(*child2)[i]] = true;
            }
            ls.localSearch(child2);
        }
        children.push_back(child);
        children.push_back(child2);
        return children;
    }
};

enum ProblemInstance
{
    TSPA,
    TSPB,
    TSPC,
    TSPD
};

std::map<ProblemInstance, std::string> ProblemInstanceStrings = {
    {TSPA, "TSPA"},
    {TSPB, "TSPB"},
    {TSPC, "TSPC"},
    {TSPD, "TSPD"}};

// times
map<ProblemInstance, double> maxTimes = {
    {TSPA, 24.3308},
    {TSPB, 24.3841},
    {TSPC, 43.9474},
    {TSPD, 45.5663}};
    map<ProblemInstance, double> maxNewTimes = {
    {TSPA, 60},
    {TSPB, 60},
    {TSPC, 60},
    {TSPD, 60}};

vector<vector<int>> read_file(string filename)
{
    vector<vector<int>> result;
    ifstream file(filename);
    string line;
    while (getline(file, line))
    {
        vector<int> row;
        stringstream ss(line);
        string cell;
        while (getline(ss, cell, ';'))
        {
            row.push_back(stoi(cell));
        }
        result.push_back(row);
    }
    return result;
}

vector<vector<int>> calcDistances(vector<vector<int>> data)
{
    vector<vector<int>> distances;
    for (int i = 0; i < data.size(); i++)
    {
        vector<int> row;
        for (int j = 0; j < data.size(); j++)
        {
            int x = data[i][0] - data[j][0];
            int y = data[i][1] - data[j][1];
            // round to nearest int
            float distance = round(sqrt(x * x + y * y));
            row.push_back(distance);
        }
        distances.push_back(row);
    }
    return distances;
}

int main()
{
    string root_path = "../data/";
    vector<ProblemInstance> problemInstances = {TSPA, TSPB};//, TSPB, TSPC, TSPD};
    vector<bool> doLocalSearch = {true, false};
    int N_TRIES = 20;
    for (auto problemInstance : problemInstances)
    {
        string file = root_path + ProblemInstanceStrings[problemInstance] + ".csv";
        auto data = read_file(file);
        auto distances = calcDistances(data);
        vector<int> costs;
        for (int i = 0; i < data.size(); i++)
        {
            costs.push_back(data[i][2]);
        }
        for (auto doLocalSearch : doLocalSearch)
        {
            cout << "Name: " << EvolutionaryAlgorithm(distances, costs, "EA", 20, 10000000, maxNewTimes[problemInstance], 20, doLocalSearch).get_name() << endl;
            cout << "Problem instance: " << ProblemInstanceStrings[problemInstance] << endl;
            Result algoResult = Result(INT32_MAX, 0, 0, vector<int>(), vector<int>());
            double avg_iterations_number = 0;

            vector<double> times;
            for (int i = 0; i < N_TRIES; i++)
            {
                cout << "Try: " << i << endl;
                EvolutionaryAlgorithm ea = EvolutionaryAlgorithm(distances, costs, "EA", 300, 10000000, maxNewTimes[problemInstance], 20, doLocalSearch);
                clock_t start, end;
                start = clock();
                Result res = ea.solve();
                end = clock();
                vector<int> solution = res.bestSolution;
                avg_iterations_number += res.averageCost; // iterations number
                double time_taken = double(end - start) / double(CLOCKS_PER_SEC);
                int cost = ea.calculate_cost(solution);
                if (cost < algoResult.bestCost)
                {
                    algoResult.bestCost = cost;
                    algoResult.bestSolution = solution;
                }
                if (cost > algoResult.worstCost)
                {
                    algoResult.worstCost = cost;
                    algoResult.worstSolution = solution;
                }
                algoResult.averageCost += cost;
                times.push_back(time_taken);
            }
            avg_iterations_number /= N_TRIES;
            algoResult.averageCost /= N_TRIES;
            cout << "Best cost: " << algoResult.bestCost << endl;
            cout << "Worst cost: " << algoResult.worstCost << endl;
            cout << "Average cost: " << algoResult.averageCost << endl;
            cout << "Average time: " << accumulate(times.begin(), times.end(), 0.0) / times.size() << endl;
            cout << "Best time: " << *min_element(times.begin(), times.end()) << endl;
            cout << "Worst time: " << *max_element(times.begin(), times.end()) << endl;
            cout << "Best solution: ";
            for (int i = 0; i < algoResult.bestSolution.size(); i++)
            {
                cout << algoResult.bestSolution[i] << " ";
            }
            cout << endl;
            cout << "Average iterations number: " << avg_iterations_number << endl;
        }
    }
}