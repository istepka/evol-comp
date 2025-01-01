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

using namespace std;

bool DOINTRA = true;
auto rng = std::default_random_engine(869468469);

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
    int starting_node;
    string name;
    Algo(vector<vector<int>> distances, vector<int> costs, int i, string name)
        : distances(distances), costs(costs), starting_node(i), name(name) {}
    virtual Result solve() = 0;
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
    string get_name()
    {
        return this->name;
    }
};

class RandomSearch : public Algo
{
public:
    RandomSearch(vector<vector<int>> distances, vector<int> costs, int i)
        : Algo(distances, costs, i, "RandomSearch") {}

    Result solve()
    {
        vector<int> worstSolution;
        int solution_size = this->distances.size() / 2;
        vector<int> current_solution = vector<int>(solution_size);
        vector<int> visited(this->distances.size());

        for (int j = 0; j < solution_size; j++)
        {
            if (j == 0)
            {
                current_solution[j] = this->starting_node;
                visited[this->starting_node] = true;
                continue;
            }
            int next = rand() % this->distances.size();
            while (visited[next])
                next = rand() % this->distances.size();
            current_solution[j] = next;
            visited[next] = true;
        }
        return Result(0, 0, 0, current_solution, worstSolution);
    }
};

class GreedyCycle : public Algo
{
public:
    GreedyCycle(vector<vector<int>> distances, vector<int> costs, int i)
        : Algo(distances, costs, i, "GreedyCycle") {}

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

enum SearchType
{
    greedy,
    steepest
};
enum InitialSolutionType
{
    randomAlg,
    GC,
    G2Rw
};
enum NeighbourhoodType
{
    intra,
    inter
};
enum InterNeighbourhoodType
{
    twoNode,
    twoEdges
};
enum ProblemInstance
{
    TSPA,
    TSPB,
    TSPC,
    TSPD
};

std::map<SearchType, std::string> SearchTypeStrings = {
    {greedy, "greedy"},
    {steepest, "steepest"}};

std::map<InitialSolutionType, std::string> InitialSolutionTypeStrings = {
    {randomAlg, "random"},
};

std::map<NeighbourhoodType, std::string> NeighbourhoodTypeStrings = {
    {intra, "intra"},
    {inter, "inter"}};

std::map<InterNeighbourhoodType, std::string> InterNeighbourhoodTypeStrings = {
    {twoNode, "twoNode"},
    {twoEdges, "twoEdges"}};

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

enum MoveEvaluationResult
{
    doMove,
    doReversed,
    removeMove,
    skipMove
};

class LocalSearch : public Algo
{
public:
    SearchType searchType;
    InitialSolutionType initialSolutionType;
    InterNeighbourhoodType intraNeighbourhoodType;
    vector<bool> visited;
    int nPoints;
    LocalSearch(SearchType searchType, InitialSolutionType initialSolutionType, InterNeighbourhoodType intraNeighbourhoodType, vector<vector<int>> distances, vector<int> costs, int i)
        : Algo(distances, costs, i, "LS"), searchType(searchType), initialSolutionType(initialSolutionType), intraNeighbourhoodType(intraNeighbourhoodType)
    {
        this->name += "_" + SearchTypeStrings[searchType];
        this->name += "_" + InitialSolutionTypeStrings[initialSolutionType];
        this->name += "_" + InterNeighbourhoodTypeStrings[intraNeighbourhoodType];
        visited = vector<bool>(distances.size());
        nPoints = distances.size();
    }

    int calculate_cost(const vector<int> &solution)
    {
        int cost = 0;
        for (int j = 0; j < solution.size() - 1; j++)
        {
            cost += this->distances[solution[j]][solution[j + 1]];
        }
        cost += this->distances[solution[solution.size() - 1]][solution[0]];
        for (int j = 0; j < solution.size(); j++)
        {
            cost += this->costs[solution[j]];
        }
        return cost;
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
        vector<int> solution = getInitialSolution(this->initialSolutionType, starting_node);
        for (int i = 0; i < visited.size(); i++)
        {
            visited[i] = false;
        }
        for (int i = 0; i < solution.size(); i++)
        {
            visited[solution[i]] = true;
        }
        localSearch(&solution);
        return Result(calculate_cost(solution), 0, 0, solution, vector<int>());
    }

    void localSearch(vector<int> *solution)
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

    vector<int> getInitialSolution(InitialSolutionType ist, int i)
    {
        if (ist == randomAlg)
        {
            RandomSearch rs = RandomSearch(distances, costs, i);
            return rs.solve().bestSolution;
        }
    }

    int calculateDelta(vector<int> &solution, vector<int> &move)
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

    void applyMove(vector<int> *solution, const vector<int> &move)
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
        if (DOINTRA)
        {
            auto intraNeighbourhoodIterator = intraEdgesNeighbourhoodGenerator(currentSolution);
            while (intraNeighbourhoodIterator.move_next())
            {
                co_yield intraNeighbourhoodIterator.current_value();
            }
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

class LSNS : public LocalSearch
{
public:
    double maxTime;
    bool doLocalSearch;
    LSNS(SearchType searchType, InitialSolutionType initialSolutionType, InterNeighbourhoodType intraNeighbourhoodType, vector<vector<int>> distances, vector<int> costs, int i, double maxTime, bool doLocalSearch)
        : LocalSearch(searchType, initialSolutionType, intraNeighbourhoodType, distances, costs, i), maxTime(maxTime), doLocalSearch(doLocalSearch)
    {
        if (doLocalSearch)
        {
            this->name = "LSNS_LS" + this->name;
        }
        else
        {
            this->name = "LSNS_NOLS" + this->name;
        }
    }
    Result solve() override
    {
        clock_t start;
        start = clock();
        LocalSearch ls = LocalSearch(searchType, initialSolutionType, intraNeighbourhoodType, distances, costs, rand() % distances.size());
        GreedyCycle gc = GreedyCycle(distances, costs, rand() % distances.size());
        vector<int> bestSolution = ls.solve().bestSolution;
        int bestCost = ls.calculate_cost(bestSolution);

        int numIters = 0;

        while (double(clock() - start) / double(CLOCKS_PER_SEC) < maxTime)
        {
            // Destroy part of the solution
            vector<int> destroyedSolution = destroySolution(bestSolution);

            // Repair the solution using GC
            destroyedSolution = gc.repair(destroyedSolution).bestSolution;

            // Update visited

            if (doLocalSearch)
            {
                for (int i = 0; i < ls.visited.size(); i++)
                {
                    ls.visited[i] = false;
                    visited[i] = false;
                }
                for (int i = 0; i < destroyedSolution.size(); i++)
                {
                    ls.visited[destroyedSolution[i]] = true;
                    visited[destroyedSolution[i]] = true;
                }
                // Local search
                ls.localSearch(&destroyedSolution);
            }
            int currentCost = ls.calculate_cost(destroyedSolution);

            if (currentCost < bestCost)
            {
                bestSolution = destroyedSolution;
                bestCost = currentCost;
            }
            numIters++;
        }
        return Result(bestCost, 0, numIters, bestSolution, vector<int>());
    }

    static bool compareDestrucions(pair<int, int> a, pair<int, int> b)
    {
        return a.second > b.second;
    }

    vector<int> destroySolution(vector<int> solution)
    {
        int destructions = rand() % 10;
        deque<pair<int, int>> destructionsNodes = deque<pair<int, int>>();
        deque<pair<int, int>> destructionsEdges = deque<pair<int, int>>();

        // possible destructions
        for (int j = 0; j < solution.size(); j++)
        {
            destructionsNodes.push_back(make_pair(j, costs[solution[j]]));
        }
        for (int j = 0; j < solution.size(); j++)
        {
            int edgeLen = distances[solution[j]][solution[fixIndex(j + 1, solution.size())]];
            destructionsEdges.push_back(make_pair(j, edgeLen));
        }

        partial_sort(destructionsNodes.begin(), destructionsNodes.begin() + destructions, destructionsNodes.end(), compareDestrucions);
        partial_sort(destructionsEdges.begin(), destructionsEdges.begin() + destructions, destructionsEdges.end(), compareDestrucions);
        vector<int> removed_indices = vector<int>();
        int toRemove;
        for (int j = 0; j < destructions; j++)
        {
            int nodeCost = destructionsNodes.front().second;
            int edgeLen = destructionsEdges.front().second;
            if (nodeCost <= rand() % nodeCost + edgeLen)
            {
                toRemove = destructionsNodes.front().first;
                destructionsNodes.pop_front();
            }
            else
            {
                toRemove = destructionsEdges.front().first;
                destructionsEdges.pop_front();
            }
            for (int k = 0; k < removed_indices.size(); k++)
            {
                if (toRemove >= removed_indices[k])
                {
                    toRemove++;
                }
            }
            toRemove = fixIndex(toRemove, solution.size());
            solution.erase(solution.begin() + toRemove);
            removed_indices.push_back(toRemove);
        }
        return solution;
    }
};

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
int N_TRIES = 20;
int main()
{
    string root_path = "../data/";
    vector<ProblemInstance> problemInstances = {TSPA, TSPB};
    vector<SearchType> searchTypes = {steepest};
    vector<InitialSolutionType> initialSolutionTypes = {randomAlg};
    vector<InterNeighbourhoodType> interNeighbourhoodTypes = {twoEdges};
    vector<bool> doLocalSearch = {true, false};

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
        for (auto searchType : searchTypes)
        {
            for (auto initialSolutionType : initialSolutionTypes)
            {
                for (auto interNeighbourhoodType : interNeighbourhoodTypes)
                {
                    for (auto doLocalSearch : doLocalSearch)
                    {
                        cout << "Name: " << LSNS(searchType, initialSolutionType, interNeighbourhoodType, distances, costs, 0, maxTimes[problemInstance], doLocalSearch).get_name() << endl;
                        cout << "Problem instance: " << ProblemInstanceStrings[problemInstance] << endl;
                        Result algoResult = Result(INT32_MAX, 0, 0, vector<int>(), vector<int>());
                        vector<double> times;
                        double avg_iterations_number = 0;

                        for (int i = 0; i < N_TRIES; i++)
                        {
                            cout << "Try: " << i << endl;
                            LSNS ls = LSNS(searchType, initialSolutionType, interNeighbourhoodType, distances, costs, -1, maxTimes[problemInstance], doLocalSearch);
                            clock_t start, end;
                            start = clock();
                            Result res = ls.solve();
                            end = clock();
                            vector<int> solution = res.bestSolution;
                            avg_iterations_number += res.averageCost; // iterations number
                            double time_taken = double(end - start) / double(CLOCKS_PER_SEC);
                            int cost = ls.calculate_cost(solution);
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
                        cout << "Best time: " << *min_element(times.begin(), times.end()) << endl;
                        cout << "Worst time: " << *max_element(times.begin(), times.end()) << endl;
                        cout << "Average time: " << accumulate(times.begin(), times.end(), 0.0) / times.size() << endl;
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
        }
    }
}