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

using namespace std;


auto rng = std::default_random_engine(869468469);

struct Result{
    int bestCost;
    int worstCost;
    int averageCost;
    vector<int> bestSolution;
    vector<int> worstSolution;

    Result(int bc, int wc, int ac, vector<int> bs, vector<int> ws)
        : bestCost(bc), worstCost(wc), averageCost(ac), bestSolution(bs), worstSolution(ws) {}
};

int calculate_cost(vector<int> solution, vector<vector<int>> distances, vector<int> costs){
        int cost = 0;
        for(int j=0; j<solution.size()-1; j++){
            cost += distances[solution[j]][solution[j+1]];
            cost+=costs[solution[j]];
        }
        cost+=distances[solution[solution.size()-1]][solution[0]];
        return cost;
    }


class Algo {
public:
    vector<vector<int>> distances;
    vector<int> costs;
    int starting_node;
    string name;
    Algo(vector<vector<int>> distances, vector<int> costs, int i, string name)
        : distances(distances), costs(costs), starting_node(i), name(name) {}
    virtual Result solve() =0;
    int calculate_cost(vector<int> solution, vector<vector<int>> distances, vector<int> costs){
        int cost = 0;
        for(int j=0; j<solution.size()-1; j++){
            cost += distances[solution[j]][solution[j+1]];
            cost+=costs[solution[j]];
        }
        cost+=distances[solution[solution.size()-1]][solution[0]];
        return cost;
    }
    string get_name(){
        return this->name;
    }
};

class RandomSearch : public Algo {
public:
    RandomSearch(vector<vector<int>> distances, vector<int> costs, int i)
        : Algo(distances, costs, i, "RandomSearch") {}
    
    Result solve() {
        vector<int> worstSolution;
        int solution_size = this->distances.size()/2;
        vector<int> current_solution = vector<int>(solution_size);
        vector<int> visited(this->distances.size());

        for(int j=0; j<solution_size; j++){
            if (j==0){
                current_solution[j] = this->starting_node;
                visited[this->starting_node] = true;
                continue;
            }
            int next = rand() % this->distances.size();
            while(visited[next])next = rand() % this->distances.size();
            current_solution[j] = next;
            visited[next]=true;
        }        
        return Result(0, 0, 0, current_solution, worstSolution);
    }
};

class GreedyCycle: public Algo {
public:
    GreedyCycle(vector<vector<int>> distances, vector<int> costs, int i)
        : Algo(distances, costs, i, "GreedyCycle") {}

    Result solve() {
        vector<int> worstSolution;
        int solution_size = this->distances.size()/2;
        vector<int> current_solution;
        vector<bool> visited(this->costs.size());
        current_solution.push_back(this->starting_node);
        visited[this->starting_node] = true;

        while(current_solution.size() < solution_size){
            int smallest_increase = INT32_MAX;
            int insert_index = -1;
            int insert_node = -1;


            for(int j=0; j<current_solution.size(); j++){  // Dla każdego nodea z cyklu
                int min_distance = INT32_MAX;
                int min_index = -1;
                for(int k=0; k<this->distances.size(); k++){ //znajdź najbliższy nieodwiedzony node
                    if(visited[k]) continue;
                    int curr = -this->distances[current_solution[j == 0 ? current_solution.size() - 1 : j - 1]][current_solution[j]] + this->distances[current_solution[j == 0 ? current_solution.size() - 1 : j - 1]][k] + this->distances[k][current_solution[j]] + this->costs[k];
                    if(curr < min_distance){
                        min_distance = curr;
                        min_index = k;
                    }
                }
                if(min_distance < smallest_increase){
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


class Greedy2RegretWieghted: public Algo {
public:
    Greedy2RegretWieghted(vector<vector<int>> distances, vector<int> costs, int i)
        : Algo(distances, costs, i, "Greedy2RegretWieghted") {}

    Result solve() {
        vector<int> worstSolution;
        int solution_size = distances.size()/2;
        vector<int> current_solution;
        current_solution.push_back(this->starting_node);
        vector<bool> visited(this->costs.size());
        visited[this->starting_node] = true;

        while(current_solution.size() < solution_size){
            
            int smallest_increase = INT32_MAX;
            int insert_index = -1;
            int insert_node = -1;
            int max_score = -INT32_MAX;


            for(int k=0; k<this->distances.size(); k++){ // dla wszystkich nieodwiedzonych nodeów
                if(visited[k]) continue;
                vector<int> insertion_cost_for_j;
                for(int j=0; j<current_solution.size(); j++){ // dla każdego nodea z cyklu
                    int curr = -this->distances[current_solution[j == 0 ? current_solution.size() - 1 : j - 1]][current_solution[j]] + this->distances[current_solution[j == 0 ? current_solution.size() - 1 : j - 1]][k] + this->distances[k][current_solution[j]] + this->costs[k];
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
                int insertion_cost = - this->distances[current_solution[left_node_idx]][current_solution[smallest_index]] + this->distances[current_solution[left_node_idx]][k] + this->distances[k][current_solution[smallest_index]] + this->costs[k];
                int score = regret - insertion_cost;
                if(score> max_score){
                    max_score = score;
                    insert_index = smallest_index;
                    insert_node = k;
                }
            }

            current_solution.insert(current_solution.begin() + insert_index, insert_node);
            visited[insert_node] = true; 
        }

        return Result(0, 0, 0, current_solution, worstSolution);
    }
};


template <typename T>
struct generator {
    struct promise_type;
    using handle_type = std::coroutine_handle<promise_type>;
    
    struct promise_type {
        T value;
        std::suspend_always initial_suspend() { return {}; }
        std::suspend_always final_suspend() noexcept { return {}; }
        generator get_return_object() { return generator{handle_type::from_promise(*this)}; }
        void unhandled_exception() { std::terminate(); }
        std::suspend_always yield_value(T val) {
            value = val;
            return {};
        }
    };

    bool move_next() { return coro ? (coro.resume(), !coro.done()) : false; }
    T current_value() { return coro.promise().value; }

    generator(generator const&) = delete;
    generator(generator&& other) : coro(other.coro) { other.coro = {}; }
    ~generator() { if (coro) coro.destroy(); }

private:
    generator(handle_type h) : coro(h) {}
    handle_type coro;
};


class Greedy2Regret: public Algo {
public:
    Greedy2Regret(vector<vector<int>> distances, vector<int> costs, int i) 
        : Algo(distances, costs, i, "Greedy2Regret"){}

    void write_vector_to_file(vector<int> sol){
        string filename = "animation_greedy/" + to_string(sol.size()) + ".csv";
        ofstream file;
        file.open(filename);
        for(int i=0; i<sol.size(); i++){
            file << sol[i] << endl;
        }
        file.close();
    }
    Result solve() {
        vector<int> worstSolution;
        int solution_size = distances.size()/2;
        vector<int> current_solution;
        current_solution.push_back(starting_node);
        vector<bool> visited(costs.size());
        visited[starting_node] = true;

        while(current_solution.size() < solution_size){
            
            int smallest_increase = INT32_MAX;
            int insert_index = -1;
            int insert_node = -1;
            int max_regret = -1;

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
                if(regret > max_regret){
                    max_regret = regret;
                    insert_index = smallest_index;
                    insert_node = k;
                }
            }
            current_solution.insert(current_solution.begin() + insert_index, insert_node);
            visited[insert_node] = true; 
            }
        
        return Result(0, 0, 0, current_solution, worstSolution);
    }
};


class NearestNeighboursSearch: public Algo {
public:
    NearestNeighboursSearch(vector<vector<int>> distances, vector<int> costs, int i)
        : Algo(distances, costs, i, "NearestNeighbours") {}
    
    Result solve() {
        vector<int> worstSolution;
        int solution_size = distances.size()/2;
        vector<int> current_solution;
        current_solution.push_back(starting_node);
        vector<bool> visited(costs.size());
        visited[starting_node] = true;
        while(current_solution.size() < solution_size){
            int min_cost = INT32_MAX;
            int min_index = -1;
            for(int j=0; j<distances.size(); j++){
                if(visited[j]) continue;
                if(distances[current_solution[current_solution.size()-1]][j] + costs[j] < min_cost){
                    min_cost = distances[current_solution[current_solution.size()-1]][j] + costs[j];
                    min_index = j;
                }
            }
            visited[min_index] = true;
            current_solution.push_back(min_index);
        }    
        return Result(0, 0, 0, current_solution, worstSolution);
    }
};

enum SearchType { greedy, steepest };
enum InitialSolutionType {randomAlg, GC, G2Rw, NN, G2R};
enum NeighbourhoodType {intra, inter};
enum InterNeighbourhoodType {twoNode, twoEdges};
enum ProblemInstance {TSPA, TSPB};

std::map<SearchType, std::string> SearchTypeStrings = {
    {greedy, "greedy"},
    {steepest, "steepest"}
};

std::map<InitialSolutionType, std::string> InitialSolutionTypeStrings = {
    {randomAlg, "random"},
    {GC, "GreedyCycle"},
    {G2Rw, "Greedy2RegretWeighted"},
    {NN, "NearestNeighbours"},
    {G2R, "Greedy2Regret"}
};

std::map<NeighbourhoodType, std::string> NeighbourhoodTypeStrings = {
    {intra, "intra"},
    {inter, "inter"}
};

std::map<InterNeighbourhoodType, std::string> InterNeighbourhoodTypeStrings = {
    {twoNode, "twoNode"},
    {twoEdges, "twoEdges"}
};

std::map<ProblemInstance, std::string> ProblemInstanceStrings = {
    {TSPA, "TSPA"},
    {TSPB, "TSPB"},
};

std::map<ProblemInstance, InitialSolutionType> BestInitialForInstance = {
    {TSPA, GC},
    {TSPB, GC},
};


class LocalSearch: public Algo {
public:
    SearchType searchType;
    InitialSolutionType initialSolutionType;
    InterNeighbourhoodType intraNeighbourhoodType;
    vector<bool> visited;
    int nPoints;
    LocalSearch(SearchType searchType, InitialSolutionType initialSolutionType, InterNeighbourhoodType intraNeighbourhoodType, vector<vector<int>> distances, vector<int> costs, int i)
        : Algo(distances, costs, i, "LocalSearch"), searchType(searchType), initialSolutionType(initialSolutionType), intraNeighbourhoodType(intraNeighbourhoodType) {
            this->name += "_" + SearchTypeStrings[searchType];
            this->name += "_" + InitialSolutionTypeStrings[initialSolutionType];
            this->name += "_" + InterNeighbourhoodTypeStrings[intraNeighbourhoodType];
            visited = vector<bool>(distances.size());
            nPoints = distances.size();
        }

    int calculate_cost(const vector<int>& solution){
        int cost = 0;
        for(int j=0; j<solution.size()-1; j++){
            cost += this->distances[solution[j]][solution[j+1]];
            cost+=this->costs[solution[j]];
        }
        cost+=this->distances[solution[solution.size()-1]][solution[0]];
        return cost;
    }

    Result solve() {
        vector<int> solution = getInitialSolution(this->initialSolutionType, starting_node);
        int solutionCost = this->calculate_cost(solution);
        for(int i=0; i<visited.size(); i++){
                visited[i] = false;
        }
        for(int i=0; i<solution.size(); i++){
            visited[solution[i]] = true;
        }
        while(true){
            //update visited
            for(int i=0; i<visited.size(); i++){
                visited[i] = false;
            }
            for(int i=0; i<solution.size(); i++){
                visited[solution[i]] = true;
            }
            shared_ptr<vector<int>> newSolution = search(solution, solutionCost);
            int newSolutionCost = calculate_cost(*newSolution);
            /////////////////////////////////////////////////////////////////////////////////////////
            //caclulate number of shared numbers in solution
            int shared = 0;
            bool temp_visited[nPoints];
            for(int i=0; i<nPoints; i++){
                temp_visited[i] = false;
            }
            for(int i=0; i<newSolution->size(); i++){
                if(!temp_visited[(*newSolution)[i]]){
                    shared++;
                    temp_visited[(*newSolution)[i]] = true;
                }
            }
            if(shared != distances.size()/2){
                cout << "Solution is not valid" << endl;
                cout << "Solution cost: " << newSolutionCost << endl;
                cout << "Solution size: " << newSolution->size() << endl;
                cout << "shared: " << shared << endl;
                cout << "Solution: ";
                for(int i=0; i<newSolution->size(); i++){
                    cout << (*newSolution)[i] << " ";
                }
                cout << endl;
                throw runtime_error("Solution is not valid");
            }
            /////////////////////////////////////////////////////////////////////////////////////////
            if(newSolutionCost == solutionCost) break;
            solution = *newSolution;
            solutionCost = newSolutionCost;
        }
        return Result(solutionCost, solutionCost, solutionCost, solution, solution);
    }

    vector<int> getInitialSolution(InitialSolutionType ist, int i){
        if(ist == randomAlg){
            RandomSearch rs = RandomSearch(distances, costs, i);
            return rs.solve().bestSolution;
        }
        if (ist == GC)
        {
            GreedyCycle gc = GreedyCycle(distances, costs, i);
            return gc.solve().bestSolution;

        }
        if (ist == G2Rw)
        {
            Greedy2RegretWieghted g2rw = Greedy2RegretWieghted(distances, costs, i);
            return g2rw.solve().bestSolution;
        }
    }

    shared_ptr<vector<int>> search(vector<int>& currentSolution, int currentSolutionCost){
        if(searchType == greedy) return greedySearch(currentSolution, currentSolutionCost);
        return steepestSearch(currentSolution, currentSolutionCost);
    }
    shared_ptr<vector<int>> greedySearch(vector<int>& currentSolution, int currentSolutionCost){
        auto neigbourhoodIterator = neighbourhoodGenerator(currentSolution);
        while(neigbourhoodIterator.move_next()){
            auto neighbour = neigbourhoodIterator.current_value();
            int neighbourCost = calculate_cost(*neighbour);
            if(neighbourCost < currentSolutionCost){
                return neighbour;
            }

        }
        return make_shared<vector<int>>(currentSolution);
    }

    shared_ptr<vector<int>> steepestSearch(vector<int>& currentSolution, int currentSolutionCost){
        auto neigbourhoodIterator = neighbourhoodGenerator(currentSolution);
        shared_ptr<vector<int>> bestNeighbour = make_shared<vector<int>>(currentSolution);
        int bestNeighbourCost = currentSolutionCost;
        while(neigbourhoodIterator.move_next()){
            auto neighbour = neigbourhoodIterator.current_value();
            int neighbourCost = calculate_cost(*neighbour);
            if(neighbourCost < bestNeighbourCost){
                bestNeighbour = make_shared<vector<int>>(*neighbour);
                bestNeighbourCost = neighbourCost;
            }

        }
        return bestNeighbour;
    }

    generator<shared_ptr<vector<int>>> neighbourhoodGenerator(vector<int>& currentSolution){
        vector<NeighbourhoodType> nTypeOrder = {intra, inter};
        shuffle(nTypeOrder.begin(), nTypeOrder.end(),rng);
        for(auto nType: nTypeOrder){
            if(nType == intra){
                auto intraNeighbourhoodIterator = intraNeighbourhoodGenerator(currentSolution);
                while(intraNeighbourhoodIterator.move_next()){
                    co_yield intraNeighbourhoodIterator.current_value();
                }
            } else {
                auto interNeighbourhoodIterator = interNeighbourhoodGenerator(currentSolution);
                while(interNeighbourhoodIterator.move_next()){
                    co_yield interNeighbourhoodIterator.current_value();
                }
            }
        }

    }

    generator<shared_ptr<vector<int>>> interNeighbourhoodGenerator(vector<int>& currentSolution){
        // co_yield currentSolution;
        shared_ptr<std::vector<int>> neighbour = make_shared<std::vector<int>>(currentSolution);
        vector<pair<int, int>> moves;
    
        for (int i = 0; i < currentSolution.size(); i++) {
            for (int j = 0; j < costs.size(); j++) {
                if (!visited[j]) {
                    moves.push_back({i, j});
                }
            }
        }

        std::shuffle(std::begin(moves), std::end(moves), rng);

        for (const auto& [i, j] : moves) {
            auto tmp = (*neighbour)[i];
            (*neighbour)[i] = j;
            co_yield neighbour;
            (*neighbour)[i] = tmp; //reversing change to save time copying memory
        }
    }

    generator<shared_ptr<vector<int>>> intraNeighbourhoodGenerator(vector<int>& currentSolution){
        if(intraNeighbourhoodType == twoNode) return intraNodesNeighbourhoodGenerator(currentSolution);
        return intraEdgesNeighbourhoodGenerator(currentSolution);
    }

    generator<shared_ptr<vector<int>>> intraNodesNeighbourhoodGenerator(vector<int>& currentSolution){
        shared_ptr<std::vector<int>> neighbour = make_shared<std::vector<int>>(currentSolution);
        vector<pair<int, int>> moves;
    
        for (int i = 0; i < currentSolution.size(); i++) {
            for(int j=i+1; j < currentSolution.size(); j++){
                moves.push_back({i, j});
            }
        }

        std::shuffle(std::begin(moves), std::end(moves), rng);

        for (const auto& [i, j] : moves) {
            auto tmp = (*neighbour)[i];
            (*neighbour)[i] = (*neighbour)[j];
            (*neighbour)[j] = tmp;
            co_yield neighbour;
            (*neighbour)[j] = (*neighbour)[i]; //reversing change to save time copying memory
            (*neighbour)[i] = tmp;
        }
    }
    
    generator<shared_ptr<vector<int>>> intraEdgesNeighbourhoodGenerator(vector<int>& currentSolution){
        shared_ptr<std::vector<int>> neighbour = make_shared<std::vector<int>>(currentSolution);
        vector<pair<pair<int, int>, pair<int, int>>> moves;
        for (int i = 0; i < neighbour->size(); i++) {
            for (int j = i + 2; j < neighbour->size(); j++) {
                moves.push_back({{i, i + 1}, {j, j == currentSolution.size() - 1 ? 0 : j + 1}});
            }
        }
        std::shuffle(std::begin(moves), std::end(moves), rng);

        for (const auto& [edge1, edge2] : moves) {
            if (edge1.second < edge2.second) {
            reverse(neighbour->begin() + edge1.second, neighbour->begin() + edge2.first + 1);
            co_yield neighbour;
            reverse(neighbour->begin() + edge1.second, neighbour->begin() + edge2.first + 1);
            // to save time copying memory
            }
            else if(edge2.second != 0){
                throw runtime_error("Incorrect edge indices: "  + to_string(edge1.second) + " " + to_string(edge2.second));
            }
        }
    }
};

vector<vector<int>> read_file(string filename) {
    vector<vector<int>> result;
    ifstream file(filename);
    string line;
    while (getline(file, line)) {
        vector<int> row;
        stringstream ss(line);
        string cell;
        while (getline(ss, cell, ';')) {
            row.push_back(stoi(cell));
        }
        result.push_back(row);
    }
    return result;
}


vector<vector<int>> calcDistances(vector<vector<int>> data){
    vector<vector<int>> distances;
    for (int i = 0; i < data.size(); i++){
        vector<int> row;
        for (int j = 0; j < data.size(); j++){
            int x = data[i][0] - data[j][0];
            int y = data[i][1] - data[j][1];
            //round to nearest int
            float distance = round(sqrt(x*x + y*y));
            row.push_back(distance);
        }
        distances.push_back(row);
    }
    return distances;
}


void write_solution_to_file(vector<int> sol, string algo_name, string data_name){
        // cout << "Writing to: " << algo_name + "_"+ data_name + ".csv" << endl;
        string filename = "results/" + algo_name + "_"+ data_name + ".csv";
        ofstream file;
        file.open(filename);
        for(int i=0; i<sol.size(); i++){
            file << sol[i] << endl;
        }
        file.close();
    }



int main1(){

    string root_path = "../data/";
    vector<ProblemInstance> problemInstances = {TSPA, TSPB};
    vector<SearchType> searchTypes = {greedy, steepest};
    vector<InitialSolutionType> initialSolutionTypes = {randomAlg, GC, G2Rw};
    vector<InterNeighbourhoodType> interNeighbourhoodTypes = {twoNode,twoEdges};


    for(auto problemInstance: problemInstances){
        string file = root_path + ProblemInstanceStrings[problemInstance] + ".csv";
        auto data = read_file(file);
        auto distances = calcDistances(data);
        vector<int> costs;
        for(int i=0; i< data.size(); i++){
            costs.push_back(data[i][2]);
        }
        for(auto searchType: searchTypes){
            for(auto initialSolutionType: initialSolutionTypes){
                for(auto interNeighbourhoodType: interNeighbourhoodTypes){
                    Result algoResult = Result(INT32_MAX, 0, 0, vector<int>(), vector<int>());
                    vector<double> times;
                    cout<<"Name: "<< LocalSearch(searchType, initialSolutionType, interNeighbourhoodType, distances, costs, 0).name << endl;
                    cout<<"Problem: "<< ProblemInstanceStrings[problemInstance] << endl;
                    for(int i=0; i<distances.size(); i++){
                        LocalSearch ls = LocalSearch(searchType, initialSolutionType, interNeighbourhoodType, distances, costs, i);
                        clock_t start, end;
                        start = clock();
                        vector<int> solution = ls.solve().bestSolution;
                        end = clock();
                        double time_taken = double(end - start) / double(CLOCKS_PER_SEC);
                        int cost = ls.calculate_cost(solution);
                        if(cost < algoResult.bestCost){
                            algoResult.bestCost = cost;
                            algoResult.bestSolution = solution;
                        }
                        if(cost > algoResult.worstCost){
                            algoResult.worstCost = cost;
                            algoResult.worstSolution = solution;
                        }
                        algoResult.averageCost += cost;
                        times.push_back(time_taken);
                    }
                    algoResult.averageCost /= distances.size();
                    cout << "Best cost: " << algoResult.bestCost << endl;
                    cout << "Worst cost: " << algoResult.worstCost << endl;
                    cout << "Average cost: " << algoResult.averageCost << endl;
                    cout << "Best time: " << *min_element(times.begin(), times.end()) << endl;
                    cout << "Worst time: " << *max_element(times.begin(), times.end()) << endl;
                    cout << "Average time: " << accumulate(times.begin(), times.end(), 0.0) / times.size() << endl;
                    cout << "Best solution: ";
                    for(int i=0; i<algoResult.bestSolution.size(); i++){
                        cout << algoResult.bestSolution[i] << " ";
                    }
                    cout << endl;
                    LocalSearch ls = LocalSearch(searchType, initialSolutionType, interNeighbourhoodType, distances, costs, 0);
                    write_solution_to_file(algoResult.bestSolution, ls.get_name(), ProblemInstanceStrings[problemInstance]);
                }
            }
        }

    }
}

    Result solveAlgo(InitialSolutionType algo, vector<vector<int>> distances, vector<int> costs, int i){
        if(algo == randomAlg){
            return RandomSearch(distances, costs, i).solve();
        }
        if (algo == GC)
        {
            return GreedyCycle(distances, costs, i).solve();
        }
        if (algo == G2Rw)
        {
            return Greedy2RegretWieghted(distances, costs, i).solve();
        }
        if (algo == NN)
        {
            return NearestNeighboursSearch(distances, costs, i).solve();
        }
        if (algo == G2R)
        {
            return Greedy2Regret(distances, costs, i).solve();
        }
    }

    int main2(){
        string root_path = "../data/";
        vector<ProblemInstance> problemInstances = {TSPA, TSPB};
        vector<InitialSolutionType> algos = {randomAlg, GC, G2Rw, NN, G2R};

        for(auto problemInstance: problemInstances){
            string file = root_path + ProblemInstanceStrings[problemInstance] + ".csv";
            auto data = read_file(file);
            auto distances = calcDistances(data);
            vector<int> costs;
            for(int i=0; i< data.size(); i++){
                costs.push_back(data[i][2]);
            }
            for(auto algo : algos){
                Result algoResult = Result(INT32_MAX, 0, 0, vector<int>(), vector<int>());
                double averageTime = 0;
                cout<<"Name: "<< InitialSolutionTypeStrings[algo] << endl;
                cout<<"Problem: "<< ProblemInstanceStrings[problemInstance] << endl;
                for(int i=0; i<distances.size(); i++){
                    clock_t start, end;
                    start = clock();
                    Result subResult = solveAlgo(algo, distances, costs, i);
                    end = clock();
                    vector<int> solution = subResult.bestSolution;
                    double time_taken = double(end - start) / double(CLOCKS_PER_SEC);
                    int cost = calculate_cost(solution, distances, costs);
                    if(cost < algoResult.bestCost){
                        algoResult.bestCost = cost;
                        algoResult.bestSolution = solution;
                    }
                    if(cost > algoResult.worstCost){
                        algoResult.worstCost = cost;
                        algoResult.worstSolution = solution;
                    }
                    algoResult.averageCost += cost;
                    averageTime += time_taken;
                }
                algoResult.averageCost /= distances.size();
                cout << "Best cost: " << algoResult.bestCost << endl;
                cout << "Worst cost: " << algoResult.worstCost << endl;
                cout << "Average cost: " << algoResult.averageCost << endl;
                averageTime /= distances.size();
                cout << "Average time: " << averageTime << endl;
                cout << "Best solution: ";
                for(int i=0; i<algoResult.bestSolution.size(); i++){
                    cout << algoResult.bestSolution[i] << " ";
                }
                cout << endl;
            }
        }
    }

    int main(){
        main1();
        main2();
    }