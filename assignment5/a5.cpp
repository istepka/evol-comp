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
#include <typeinfo>
#include<unistd.h>
#include <set>

using namespace std;

bool DOINTRA = true;
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
        }
        cost+=distances[solution[solution.size()-1]][solution[0]];
        for(int j=0;j<solution.size();j++){
            cost+=costs[solution[j]];
        }
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

enum SearchType { greedy, steepest };
enum InitialSolutionType {randomAlg, GC, G2Rw};
enum NeighbourhoodType {intra, inter};
enum InterNeighbourhoodType {twoNode, twoEdges};
enum ProblemInstance {TSPA, TSPB, TSPC, TSPD};

std::map<SearchType, std::string> SearchTypeStrings = {
        {greedy, "greedy"},
        {steepest, "steepest"}
};

std::map<InitialSolutionType, std::string> InitialSolutionTypeStrings = {
        {randomAlg, "random"},
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
        {TSPC, "TSPC"},
        {TSPD, "TSPD"}
};

std::map<ProblemInstance, InitialSolutionType> BestInitialForInstance = {
        {TSPA, G2Rw},
        {TSPB, G2Rw},
        {TSPC, GC},
        {TSPD, G2Rw}
};



enum MoveEvaluationResult {doMove, doReversed,  removeMove, skipMove};

class LocalSearch: public Algo {
public:
    SearchType searchType;
    InitialSolutionType initialSolutionType;
    InterNeighbourhoodType intraNeighbourhoodType;
    vector<bool> visited;
    int nPoints;
    LocalSearch(SearchType searchType, InitialSolutionType initialSolutionType, InterNeighbourhoodType intraNeighbourhoodType, vector<vector<int>> distances, vector<int> costs, int i)
            : Algo(distances, costs, i, "LS"), searchType(searchType), initialSolutionType(initialSolutionType), intraNeighbourhoodType(intraNeighbourhoodType) {
        this->name += "_" + SearchTypeStrings[searchType];
        this->name += "_" + InitialSolutionTypeStrings[initialSolutionType];
        this->name += "_" + InterNeighbourhoodTypeStrings[intraNeighbourhoodType];
        this->name += "Candidate";
        visited = vector<bool>(distances.size());
        nPoints = distances.size();
    }

    int calculate_cost(const vector<int>& solution){
        int cost = 0;
        for(int j=0; j<solution.size()-1; j++){
            cost += this->distances[solution[j]][solution[j+1]];
        }
        cost+=this->distances[solution[solution.size()-1]][solution[0]];
        for(int j=0;j<solution.size();j++){
            cost+=this->costs[solution[j]];
        }
        return cost;
    }

    int fixIndex(int index, int solutionSize){
        if(index < 0){
            return solutionSize + index;
        }
        if(index >= solutionSize){
            return index - solutionSize;
        }
        return index;
    }

    Result solve() {
        vector<int> solution = getInitialSolution(this->initialSolutionType, starting_node);
        for(int i=0; i<visited.size(); i++){
            visited[i] = false;
        }
        for(int i=0; i<solution.size(); i++){
            visited[solution[i]] = true;
        }
        multiset<pair<int, vector<int>>> LM;
        auto neighbourhoodIterator = neighbourhoodGenerator(solution);
        while(neighbourhoodIterator.move_next()){
            vector<int> move = neighbourhoodIterator.current_value();
            int delta = calculateDelta(solution, move);
            if(delta < 0){
                LM.insert(make_pair(delta, move));
            }
        }

        while(LM.size() > 0){
            auto lmIt = LM.begin();
            vector<int> selectedMove;
            while(lmIt != LM.end()){
                auto moveEvaluationResult = evaluateMove(solution, lmIt->second);
                if(moveEvaluationResult == skipMove){
                    lmIt++;
                }else if(moveEvaluationResult == removeMove){
                    lmIt = LM.erase(lmIt);
                }else{
                    selectedMove = lmIt->second;
                    if(moveEvaluationResult == doReversed){
                        std::swap(selectedMove[4], selectedMove[5]);
                        std::swap(selectedMove[6], selectedMove[7]);
                    }
                    break;
                }
            }
            if(lmIt == LM.end()){
                break;
            }
            int cost_before = calculate_cost(solution);
            applyMove(&solution, selectedMove);
            int cost_after = calculate_cost(solution);
            if(cost_after-cost_before != lmIt->first){
                cout << "Wrong delta calculation" << endl;
            }
            vector<vector<int>> newMoves;
            generateNewMoves(newMoves, solution, selectedMove);
            for(auto newMove: newMoves){
                int delta = calculateDelta(solution, newMove);
                if(delta < 0){
                    LM.insert(make_pair(delta, newMove));
                }
            }
            if(!validateSolution(solution)){
                cout<< "Solution not valid" << endl;
            }

        }
        return Result(calculate_cost(solution), 0, 0, solution, vector<int>());
    }

    bool validateSolution(vector<int>& solution){
        bool temp_visited[distances.size()];
        for(int i=0; i<distances.size(); i++){
            temp_visited[i] = false;
        }
        for(int i=0; i<solution.size(); i++){
            if(temp_visited[solution[i]]){
                return false;
            }
            temp_visited[solution[i]] = true;
        }
        return true;
    }

    vector<int> getInitialSolution(InitialSolutionType ist, int i){
        if(ist == randomAlg){
            RandomSearch rs = RandomSearch(distances, costs, i);
            return rs.solve().bestSolution;
        }
    }

    int calculateDelta(vector<int>& solution, vector<int>& move){
        int delta;
        if(move.size() == 5){
            // exchange nodes
            int i = move[0];
            int new_node = move[1];
            int old_node = solution[i];
            int oldCost = costs[old_node] + distances[old_node][solution[fixIndex(i+1,solution.size())]] + distances[old_node][solution[fixIndex(i-1+solution.size(), solution.size())]];
            int newCost = costs[new_node] + distances[new_node][solution[fixIndex(i+1,solution.size())]] + distances[new_node][solution[fixIndex(i-1+solution.size(), solution.size())]];
            delta = newCost - oldCost;
        }
        else if(move.size() == 8){
            // edge exchnge
            int edge1_first = solution[move[0]];
            int edge1_second = solution[move[1]];
            int edge2_first = solution[move[2]];
            int edge2_second = solution[move[3]];
            int oldCost = distances[edge1_first][edge1_second] + distances[edge2_first][edge2_second];
            int newCost = distances[edge1_first][edge2_first] + distances[edge1_second][edge2_second];
            delta = newCost - oldCost;
        }
        else{
            throw runtime_error("Wrong size of best move");
        }
        return delta;
    }

    MoveEvaluationResult evaluateMove(vector<int>& solution, const vector<int>& move){
        if(move.size() == 5){
            //check if can do and then if delta calculation is still valid
            if(visited[move[1]] || solution[move[0]] != move[2] || solution[fixIndex(move[0]-1,solution.size())] != move[3] || solution[fixIndex(move[0]+1, solution.size())] != move[4]){
                return removeMove;
            }
            return doMove;
        }else if(move.size() == 8){//edge move
            //if all nodes are the same on the same positions return doMove
            if(solution[move[0]] == move[4] && solution[move[1]] == move[5] && solution[move[2]] == move[6] && solution[move[3]] == move[7]){
                return doMove;
            }
            //if all nodes are the same but reversed do rewerse move
            if(solution[move[0]] == move[5] && solution[move[1]] == move[4] && solution[move[2]] == move[7] && solution[move[3]] == move[6]){
                //swap move[4] move[5] and move[6] move[7]
                return doReversed;
            }
            //removed edges occur in the current solution in a different direction from the saved one – not applicable now but the move can be applied in the future - skip
            if(solution[move[0]] == move[5] && solution[move[1]] == move[4] || solution[move[2]] == move[7] && solution[move[3]] == move[6]){
                return skipMove;
            }
            //else removeMove
            return removeMove;

        }else{
            throw runtime_error("Wrong size of move");
        }
    }

    void applyMove(vector<int>* solution, const vector<int>& move){//modifies solution i think usuall  poinnter is enuogh
        if(move.size() == 5){
            int i = move[0];
            int j = move[1];
            (*solution)[i] = j;
            visited[move[2]] = false;
            visited[j] = true;
        }else if(move.size() == 8){
            int j = move[1];
            int k = move[2];
            reverse(solution->begin() + j, solution->begin() + k + 1);
        }
    }

    void generateVisitedToNonVisited(vector<vector<int>>& newMoves, vector<int> & solution, int nodeId){
        for(int i=0; i<distances.size(); i++){
            if(!visited[i]){
                newMoves.push_back(makeInterMove(nodeId, i, solution[nodeId], solution[fixIndex(nodeId-1, solution.size())], solution[fixIndex(nodeId+1, solution.size())]));
            }
        }
    }

    void fromEdgeToAllOtherEdges(vector<vector<int>>& newMoves, vector<int> & solution, int edge_first){
        for(int i=0; i<edge_first-1; i++){
            newMoves.push_back(makeIntraMove(i, fixIndex(i+1, solution.size()), edge_first, fixIndex(edge_first+1, solution.size()), solution[i], solution[fixIndex(i+1, solution.size())], solution[edge_first], solution[fixIndex(edge_first+1, solution.size())]));
        }
        for(int i=edge_first+2; i<solution.size(); i++){
            newMoves.push_back(makeIntraMove(edge_first, fixIndex(edge_first+1, solution.size()), i, fixIndex(i+1, solution.size()), solution[edge_first], solution[fixIndex(edge_first+1, solution.size())], solution[i], solution[fixIndex(i+1, solution.size())]));
        }

    }

    void generateNewMoves(vector<vector<int>>& newMoves, vector<int> & solution, const vector<int>& move) {
        if (move.size() == 5) {
//            throw runtime_error("move 5 should not happen");
            int currentNodeId = move[0];
            int currentNode = move[2];
            int previousNodeIndex = fixIndex(move[0] - 1, solution.size());
            int previousNode = solution[previousNodeIndex];//could take fro move cause they do not change
            int nextNodeIndex = fixIndex(move[0] + 1, solution.size());
            int nextNode = solution[nextNodeIndex];
            int newNode = move[1];
            generateVisitedToNonVisited(newMoves, solution, currentNodeId);
            generateVisitedToNonVisited(newMoves, solution, previousNodeIndex);
            generateVisitedToNonVisited(newMoves, solution, nextNodeIndex);
            //generate moves from previously inserted node to all already in solution
            for (int i = 0; i < solution.size(); i++) {
                if (i == move[0]) continue;//skip the node that was just inserted
                newMoves.push_back(
                        makeInterMove(i, currentNode, solution[i], solution[fixIndex(i - 1, solution.size())],
                                      solution[fixIndex(i + 1, solution.size())]));
            }
            //edges moves
            //we have new edge from previous to new node and from new to next node
            //previous to new
            if(DOINTRA) {
                fromEdgeToAllOtherEdges(newMoves, solution, previousNodeIndex);
                fromEdgeToAllOtherEdges(newMoves, solution, currentNodeId);
            }

        } else if (move.size() == 8) {
            //throw runtime_error("move 8 should not happen");
            //gnerate moves from new edges to all already in solution
            //generate for edge1
            for(int i=move[0]; i <= move[2]; i++) {
                fromEdgeToAllOtherEdges(newMoves, solution, i);
            }
            //generate for edge2
//            vector<vector<int>> generatedEdge2 = fromEdgeToAllOtherEdges(newMoves, solution, move[2]);
            //add missing inter moves
            generateVisitedToNonVisited(newMoves, solution, move[0]);
            generateVisitedToNonVisited(newMoves, solution, move[1]);
            generateVisitedToNonVisited(newMoves, solution, move[2]);
            generateVisitedToNonVisited(newMoves, solution, move[3]);
        } else {
            throw runtime_error("Wrong size of move");
        }
    }

    generator<vector<int>> neighbourhoodGenerator(vector<int>& currentSolution){
        if(DOINTRA) {
            auto intraNeighbourhoodIterator = intraEdgesNeighbourhoodGenerator(currentSolution);
            while (intraNeighbourhoodIterator.move_next()) {
                co_yield
                intraNeighbourhoodIterator.current_value();
            }
        }
        auto interNeighbourhoodIterator = interNeighbourhoodGenerator(currentSolution);
        while(interNeighbourhoodIterator.move_next()){
            co_yield interNeighbourhoodIterator.current_value();
        }

    }

    generator<vector<int>> interNeighbourhoodGenerator(vector<int>& currentSolution){
        vector<int> move = {0, 0};
        for (int i = 0; i < currentSolution.size(); i++) {
            int currentnode = currentSolution[i];
            for (int j = 0; j < distances.size(); j++) {
                if (!visited[j]) {
                    co_yield makeInterMove(i, j, currentSolution[i], currentSolution[fixIndex(i-1,currentSolution.size())], currentSolution[fixIndex(i+1, currentSolution.size())]);
                }
            }
        }
    }

    vector<int> makeInterMove(int currentNodeId, int newNode, int currentNode, int previousNode, int nextNode){
        return {currentNodeId, newNode, currentNode, previousNode, nextNode};
    }


    generator<vector<int>> intraEdgesNeighbourhoodGenerator(vector<int>& currentSolution){
        vector<int> temp_vec = {0, 0, 0, 0};
        vector<int> move = vector<int>(temp_vec);
        for (int i = 0; i < currentSolution.size(); i++) {
            int node1 = currentSolution[i];
            int node1_next = currentSolution[fixIndex(i + 1, currentSolution.size())];
            for (int j = i + 2; j < currentSolution.size(); j++) {
                int node2 = currentSolution[j];
                int node2_next = currentSolution[fixIndex(j + 1, currentSolution.size())];
                co_yield makeIntraMove(i, i+1, j, fixIndex(j + 1, currentSolution.size()), node1, node1_next, node2, node2_next);
            }
        }
    }

    vector<int> makeIntraMove(int edge1_first, int edge1_second, int edge2_first, int edge2_second, int edge1_first_node, int edge1_second_node, int edge2_first_node, int edge2_second_node){
        return {edge1_first, edge1_second, edge2_first, edge2_second, edge1_first_node, edge1_second_node, edge2_first_node, edge2_second_node};
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
    string filename = "results/" + algo_name + "_"+ data_name + ".csv";
    ofstream file;
    file.open(filename);
    for(int i=0; i<sol.size(); i++){
        file << sol[i] << endl;
    }
    file.close();
}


int main(){
    string root_path = "../data/";
    vector<ProblemInstance> problemInstances = {TSPA, TSPB, TSPC, TSPD};
    vector<SearchType> searchTypes = {steepest};
    vector<InitialSolutionType> initialSolutionTypes = {randomAlg};
    vector<InterNeighbourhoodType> interNeighbourhoodTypes = {twoEdges};


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
                    cout << "Problem instance: " << ProblemInstanceStrings[problemInstance] << endl;
                    cout << "Name: " << LocalSearch(searchType, initialSolutionType, interNeighbourhoodType, distances, costs, 0).name << endl;
                    Result algoResult = Result(INT32_MAX, 0, 0, vector<int>(), vector<int>());
                    double averageTime = 0;
                    for(int i=0; i<distances.size(); i++){
                        LocalSearch ls = LocalSearch(searchType, initialSolutionType, interNeighbourhoodType, distances, costs, i);
                        clock_t start, end;
                        start = clock();
                        vector<int> solution = ls.solve().bestSolution;
                        end = clock();
//                        cout << i << " " << ls.calculate_cost(solution) << endl;
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
                    LocalSearch ls = LocalSearch(searchType, initialSolutionType, interNeighbourhoodType, distances, costs, 0);
                    write_solution_to_file(algoResult.bestSolution, ls.get_name(), ProblemInstanceStrings[problemInstance]);
                }
            }
        }

    }

}