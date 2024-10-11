# Evolutionary computation class
Authors: Ignacy Stępka, Michał Wiliński

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

The report should contain:
- Description of the problem
- Pseudocode of all implemented algorithms
- Results of a computational experiment: for each instance and method min, max and average
value of the objective function.
- 2D visualization of the best solution for each instance and method. Cost of nodes should be
presented e.g. by a color, greyscale, or size.
- The best solutions for each instance and method presented as a list of nodes indices (starting
from 0).
- Information whether the best solutions have been checked with the solution checker.
- (Link to) the source code
- Conclusions