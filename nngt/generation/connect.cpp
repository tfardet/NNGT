// connect.cpp
//
// Accelerated network generation functions

#include <random>
#include <vector>
#include <array>
#include <unordered_map>
#include <algorithm>


namespace generation {

/*
 * Sort a vector to move the N unique numbers in the N first entries.
 *
 * \param a - Source array.
 * 
 * .. note::
 *   The array is modified inplace.
 *
 * \return num_unique - Number of unique entries.
 */
size_t _unique_1d(vector<int>& a)
{
    unordered_map<int, int> hash_map;
    int number;
    size_t total_unique = 0

    for (size_t i = 0; i < a.size(); i++)
    {
        number = a[i];
        // check if this number is already in the map
        if (hash_map.find(number) == hash_map.end())
        {
            // it's not in there yet so add it and set the count to 1
            hash_map.insert(number, 1);
            a[total_unique] = a[i];
            total_unique += 1;
        }

    return total_unique;
}


/*
 * Sort a 2-D vector to move the N unique pairs in the N first columns.
 *
 * \param a - Source array.
 * 
 * .. note::
 *   The array is modified inplace.
 *
 * \return num_unique - Number of unique entries.
 */
size_t _unique_2d(vector< vector<int> >& a)
{
    unordered_map< array<int, 2>, int > hash_map;
    array<int, 2> edge;
    size_t total_unique = 0
    size_t num_edges = a[0].size()

    for (size_t i = 0; i < size; i++)
    {
        edge[0] = a[0][i];
        edge[1] = a[1][i];
        // check if this number is already in the map
        if (hash_map.find(edge) == hash_map.end())
        {
            // it's not in there yet so add it and set the count to 1
            hash_map.insert(edge, 1);
            a[0][total_unique] = a[0][i];
            a[1][total_unique] = a[1][i];
            total_unique += 1;
        }

    return total_unique;
}


/*
 * Generate the complementary nodes for desired edges.
 *
 * \param seed           - Seed for the random number generator.
 * \param nodes          - Population from whch to draw the complementary end of the edges.
 * \param other_end      - Node at the other end of the edges.
 * \param degree         - Degree of node `other_end` (length of the returned array)
 * \param existing_edges - 2D-array containing the existing edges.
 * \param multigraph     - Whether multiple edges are allowed.
 *
 * \return result        - The desired vecotr of complementary nodes.
 */
vector<int> _gen_edge_complement(
  long seed, vector<size_t> nodes, size_t other_end, size_t degree,
  vector< vector<size_t> > existing_edges, bool multigraph)
{
    // Initialize the RNG
    std::mt19937 generator_(seed);
    std::uniform_int_distribution<size_t> uniform_(std::min(nodes), std::max(nodes));

    // generate the complements
    vector<size_t> result(degree);
    

    return result;
}


/*
 * Generate random edges from a list of nodes, their target degrees, and a
 * second population of nodes.
 *
 * \param first_nodes    - Population the degree of which is known.
 * \param degrees        - Degree of each node in `first_nodes`.
 * \param second_nodes   - Population from whch to draw the complementary end of the edges.
 * \param existing_edges - 2D-array containing the existing edges.
 * \param multigraph     - Whether multiple edges are allowed.
 * \param directed       - Whether the edges are directed or not.
 * \param msd            - Master seed.
 * \param omp            - Number of OpenMP threads.
 *
 * \return result        - The desired vecotr of complementary nodes.
 */
vector< vector<size_t> > _gen_edges(
  vector<size_t> first_nodes, vector<size_t> degrees,
  vector<size_t> second_nodes, vector< vector<size_t> > existing_edges,
  bool multigraph, bool directed, long seed, size_t omp)
{
    // Initialize the RNG
    std::mt19937 generator_(msd);
    std::uniform_int_distribution<size_t> uniform_(std::min(nodes), std::max(nodes));

    // generate the complements
    vector< vector<size_t> > edges;

    return edges;
}
    
}
