// connect.cpp
//
// Accelerated network generation functions

#include "func_connect.h"


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
size_t _unique_1d(std::vector<size_t>& a)
{
    std::unordered_map<size_t, size_t> hash_map;
    size_t number;
    size_t total_unique = 0;

    for (size_t i = 0; i < a.size(); i++)
    {
        number = a[i];
        // check if this number is already in the map
        if (hash_map.find(number) == hash_map.end())
        {
            // it's not in there yet so add it and set the count to 1
            hash_map.insert({number, 1});
            a[total_unique] = a[i];
            total_unique += 1;
        }
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
size_t _unique_2d(std::vector< std::vector<size_t> >& a)
{
    map_t hash_map; // not working
    size_t total_unique = 0;
    size_t num_edges = a[0].size();

    edge_t edge;
    for (size_t i = 0; i < num_edges; i++)
    {
        edge = edge_t(a[0][i], a[1][i]);
        // check if this number is already in the map
        if (hash_map.find(edge) == hash_map.end())
        {
            // it's not in there yet so add it and set the count to 1
            hash_map.insert({edge, 1});
            a[0][total_unique] = a[0][i];
            a[1][total_unique] = a[1][i];
            total_unique += 1;
        }
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
std::vector<size_t> _gen_edge_complement(
  long seed, std::vector<size_t> nodes, size_t other_end, size_t degree,
  const std::vector< std::vector<size_t> >* existing_edges, bool multigraph)
{
    // Initialize the RNG
    std::mt19937 generator_(seed);
    size_t min_idx = *std::min_element(nodes.begin(), nodes.end());
    size_t max_idx = *std::max_element(nodes.begin(), nodes.end());
    std::uniform_int_distribution<size_t> uniform_(min_idx, max_idx);

    // generate the complements
    std::vector<size_t> result;
    size_t ecurrent = 0;

    // check the existing edges
    const size_t num_old_edges = existing_edges ? existing_edges[0].size() : 0;
    for (size_t i=0; i < num_old_edges; i++)
    {
        if (existing_edges->at(0)[i] == other_end)
        {
            result.push_back(existing_edges->at(1)[i]);
        }
    }
    ecurrent = result.size();
    result.resize(ecurrent + degree);
    
    size_t remaining = degree;
    size_t cplt, j;
    const size_t target_degree = ecurrent + degree;
    while (ecurrent < target_degree)
    {
        remaining = target_degree - ecurrent;
        j = 0;
        while (j < remaining)
        {
            cplt = uniform_(generator_);
            if (cplt != other_end)
            {
                result[ecurrent + j] = cplt;
                j++;
            }
        }
        // update ecurrent and (potentially) the results
        ecurrent = multigraph ? target_degree : _unique_1d(result);
    }

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
 * \return result        - The desired vector of complementary nodes.
 */
std::vector< std::vector<size_t> > _gen_edges(
  const std::vector<size_t>& first_nodes, const std::vector<size_t>& degrees,
  const std::vector<size_t>& second_nodes,
  const std::vector< std::vector<size_t> >& existing_edges,
  bool multigraph, bool directed, long msd, size_t omp)
{
    // Initialize secondary seeds
    std::vector<long> seeds(omp, msd);
    for (size_t i=0; i < omp; i++)
    {
        seeds[i] += i + 1;
    }

    // compute the cumulated sum of the degrees
    std::vector<size_t> cum_degrees(degrees.size());
    std::partial_sum(degrees.begin(), degrees.end(), cum_degrees.begin());

    // generate the edges
    std::vector< std::vector<size_t> > edges;
    #pragma omp parallel for schedule(dynamic)
    for (size_t node=0; node < first_nodes.size(); node++)
    {
        // generate the vector of complementary nodes
        std::vector<size_t> res_tmp = _gen_edge_complement(
          seeds[node], second_nodes, node, degrees[node], &existing_edges,
          multigraph);
        // fill the edges
        size_t idx_start = cum_degrees[node];
        for (size_t j = 0; j < res_tmp.size(); j++)
        {
            edges[0][idx_start + j] = node;
            edges[1][idx_start + j] = res_tmp[j];
        }
    }

    return edges;
}
    
}
