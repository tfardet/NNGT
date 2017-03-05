// connect.cpp
//
// Accelerated network generation functions

#include "func_connect.h"

#include <omp.h>

#define _USE_MATH_DEFINES
#include <cmath>
#include <limits>
#include <random>

#include <assert.h>


namespace generation {

void _init_seeds(std::vector<long>& seeds, unsigned int omp, long msd)
{
    for (size_t i=0; i < omp; i++)
    {
        seeds[i] = msd + i + 1;
    }
}

size_t _unique_1d(std::vector<size_t>& a,
                  std::unordered_map<size_t, size_t>& hash_map)
{
    size_t number;
    size_t total_unique = hash_map.size();

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

size_t _unique_2d(std::vector< std::vector<size_t> >& a, map_t& hash_map)
{
    size_t total_unique = hash_map.size();
    size_t num_edges = a[0].size();
    edge_t edge;

    for (size_t i = total_unique; i < num_edges; i++)
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
    // this is required to avoid parasitic values
    a[0].resize(total_unique);
    a[1].resize(total_unique);

    return total_unique;
}

std::vector<size_t> _gen_edge_complement(
  std::mt19937& generator, const std::vector<size_t>& nodes, size_t other_end,
  size_t degree, const std::vector< std::vector<size_t> >* existing_edges,
  bool multigraph)
{
    // Initialize the RNG
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
    std::unordered_map<size_t, size_t> hash_map;
    
    assert(target_degree == degree);
    
    while (ecurrent < target_degree)
    {
        remaining = target_degree - ecurrent;
        j = 0;
        while (j < remaining)
        {
            cplt = uniform_(generator);
            if (cplt != other_end)
            {
                result[ecurrent + j] = cplt;
                j++;
            }
        }
        // update ecurrent and (potentially) the results
        ecurrent = multigraph ? target_degree : _unique_1d(result, hash_map);
    }

    return result;
}

void _gen_edges(
  size_t* ia_edges, const std::vector<size_t>& first_nodes,
  const std::vector<size_t>& degrees, const std::vector<size_t>& second_nodes,
  const std::vector< std::vector<size_t> >& existing_edges, unsigned int idx,
  bool multigraph, bool directed, long msd, unsigned int omp)
{
    // Initialize secondary seeds
    std::vector<long> seeds(omp);
    _init_seeds(seeds, omp, msd);

    // compute the cumulated sum of the degrees
    std::vector<size_t> cum_degrees(degrees.size());
    std::partial_sum(degrees.begin(), degrees.end(), cum_degrees.begin());

    // generate the edges
    #pragma omp parallel num_threads(omp)
    {
        std::mt19937 generator_(seeds[omp_get_thread_num()]);
        
        #pragma omp for schedule(static)
        for (size_t node=0; node < first_nodes.size(); node++)
        {
            // generate the vector of complementary nodes
            std::vector<size_t> res_tmp = _gen_edge_complement(
              generator_, second_nodes, node, degrees[node], &existing_edges,
              multigraph);
            // fill the edges
            size_t idx_start = cum_degrees[node] - degrees[node];
            for (size_t j = 0; j < degrees[node]; j++)
            {
                ia_edges[2*(idx_start + j) + idx] = node;
                ia_edges[2*(idx_start + j) + 1 - idx] = res_tmp[j];
            }
        }
    }
}


/*
* Distance-rule algorithms
*/

double _proba(int rule, double scale, double distance)
{
    if (rule == 0) // linear
        return 2 * std::max(0., 1. - distance/scale) / scale;
    else
        return std::exp(-distance / scale) / scale;
}

void _cdistance_rule(size_t* ia_edges, const std::vector<size_t>& source_nodes,
  const std::vector<size_t>& target_nodes, const std::string& rule,
  double scale, const std::vector<double>& x, const std::vector<double>& y,
  double area, size_t num_neurons, size_t num_edges,
  const std::vector< std::vector<size_t> >& existing_edges, bool multigraph,
  long msd, unsigned int omp)
{
    // Initialize secondary seeds and RNGs
    std::vector<long> seeds(omp);
    _init_seeds(seeds, omp, msd);

    size_t min_src = *std::min_element(
        source_nodes.begin(), source_nodes.end());
    size_t max_src = *std::max_element(
        source_nodes.begin(), source_nodes.end());
    size_t min_tgt = *std::min_element(
        target_nodes.begin(), target_nodes.end());
    size_t max_tgt = *std::max_element(
        target_nodes.begin(), target_nodes.end());
    std::uniform_int_distribution<size_t> rnd_source(min_src, max_src);
    std::uniform_int_distribution<size_t> rnd_target(min_tgt, max_tgt);
    std::uniform_real_distribution<double> rnd_uniform(0., 1.);
    
    // initialize edge container and hash map to check uniqueness
    std::vector< std::vector<size_t> > edges_tmp(2, std::vector<size_t>());
    if (!existing_edges.empty())
    {
        edges_tmp.push_back(existing_edges[0]);
        edges_tmp.push_back(existing_edges[1]);
    }
    map_t hash_map;
    
    // unordered map to translate rule into int
    std::unordered_map<std::string, int> r_to_int = {{"lin", 0}, {"exp", 1}};
    int rule_type = r_to_int[rule];

    size_t initial_enum = existing_edges.empty() ?
        0 : existing_edges[0].size(); // initial number of edges
    size_t current_enum = initial_enum;             // current number of edges
    size_t target_enum = current_enum + num_edges;  // target number of edges
    
    edges_tmp[0].reserve(target_enum);
    edges_tmp[1].reserve(target_enum);
    
    // estimate the number of tests that should be necessary
    //~ double avg_distance = sqrt(area / num_neurons);
    double typical_distance = sqrt(area);
    double avg_distance = typical_distance * sqrt(M_PI / 2.);
    double avg_proba = _proba(rule_type, scale, avg_distance);
    //~ double avg_proba = avg_distance * std::exp(-avg_distance*avg_distance /
        //~ (4*typical_distance*typical_distance)) * _proba(rule_type, scale,
        //~ avg_distance) / (typical_distance*typical_distance);
    double proba_c = num_edges / ((double) num_neurons * (num_neurons - 1));
    size_t num_tests = avg_proba <= proba_c ? num_neurons * (num_neurons - 1)
                                            : num_edges / avg_proba;
    if (current_enum != 0)
    {
        num_tests *=
            1. - existing_edges.size() / (num_neurons * (num_neurons - 1));
    }

    printf("%f %f %f %lu; Num tests: %lu\n", proba_c, avg_distance, avg_proba, num_edges, num_tests);
    size_t ntests = 0;

    // test whether we would statistically need to make more tests than the
    // total number of possible edges.
    if (num_tests >= num_neurons * (num_neurons - 1))
    {
        printf("BIGGER\n");
        // here RNG is serial, so we use "master" seed
        std::mt19937 generator(msd);
        // if yes, make a map containing the proba for each possible edge
        map_proba proba_edges;
        double distance;
        #pragma omp parallel num_threads(omp)
        {
            map_proba proba_local;
            #pragma omp for nowait schedule(static)
            for (size_t i=0; i<source_nodes.size(); i++)
            {
                for (size_t j=0; j<target_nodes.size(); j++)
                {
                    distance = sqrt((x[j] - x[i])*(x[j] - x[i])
                                    + (y[j] - y[i])*(y[j] - y[i]));
                    proba_local[edge_t(source_nodes[i], target_nodes[j])] =
                        _proba(rule_type, scale, distance);
                }
            }
            #pragma omp critical
            proba_edges.insert(proba_local.begin(), proba_local.end());
        }
        printf("proba computed\n");
        // generate the edges
        while (current_enum < target_enum)
        {
            size_t src, tgt, i(current_enum);
            while (i < target_enum)
            {
                src = rnd_source(generator);
                tgt = rnd_target(generator);
                if (proba_edges[edge_t(src, tgt)] > rnd_uniform(generator))
                {
                    edges_tmp[0].push_back(src);
                    edges_tmp[1].push_back(tgt);
                    i++;
                }
            }

            // update ecurrent and (potentially) the results
            current_enum = multigraph ?
                target_enum : _unique_2d(edges_tmp, hash_map);

            printf("current: %lu\n", current_enum);
        }
    }
    else
    {
        // compute the distance only when testing an edge
        #pragma omp parallel num_threads(omp)
        {
            double distance, proba;
            size_t src, tgt, local_tests(0);
            std::mt19937 generator_(seeds[omp_get_thread_num()]);
            // thread local edges
            std::vector< std::vector<size_t> > elocal(2,
                                                      std::vector<size_t>());
            elocal[0].reserve((size_t) (num_edges / std::max(1u, omp - 1)));
            elocal[1].reserve((size_t) (num_edges / std::max(1u, omp - 1)));
            
            while (current_enum < target_enum)
            {
                #pragma omp for nowait schedule(static)
                for (size_t i=0; i<num_tests; i++)
                {
                    src = rnd_source(generator_);
                    tgt = rnd_target(generator_);
                    distance = sqrt((x[tgt] - x[src])*(x[tgt] - x[src]) +
                                    (y[tgt] - y[src])*(y[tgt] - y[src]));
                    proba = _proba(rule_type, scale, distance);
                    if (proba >= rnd_uniform(generator_))
                    {
                        elocal[0].push_back(src);
                        elocal[1].push_back(tgt);
                    }
                    local_tests += 1;
                }
                
                #pragma omp critical
                {
                    edges_tmp[0].insert(edges_tmp[0].end(),
                                        elocal[0].begin(), elocal[0].end());
                    edges_tmp[1].insert(edges_tmp[1].end(),
                                        elocal[1].begin(), elocal[1].end());
                    ntests += local_tests;
                }

                elocal[0].clear();
                elocal[1].clear();

                #pragma omp barrier // make sure edges_tmp ready before single

                #pragma omp single
                {
                current_enum = multigraph ?
                    target_enum : _unique_2d(edges_tmp, hash_map);
                num_tests = (target_enum - current_enum)
                    * (1. - current_enum / (num_neurons * (num_neurons - 1)))
                    / avg_proba;
                }
                #pragma omp barrier // current_enum ready for all
            }
        }
    }

    printf("Total tests: %lu", ntests);

    // fill the final edge container
    for (size_t i=0; i<num_edges; i++)
    {
        ia_edges[2*i] = edges_tmp[0][i + initial_enum];
        ia_edges[2*i + 1] = edges_tmp[1][i + initial_enum];
    }
}

}
