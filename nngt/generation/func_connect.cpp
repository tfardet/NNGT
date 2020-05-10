// connect.cpp
//
// Accelerated network generation functions
//
// Note: signed ints are required because MSVC does not implement OpenMP 3
// standards which allow unsigned variables in for loops.

#include "func_connect.h"

#include <omp.h>

#define _USE_MATH_DEFINES
#include <limits>
#include <random>
#include <cmath>
#include <numeric>  // partial_sum

#include <stdexcept>
#include <assert.h>


namespace generation {

size_t _unique_1d(std::vector<size_t>& a,
                  std::unordered_set<size_t>& hash_set)
{
    size_t number;
    size_t total_unique = hash_set.size();

    for (size_t i = 0; i < a.size(); i++)
    {
        number = a[i];

        // check if this number is negative () or already in the set
        if (hash_set.find(number) == hash_set.end())
        {
            // it's not in there yet so add it and set the count to 1
            hash_set.insert(number);
            a[total_unique] = a[i];
            total_unique += 1;
        }
    }

    return total_unique;
}


size_t _unique_2d(std::vector< std::vector<size_t> >& a, set_t& hash_set,
                  set_t& recip_set, bool directed)
{
    size_t total_unique = hash_set.size();
    size_t num_edges = a[0].size();
    size_t s, t;
    edge_t edge;

    for (size_t i = total_unique; i < num_edges; i++)
    {
        s = a[0][i];
        t = a[1][i];

        edge = edge_t(s, t);

        // check if this number is already in the set
        if (hash_set.find(edge) == hash_set.end())
        {
            if (directed)
            {
                // it's not in there yet so add it and set the count to 1
                hash_set.insert(edge);

                a[0][total_unique] = s;
                a[1][total_unique] = t;

                total_unique += 1;
            }
            else if (recip_set.find(edge) == recip_set.end())
            {
                hash_set.insert(edge);
                recip_set.insert(edge_t(t, s));

                a[0][total_unique] = s;
                a[1][total_unique] = t;

                total_unique += 1;
            }
        }
    }

    return total_unique;
}


size_t _unique_2d(std::vector< std::vector<size_t> >& a, set_t& hash_set,
                  std::vector<float>& dist, const std::vector<float>& dist_tmp,
                  set_t& recip_set, bool directed)
{
    size_t total_unique = hash_set.size();
    size_t num_edges = a[0].size();
    size_t initial_enum = total_unique;
    size_t s, t;
    edge_t edge;

    for (size_t i = total_unique; i < num_edges; i++)
    {
        s = a[0][i];
        t = a[1][i];

        edge = edge_t(s, t);

        // check if this number is already in the set
        if (hash_set.find(edge) == hash_set.end())
        {
            if (directed)
            {
                // it's not in there yet so add it and set the count to 1
                hash_set.insert(edge);

                a[0][total_unique] = s;
                a[1][total_unique] = t;

                dist.push_back(dist_tmp[i - initial_enum]);

                total_unique += 1;
            }
            else if (recip_set.find(edge) == recip_set.end())
            {
                hash_set.insert(edge);
                recip_set.insert(edge_t(t, s));

                a[0][total_unique] = s;
                a[1][total_unique] = t;

                dist.push_back(dist_tmp[i - initial_enum]);

                total_unique += 1;
            }
        }
    }

    return total_unique;
}


std::vector<size_t> _gen_edge_complement(
  std::mt19937& generator, const std::vector<size_t>& nodes, size_t other_end,
  unsigned int degree,
  const std::vector< std::vector<size_t> >* existing_edges,
  bool multigraph, bool directed)
{
    // Initialize the RNG
    size_t min_idx = *std::min_element(nodes.begin(), nodes.end());
    size_t max_idx = *std::max_element(nodes.begin(), nodes.end());
    std::uniform_int_distribution<size_t> uniform_(min_idx, max_idx);

    // generate the complements
    std::vector<size_t> result;
    size_t ecurrent = 0;

    std::unordered_set<size_t> hash_set;

    // check the existing edges
    const size_t num_old_edges = existing_edges ? existing_edges[0].size() : 0;
    size_t node;

    for (size_t i=0; i < num_old_edges; i++)
    {
        if (existing_edges->at(0)[i] == other_end)
        {
            node = existing_edges->at(1)[i];

            result.push_back(node);
            hash_set.insert(node);
        }

        if (not directed and existing_edges->at(1)[i] == other_end)
        {
            node = existing_edges->at(1)[i];

            if (hash_set.find(node) == hash_set.end())
            {
                result.push_back(node);
                hash_set.insert(node);
            }
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
            cplt = uniform_(generator);
            if (cplt != other_end)
            {
                result[ecurrent + j] = cplt;
                j++;
            }
        }

        // update ecurrent and (potentially) the results
        ecurrent = multigraph ? target_degree : _unique_1d(result, hash_set);
    }

    return result;
}


void _gen_edges(
  int64_t* ia_edges, const std::vector<size_t>& first_nodes,
  const std::vector<unsigned int>& degrees,
  const std::vector<size_t>& second_nodes,
  const std::vector< std::vector<size_t> >& existing_edges, unsigned int idx,
  bool multigraph, bool directed, std::vector<long>& seeds)
{
    // compute the cumulated sum of the degrees
    std::vector<size_t> cum_degrees(degrees.size());
    std::partial_sum(degrees.begin(), degrees.end(), cum_degrees.begin());

    int omp = seeds.size();

    // generate the edges
    #pragma omp parallel num_threads(omp)
    {
        std::mt19937 generator_(seeds[omp_get_thread_num()]);
        std::vector<size_t> res_tmp;

        #pragma omp for schedule(static)
        for (size_t node=0; node < first_nodes.size(); node++)
        {
            // generate the vector of complementary nodes
            res_tmp = _gen_edge_complement(
                generator_, second_nodes, node, degrees[node],
                &existing_edges, multigraph, true);

            // fill the edges
            size_t idx_start = cum_degrees[node] - degrees[node];

            for (unsigned int j = 0; j < degrees[node]; j++)
            {
                ia_edges[2*(idx_start + j) + idx] = first_nodes[node];
                ia_edges[2*(idx_start + j) + 1 - idx] = res_tmp[j];
            }
        }
    }
}


/*
* Distance-rule algorithms
*/

void _cdistance_rule(int64_t* ia_edges, const std::vector<size_t>& source_nodes,
  const std::vector<std::vector<size_t>>& target_nodes,
  const std::string& rule, float scale, float norm,
  const std::vector<float>& x, const std::vector<float>& y, size_t num_neurons,
  size_t num_edges, const std::vector< std::vector<size_t> >& existing_edges,
  std::vector<float>& dist, bool multigraph, bool directed,
  std::vector<long>& seeds)
{
    float inv_scale = 1. / scale;
    int num_omp = seeds.size();

    std::uniform_real_distribution<float> rnd_uniform(0., 1.);

    // rule into int
    int rule_type = -1;
    if (rule == "lin")
    {
        rule_type = 0;
    }
    else if (rule == "exp")
    {
        rule_type = 1;
    }
    else if (rule == "gaussian")
    {
        rule_type = 2;
    }
    else
    {
        throw std::invalid_argument("`rule` must be among 'lin', 'exp', or "
                                    "'gaussian'.");
    }

    size_t initial_enum = existing_edges.empty() ?
        0 : existing_edges[0].size();               // initial number of edges
    size_t current_enum = initial_enum;             // current number of edges
    size_t ecount_fill = 0;                         // tmp number of edges
    size_t target_enum = current_enum + num_edges;  // target number of edges
    dist.reserve(num_edges);

    // set the number of tests associated to each node proportionnaly to its
    // number of neighbours
    size_t tot_neighbours = 0;

    for (size_t i=0; i < target_nodes.size(); i++)
    {
        tot_neighbours += target_nodes[i].size();
    }
    double neigh_norm = 1. / tot_neighbours;

    // if not using multigraph, assert that we have enough neighbours
    if (tot_neighbours < target_enum)
    {
        throw std::invalid_argument("Scale is too small: there are not enough "
                                    "close neighbours to create the required "
                                    "number of connections. Increase `scale` "
                                    "or `neuron_density`.");
    }

    // create the edges
    if (num_edges > 0)
    {
        #pragma omp parallel num_threads(num_omp)
        {
            float distance, proba;
            size_t src, tgt, local_tests, nln, rnd;
            std::vector<size_t> local_tgts;
            std::mt19937 generator_(seeds[omp_get_thread_num()]);
            // thread local edges
            set_t hash_set, recip_set;
            size_t num_elocal = 0;
            std::vector< std::vector<size_t> > local_edges(
                2, std::vector<size_t>());
            std::vector< std::vector<size_t> > elocal_tmp(
                2, std::vector<size_t>());
            std::vector< float > local_dist, dist_tmp;
            do {
                // reset edge number (recompute it each time)
                #pragma omp single
                {
                    current_enum = initial_enum;
                }
                #pragma omp barrier
                #pragma omp flush(current_enum)
                
                // the static schedule is CAPITAL: each thread must always
                // handle the same nodes
                #pragma omp for schedule(static)
                for (size_t i=0; i<target_nodes.size(); i++)
                {
                    local_tgts = target_nodes[i];
                    local_tests = target_nodes[i].size()
                                  * (target_enum - current_enum) * neigh_norm;
                    // test at least all neighbours
                    local_tests = std::max(local_tests, local_tgts.size());
                    elocal_tmp[0].reserve(local_tests);
                    elocal_tmp[1].reserve(local_tests);
                    dist_tmp.reserve(local_tests);
                    // initialize source; set target generator
                    src = source_nodes[i];
                    nln = local_tgts.size();  // number of local neighbours
                    std::uniform_int_distribution<size_t> rnd_target(
                        0, nln - 1);

                    for (size_t j=0; j<local_tests; j++)
                    {
                        tgt = src;
                        while (tgt == src)
                        {
                            rnd = rnd_target(generator_);
                            tgt = local_tgts[rnd];
                        }
                        distance = std::sqrt(
                            (x[tgt] - x[src])*(x[tgt] - x[src]) +
                            (y[tgt] - y[src])*(y[tgt] - y[src]));
                        proba = _proba(rule_type, norm, inv_scale, distance);
                        if (proba >= rnd_uniform(generator_))
                        {
                            elocal_tmp[0].push_back(src);
                            elocal_tmp[1].push_back(tgt);
                            dist_tmp.push_back(distance);
                        }
                    }
                }

                local_edges[0].insert(local_edges[0].end(),
                                      elocal_tmp[0].begin(),
                                      elocal_tmp[0].end());
                local_edges[1].insert(local_edges[1].end(),
                                      elocal_tmp[1].begin(),
                                      elocal_tmp[1].end());
                num_elocal = multigraph
                             ? local_edges[0].size()
                             : _unique_2d(local_edges, hash_set,
                                          local_dist, dist_tmp, recip_set);

                local_edges[0].resize(num_elocal);
                local_edges[1].resize(num_elocal);

                #pragma omp atomic
                current_enum += num_elocal;

                #pragma omp barrier              // ensure everyone is ready
                #pragma omp flush(current_enum)  // and gets last value
                #pragma omp barrier              // for sure
            } while (current_enum < target_enum);

            // fill the edge container: first with the existing edges
            #pragma omp single
            {
                for (size_t i=0; i<initial_enum; i++)
                {
                    ia_edges[2*i]     = existing_edges[0][i];
                    ia_edges[2*i + 1] = existing_edges[1][i];
                }
                ecount_fill += initial_enum;
            }
            // then, once this is done
            #pragma omp barrier
            // each thread successively adds its local edges
            #pragma omp critical
            {
                if (current_enum > num_edges)
                {
                    // more connections than needed, we need to randomize the
                    // generated edges and keep only a fraction
                    size_t keep = ceil(
                        local_edges[0].size() * num_edges /
                        static_cast<double>(current_enum - initial_enum));
                    // randomize (first copy MT generator
                    auto rng_copy  = generator_;
                    auto rng_copy2 = generator_;
                    std::shuffle(local_edges[0].begin(), local_edges[0].end(), generator_);
                    std::shuffle(local_edges[1].begin(), local_edges[1].end(), rng_copy);
                    std::shuffle(local_dist.begin(), local_dist.end(), rng_copy2);
                    // keep only chosen ones
                    local_edges[0].resize(keep);
                    local_edges[1].resize(keep);
                    local_dist.resize(keep);
                    num_elocal = keep;
                }
                size_t i = 0;
                #pragma omp flush(ecount_fill)
                if (ecount_fill + num_elocal <= target_enum)
                {
                    dist.insert(dist.begin() + ecount_fill, local_dist.begin(),
                                local_dist.end());
                }
                else
                {
                    dist.insert(dist.begin() + ecount_fill, local_dist.begin(),
                                local_dist.begin() + target_enum
                                - ecount_fill);
                }
                while (i < num_elocal && ecount_fill < target_enum)
                {
                    ia_edges[2*ecount_fill]     = local_edges[0][i];
                    ia_edges[2*ecount_fill + 1] = local_edges[1][i];
                    ecount_fill += 1;
                    i += 1;
                }
            }
        }
    }
}

}
