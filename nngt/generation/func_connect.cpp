// connect.cpp
//
// Accelerated network generation functions

#include "func_connect.h"


namespace generation {

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
        ecurrent = multigraph ? target_degree : _unique_1d(result);
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
    std::vector<long> seeds(omp, msd);
    for (size_t i=0; i < omp; i++)
    {
        seeds[i] += i + 1;
    }

    // compute the cumulated sum of the degrees
    std::vector<size_t> cum_degrees(degrees.size());
    std::partial_sum(degrees.begin(), degrees.end(), cum_degrees.begin());

    // generate the edges
    #pragma omp parallel num_threads(omp)
    {
        std::mt19937 generator_(seeds[omp_get_thread_num()]);
        
        #pragma omp for schedule(dynamic)
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
void _distance_rule(size_t* ia_edges, const std::vector<size_t>& first_nodes,
  const std::vector<size_t>& second_nodes, const std::string& rule,
  double scale, const std::vector<double>& x, const std::vector<double>& y
  size_t num_edges, const std::vector< std::vector<size_t> >& existing_edges,
  bool multigraph, bool directed, long msd, unsigned int omp)
{
    // Initialize secondary seeds
    std::vector<long> seeds(omp, msd);
    for (size_t i=0; i < omp; i++)
    {
        seeds[i] += i + 1;
    }

    def exp_rule(pos_src, pos_target):
        dist = np.linalg.norm(pos_src-pos_target,axis=0)
        return np.exp(np.divide(dist,-scale))
    def lin_rule(pos_src, pos_target):
        dist = np.linalg.norm(pos_src-pos_target,axis=0)
        return np.divide(scale-dist,scale).clip(min=0.)
    dist_test = exp_rule if rule == "exp" else lin_rule
    # compute the required values
    source_ids = np.array(source_ids).astype(int)
    target_ids = np.array(target_ids).astype(int)
    num_source, num_target = len(source_ids), len(target_ids)
    edges, _ = _compute_connections(num_source, num_target,
                             density, edges, avg_deg, directed, reciprocity=-1)
    b_one_pop = _check_num_edges(
        source_ids, target_ids, edges, directed, multigraph)
    # create the edges
    ia_edges = np.zeros((edges,2), dtype=int)
    num_test, num_ecurrent = 0, 0
    while num_ecurrent != edges and num_test < MAXTESTS:
        num_create = edges-num_ecurrent
        ia_sources = source_ids[randint(0, num_source, num_create)]
        ia_targets = target_ids[randint(0, num_target, num_create)]
        test = dist_test(positions[:,ia_sources],positions[:,ia_targets])
        ia_valid = np.greater(test,np.random.uniform(size=num_create))
        ia_edges_tmp = np.array([ia_sources[ia_valid],ia_targets[ia_valid]]).T
        ia_edges, num_ecurrent = _filter(ia_edges, ia_edges_tmp, num_ecurrent,
                                         b_one_pop, multigraph)
        num_test += 1
    if num_test == MAXTESTS:
        ia_edges = ia_edges[:num_ecurrent,:]
        warnings.warn("Maximum number of tests reached, stopped  generation \
with {} edges.".format(num_ecurrent), RuntimeWarning)
    if not directed:
        ia_edges = np.concatenate((ia_edges, ia_edges[:,::-1]))
        ia_edges = _unique_rows(ia_edges)
    return ia_edges*/
    
}
