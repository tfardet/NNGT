// connect.h
//
// Accelerated network generation functions

#ifndef FUNC_CONNECT_H
#define FUNC_CONNECT_H

#include <vector>
#include <tuple>
#include <unordered_map>

#include <cmath>
#include <algorithm>


namespace generation {

/*
 * Typedefs and struct for edge unordered map definition.
 */

typedef std::tuple<size_t, size_t> edge_t;

struct key_hash : public std::unary_function<edge_t, std::size_t>
{
   std::size_t operator()(const edge_t& k) const
   {
      return std::get<0>(k) ^ std::get<1>(k);
   }
};

struct key_equal : public std::binary_function<edge_t, edge_t, bool>
{
   bool operator()(const edge_t& v0, const edge_t& v1) const
   {
      return ( std::get<0>(v0) == std::get<0>(v1) &&
               std::get<1>(v0) == std::get<1>(v1) );
   }
};

typedef std::unordered_map<edge_t, size_t, key_hash, key_equal> map_t;
//~ typedef std::unordered_map<size_t, std::unordered_map<size_t, int>> map_t;


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
size_t _unique_1d(std::vector<int>& a,
                  std::unordered_map<size_t, size_t>& hash_map);


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
size_t _unique_2d(std::vector< std::vector<int> >& a, map_t& hash_map);


size_t _unique_2d(std::vector< std::vector<int> >& a, map_t& hash_map,
                  std::vector<float>& dist,
                  const std::vector<float>& dist_tmp);


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
  long seed, const std::vector<size_t>& nodes, const size_t other_end,
  const size_t degree, const std::vector< std::vector<size_t> >* existing_edges,
  const bool multigraph);


/*
 * Generate random edges from a list of nodes, their target degrees, and a
 * second population of nodes.
 *
 * \param ia_edges       - Linearized (E, 2) array that will contain the edges
 * \param first_nodes    - Population the degree of which is known.
 * \param degrees        - Degree of each node in `first_nodes`.
 * \param second_nodes   - Population from whch to draw the complementary end of the edges.
 * \param existing_edges - 2D-array containing the existing edges.
 * \param multigraph     - Whether multiple edges are allowed.
 * \param idx            - Index determining source/target from first/second nodes
 * \param directed       - Whether the edges are directed or not.
 * \param msd            - Master seed.
 * \param omp            - Number of OpenMP threads.
 */
void _gen_edges(
  size_t* ia_edges, const std::vector<size_t>& first_nodes,
  const std::vector<size_t>& degrees, const std::vector<size_t>& second_nodes,
  const std::vector< std::vector<size_t> >& existing_edges, unsigned int idx,
  bool multigraph, bool directed, long msd, unsigned int omp);


/*
 * Parallel distance-rule generator.
 * 
 * \param ia_edges       - array that will contain the edges
 * \param source_nodes   - array containing the ids of the source nodes
 * \param target_nodes   - array containing the ids of the target nodes
 * \param rule           - rule for prabability computation ("exp" or "lin")
 * \param scale          - typical distance for probability computation
 * \param x              - x coordinate of the neurons' positions
 * \param y              - y coordinate of the neurons' positions
 * \param area           - total area of the spatial environment
 * \param num_neurons    - total number of neurons
 * \param num_edges      - desired number of edges
 * \param existing_edges - array containing the edges that are already present
 *                         in the graph
 * \param multigraph     - whether the graph can have duplicate edges
 * \param msd            - master seed given by numpy in the cython function
 * \param omp            - number of OpenMP threads
 */
void _cdistance_rule(size_t* ia_edges, const std::vector<size_t>& source_nodes,
  const std::vector<std::vector<size_t>>& target_nodes,
  const std::string& rule, float scale, const std::vector<float>& x,
  const std::vector<float>& y, size_t num_neurons, size_t num_edges,
  const std::vector< std::vector<size_t> >& existing_edges,
  std::vector<float>& dist, bool multigraph, long msd, unsigned int omp);


//~ static float exp_adjust[256] = {
    //~ 1.040389835, 1.039159306, 1.037945888, 1.036749401, 1.035569671,
    //~ 1.034406528, 1.033259801, 1.032129324, 1.031014933, 1.029916467,
    //~ 1.028833767, 1.027766676, 1.02671504, 1.025678708, 1.02465753, 1.023651359,
    //~ 1.022660049, 1.021683458, 1.020721446, 1.019773873, 1.018840604,
    //~ 1.017921503, 1.017016438, 1.016125279, 1.015247897, 1.014384165,
    //~ 1.013533958, 1.012697153, 1.011873629, 1.011063266, 1.010265947,
    //~ 1.009481555, 1.008709975, 1.007951096, 1.007204805, 1.006470993,
    //~ 1.005749552, 1.005040376, 1.004343358, 1.003658397, 1.002985389,
    //~ 1.002324233, 1.001674831, 1.001037085, 1.000410897, 0.999796173,
    //~ 0.999192819, 0.998600742, 0.998019851, 0.997450055, 0.996891266,
    //~ 0.996343396, 0.995806358, 0.995280068, 0.99476444, 0.994259393,
    //~ 0.993764844, 0.993280711, 0.992806917, 0.992343381, 0.991890026,
    //~ 0.991446776, 0.991013555, 0.990590289, 0.990176903, 0.989773325,
    //~ 0.989379484, 0.988995309, 0.988620729, 0.988255677, 0.987900083,
    //~ 0.987553882, 0.987217006, 0.98688939, 0.98657097, 0.986261682, 0.985961463,
    //~ 0.985670251, 0.985387985, 0.985114604, 0.984850048, 0.984594259,
    //~ 0.984347178, 0.984108748, 0.983878911, 0.983657613, 0.983444797,
    //~ 0.983240409, 0.983044394, 0.982856701, 0.982677276, 0.982506066,
    //~ 0.982343022, 0.982188091, 0.982041225, 0.981902373, 0.981771487,
    //~ 0.981648519, 0.981533421, 0.981426146, 0.981326648, 0.98123488,
    //~ 0.981150798, 0.981074356, 0.981005511, 0.980944219, 0.980890437,
    //~ 0.980844122, 0.980805232, 0.980773726, 0.980749562, 0.9807327, 0.9807231,
    //~ 0.980720722, 0.980725528, 0.980737478, 0.980756534, 0.98078266,
    //~ 0.980815817, 0.980855968, 0.980903079, 0.980955475, 0.981017942,
    //~ 0.981085714, 0.981160303, 0.981241675, 0.981329796, 0.981424634,
    //~ 0.981526154, 0.981634325, 0.981749114, 0.981870489, 0.981998419,
    //~ 0.982132873, 0.98227382, 0.982421229, 0.982575072, 0.982735318,
    //~ 0.982901937, 0.983074902, 0.983254183, 0.983439752, 0.983631582,
    //~ 0.983829644, 0.984033912, 0.984244358, 0.984460956, 0.984683681,
    //~ 0.984912505, 0.985147403, 0.985388349, 0.98563532, 0.98588829, 0.986147234,
    //~ 0.986412128, 0.986682949, 0.986959673, 0.987242277, 0.987530737,
    //~ 0.987825031, 0.988125136, 0.98843103, 0.988742691, 0.989060098,
    //~ 0.989383229, 0.989712063, 0.990046579, 0.990386756, 0.990732574,
    //~ 0.991084012, 0.991441052, 0.991803672, 0.992171854, 0.992545578,
    //~ 0.992924825, 0.993309578, 0.993699816, 0.994095522, 0.994496677,
    //~ 0.994903265, 0.995315266, 0.995732665, 0.996155442, 0.996583582,
    //~ 0.997017068, 0.997455883, 0.99790001, 0.998349434, 0.998804138,
    //~ 0.999264107, 0.999729325, 1.000199776, 1.000675446, 1.001156319,
    //~ 1.001642381, 1.002133617, 1.002630011, 1.003131551, 1.003638222,
    //~ 1.00415001, 1.004666901, 1.005188881, 1.005715938, 1.006248058,
    //~ 1.006785227, 1.007327434, 1.007874665, 1.008426907, 1.008984149,
    //~ 1.009546377, 1.010113581, 1.010685747, 1.011262865, 1.011844922,
    //~ 1.012431907, 1.013023808, 1.013620615, 1.014222317, 1.014828902,
    //~ 1.01544036, 1.016056681, 1.016677853, 1.017303866, 1.017934711,
    //~ 1.018570378, 1.019210855, 1.019856135, 1.020506206, 1.02116106,
    //~ 1.021820687, 1.022485078, 1.023154224, 1.023828116, 1.024506745,
    //~ 1.025190103, 1.02587818, 1.026570969, 1.027268461, 1.027970647, 1.02867752,
    //~ 1.029389072, 1.030114973, 1.030826088, 1.03155163, 1.032281819, 1.03301665,
    //~ 1.033756114, 1.034500204, 1.035248913, 1.036002235, 1.036760162,
    //~ 1.037522688, 1.038289806, 1.039061509, 1.039837792, 1.040618648
//~ };


static inline float _proba(int rule, float inv_scale, float distance)
{
    if (rule == 0)  // linear
    {
        return std::max(0., 1. - distance * inv_scale);
    }
    else  // exponential
    {
        return std::exp(-distance * inv_scale);
        //~ float x = -1.442695040f * distance * inv_scale;
        //~ float offset = (x < 0) ? 1.0f : 0.0f;
        //~ float clipp = (x < -126) ? -126.0f : x;
        //~ int w = clipp;
        //~ float z = clipp - w + offset;
        //~ union { uint32_t i; float f; } u = {static_cast<uint32_t>(
            //~ (1 << 23) * (clipp + 121.2740575f + 27.7280233f / (4.84252568f - z)
                         //~ - 1.49012907f * z))
        //~ };

        //~ return u.f;
        
        //~ long tmp = -1512775 * distance * inv_scale + 1072632447;
        //~ uint index = (tmp >> 12) & 0xFF;
        //~ v.i = tmp << 32;
        //~ return static_cast<float>(v.f * exp_adjust[index]);
    }
};

}

#endif // FUNC_CONNECT_H
