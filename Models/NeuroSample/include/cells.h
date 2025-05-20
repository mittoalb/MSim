#ifndef CELLS_H
#define CELLS_H

#include <cstdint>
#include <array>
#include <vector>

// build a 1-D Gaussian kernel
std::vector<double> gaussian_kernel(int radius, double sigma);

// carve warped macro-regions into labels[]
void add_macroregions(
    uint8_t*        labels,
    uint8_t*        occ,
    int             nz, int ny, int nx,
    int             macro_regions,
    double          region_smoothness
);

void branch_axons(
    uint8_t* labels,
    uint8_t* occ,
    int nz, int ny, int nx,
    int z, int y, int x,
    int depth, int idx,
    double dx, double dy, double dz,
    int radius, int max_depth,
    const float* rng_vals, int rng_len,
    double& total_length,
    int max_branches_base,
    uint8_t label
);

void grow_dendrites_from(
    uint8_t* labels, uint8_t* occ,
    int nz, int ny, int nx,
    int z0, int y0, int x0,
    int base_radius,
    int max_depth,
    int max_branch_base,
    const float* rng_vals, int rng_len,
    int& rng_index,
    double& total_length,
    uint8_t label
);

double add_neurons(
    uint8_t *labels,
    uint8_t *occ,
    int nz, int ny, int nx,
    const std::array<double, 3> &voxel_size,
    int num_cells,
    const std::array<double, 2> &cell_radius_range,
    const std::array<double, 2> &axon_dia_range,
    int max_depth,
    std::vector<std::array<int,3>>& centers     // ← NEW output argument
);

// Original draw_axon_tube signature with static RNG array:
std::array<int,3> draw_axon_tube(
    uint8_t* labels,
    uint8_t* occ,
    int nz, int ny, int nx,
    int z0, int y0, int x0,
    double dx, double dy, double dz,
    int steps,
    int radius,
    const float* rng_vals,
    int rng_len,
    int& rng_index,
    double& total_length,
    double jitter,
    double persist,
    const std::vector<std::array<double,3>>* converge_pts,
    double converge_radius,
    double converge_strength,
    uint8_t wall_label,
    uint8_t lumen_label
);

double add_glial(
    uint8_t* labels,             // 3D label volume
    uint8_t* occ,                // 3D occupancy mask
    int nz, int ny, int nx,      // volume dimensions
    int num_glia,                // how many glia to place
    int glia_radius_min,         // min ellipsoid radius
    int glia_radius_max,         // max ellipsoid radius
    int dend_depth,              // dendrite length
    int dend_branches,           // dendrite fan‐out per soma
    const float* rng_vals,       // precomputed RNG floats [0..1)
    int rng_len,                 // length of rng_vals
    int& rng_index,              // in/out index into rng_vals
    double& total_length         // in/out accumulator (µm or voxels)
);


void connect_somas_with_synapses(
    uint8_t* labels,                         // 1
    uint8_t* occ,                            // 2
    int nz, int ny, int nx,                  // 3,4,5
    const std::vector<std::array<int,3>>& centers, // 6
    const float* rng_vals,                   // 7
    int rng_len,                             // 8
    int& rng_index,                          // 9
    double& total_length,                    // 10
    int axon_steps,                          // 11
    int axon_radius,                         // 12
    float jitter,                            // 13
    float persist,                           // 14
    int dend_branches,                       // 15
    float extra_conn_prob,                   // 16
    uint8_t synapse_label                    // 17
);

void connect_glia_to_neurons(
    uint8_t* labels,
    int nz, int ny, int nx,
    const std::vector<std::array<int,3>>& neuron_centers,
    int contact_label = 12,
    int contact_radius = 5
);
#endif // CELLS_H