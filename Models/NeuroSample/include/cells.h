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

// place cell bodies & nuclei, then wire up with axons
double add_neurons(
    uint8_t*                        labels,
    uint8_t*                        occ,
    int                             nz, int ny, int nx,
    const std::array<double,3>&     voxel_size,
    int                             num_cells,
    const std::array<double,2>&     cell_radius_range,
    const std::array<double,2>&     axon_dia_range,
    int                             max_depth
);

std::array<int, 3> draw_axon_tube(
    uint8_t* labels,
    uint8_t* occ,
    int nz, int ny, int nx,
    int z0, int y0, int x0,
    double dx0, double dy0, double dz0,
    int steps,
    int radius,
    const float* rng_vals, int rng_len,
    int& rng_index,
    double& total_length,
    double jitter = 0.2  // <── ADD THIS
);

void add_glial(
    uint8_t* labels,
    uint8_t* occ,
    int nz, int ny, int nx,
    int num_glia,
    int glia_radius_min,
    int glia_radius_max,
    int dend_depth,
    int dend_branches,
    const float* rng_vals,
    int rng_len,
    double& total_length
);
#endif // CELLS_H