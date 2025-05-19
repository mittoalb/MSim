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

// iteratively grow axonal branches from one seed
void branch_axons(
    uint8_t*        labels,
    uint8_t*        occ,
    int             nz, int ny, int nx,
    int             z, int y, int x,
    int             depth, int idx,
    double          dx, double dy, double dz,
    int             radius, int max_depth,
    const float*    rng_vals, int rng_len,
    double&         total_length
);

// grow a straight axon trunk then branch
void draw_axons(
    uint8_t*        labels,
    uint8_t*        occ,
    int             nz, int ny, int nx,
    int             z0, int y0, int x0,
    int             z1, int y1, int x1,
    int             max_depth,
    int             base_radius,
    const float*    rng_vals, int rng_len,
    double&         total_length
);

// connect a set of cell centers via MST + axons
void connect_cells(
    uint8_t*                             labels,
    uint8_t*                             occ,
    double&                              total_length,
    int                                  nz, int ny, int nx,
    const std::vector<std::array<int,3>>& centers,
    int                                  max_depth,
    double                               axon_dia_px,
    const float*                         rng_vals,
    int                                  rng_len
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

#endif // CELLS_H

