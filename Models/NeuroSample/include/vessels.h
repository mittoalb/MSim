#ifndef VESSELS_H
#define VESSELS_H

#include <cstdint>

// carve one tree from a single root, return carved length (in carved voxels)
// now caps each branch at MAX_BRANCH_LEN
double draw_vessels(
    uint8_t*    mask,
    uint8_t*    occ,
    int         nz, int ny, int nx,
    int         root_z, int root_y, int root_x,
    int         max_depth,
    int         base_radius,
    const float* rng_vals,
    int         rng_len,
    double      init_dx,
    double      init_dy,
    double      init_dz,
    int         TRUNK_LEN,
    int         JITTER_INTERVAL,
    int         MAX_BRANCHES_BASE,
    int         MAX_BRANCH_LEN,
    double      RADIUS_DECAY
);

// plant many roots, carve them into labels[], return total center-line length
double add_vessels(
    uint8_t* labels,
    int      nz, int ny, int nx,
    int      num_vessels,
    int      max_depth,
    int      vessel_radius_avg,      // new: average radius
    double   vessel_radius_jitter,   // new: jitter fraction [0..1]
    int      trunk_len,
    int      jitter_interval,
    int      max_branches,
    int      branch_len,
    double   radius_decay,
    int      seed
);

#endif // VESSELS_H

