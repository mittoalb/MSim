// vessels.h
#ifndef VESSELS_H
#define VESSELS_H

#include <cstdint>

// Draw a vessel tree with simple binary splits (area-conserving)
double draw_vessels(
    uint8_t*     mask,
    uint8_t*     occ,
    int          nz, int ny, int nx,
    int          root_z, int root_y, int root_x,
    int          max_depth,
    int          base_radius,
    const float* rng_vals,
    int          rng_len,
    double       init_dx,
    double       init_dy,
    double       init_dz,
    int          TRUNK_LEN,
    int          JITTER_INTERVAL,
    int          MAX_BRANCH_LEN
);

double add_vessels(
    uint8_t* labels,
    int      nz, int ny, int nx,
    int      num_vessels,
    int      max_depth,
    int      vessel_radius_avg,
    double   vessel_radius_jitter,
    int      trunk_len,
    int      jitter_interval,
    int      max_branches,
    int      branch_len,
    double   radius_decay,
    int      seed
);


void carve_hollow_ellipsoid(
    uint8_t* labels, uint8_t* occ,
    int nz, int ny, int nx,
    int zc, int yc, int xc,
    double a, double b, double c,
    double inner_radius,
    uint8_t label,
    double& total_len
);


#endif // VESSELS_H
