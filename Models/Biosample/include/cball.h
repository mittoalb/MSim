//cball.h
#pragma once
#include <cstdint>
#include <cmath>

// carve_ball: fills a sphere of radius `radius` at (z0,y0,x0),
// labeling voxels with `label`, skipping already-occupied voxels,
// and counting newly carved voxels into `length_counter`.
inline void carve_ball(
    uint8_t*   labels,
    uint8_t*   occ,             // occupancy mask, 0=free, 1=occupied
    int        nz, int ny, int nx,
    int        z0, int y0, int x0,
    int        radius,
    uint8_t    label,
    double&    length_counter
) {
    int rr = radius * radius;
    for (int dz = -radius; dz <= radius; ++dz) {
        int zz = z0 + dz; if (zz < 0 || zz >= nz) continue;
        int ddz = dz * dz;
        for (int dy = -radius; dy <= radius; ++dy) {
            int yy = y0 + dy; if (yy < 0 || yy >= ny) continue;
            int ddy = dy * dy;
            for (int dx = -radius; dx <= radius; ++dx) {
                if (ddz + ddy + dx*dx > rr) continue;
                int xx = x0 + dx; if (xx < 0 || xx >= nx) continue;
                int idx = (zz*ny + yy)*nx + xx;
                if (!occ[idx]) {
                    labels[idx] = label;
                    occ[idx]    = 1;
                    length_counter += 1.0;
                }
            }
        }
    }
}

