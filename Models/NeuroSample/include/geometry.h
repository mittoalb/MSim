//geometry.h
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

inline void carve_ellipsoid(
    uint8_t*   labels,
    uint8_t*   occ,
    int        nz, int ny, int nx,
    int        z0, int y0, int x0,
    int        rz, int ry, int rx,
    uint8_t    label,
    double&    length_counter
) {
    for (int dz = -rz; dz <= rz; ++dz) {
        int zz = z0 + dz; if (zz < 0 || zz >= nz) continue;
        for (int dy = -ry; dy <= ry; ++dy) {
            int yy = y0 + dy; if (yy < 0 || yy >= ny) continue;
            for (int dx = -rx; dx <= rx; ++dx) {
                int xx = x0 + dx; if (xx < 0 || xx >= nx) continue;

                double norm = (dz / double(rz)) * (dz / double(rz)) +
                              (dy / double(ry)) * (dy / double(ry)) +
                              (dx / double(rx)) * (dx / double(rx));
                if (norm <= 1.0) {
                    int idx = (zz * ny + yy) * nx + xx;
                    if (!occ[idx]) {
                        labels[idx] = label;
                        occ[idx]    = 1;
                        length_counter += 1.0;
                    }
                }
            }
        }
    }
}


// can_place_ellipsoid: check if we can place an ellipsoid at (z0,y0,x0)
inline bool can_place_ellipsoid(
    const uint8_t* occ,
    int nz, int ny, int nx,
    int z0, int y0, int x0,
    int rz, int ry, int rx
) {
    for (int dz = -rz; dz <= rz; ++dz)
        for (int dy = -ry; dy <= ry; ++dy)
            for (int dx = -rx; dx <= rx; ++dx) {
                double val = (dx * dx) / double(rx * rx)
                           + (dy * dy) / double(ry * ry)
                           + (dz * dz) / double(rz * rz);
                if (val > 1.0) continue;
                int zz = z0 + dz, yy = y0 + dy, xx = x0 + dx;
                if (zz < 0 || zz >= nz || yy < 0 || yy >= ny || xx < 0 || xx >= nx) return false;
                int idx = (zz * ny + yy) * nx + xx;
                if (occ[idx]) return false;
            }
    return true;
}

// Carve a solid cylinder between two points (z0,y0,x0) → (z1,y1,x1)
inline void carve_cylinder(
    uint8_t* labels,
    uint8_t* occ,
    int nz, int ny, int nx,
    int z0, int y0, int x0,
    int z1, int y1, int x1,
    int radius,
    uint8_t label,
    double& length_counter
) {
    double dz = z1 - z0;
    double dy = y1 - y0;
    double dx = x1 - x0;
    double length = std::sqrt(dx*dx + dy*dy + dz*dz);
    int steps = std::max(1, int(length * 1.5));

    for (int i = 0; i <= steps; ++i) {
        double t = i / double(steps);
        int zc = int(std::round(z0 + dz * t));
        int yc = int(std::round(y0 + dy * t));
        int xc = int(std::round(x0 + dx * t));

        int rr = radius * radius;
        for (int dz_ = -radius; dz_ <= radius; ++dz_) {
            int zz = zc + dz_; if (zz < 0 || zz >= nz) continue;
            for (int dy_ = -radius; dy_ <= radius; ++dy_) {
                int yy = yc + dy_; if (yy < 0 || yy >= ny) continue;
                for (int dx_ = -radius; dx_ <= radius; ++dx_) {
                    if (dz_*dz_ + dy_*dy_ + dx_*dx_ > rr) continue;
                    int xx = xc + dx_; if (xx < 0 || xx >= nx) continue;
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
}

// Carve a hollow cylindrical shell between (z0,y0,x0) and (z1,y1,x1)
inline void carve_cylinder_segment(
    uint8_t* labels,
    uint8_t* occ,
    int nz, int ny, int nx,
    double z0, double y0, double x0,
    double z1, double y1, double x1,
    int radius,
    uint8_t wall_label,
    uint8_t lumen_label,
    double& length_counter
) {
    int steps = int(std::ceil(std::sqrt(
        (z1 - z0) * (z1 - z0) +
        (y1 - y0) * (y1 - y0) +
        (x1 - x0) * (x1 - x0)
    ))) + 1;

    double dz = (z1 - z0) / steps;
    double dy = (y1 - y0) / steps;
    double dx = (x1 - x0) / steps;

    double inner_r2 = std::pow(radius * 0.5, 2);
    double outer_r2 = std::pow(radius, 2);

    for (int i = 0; i <= steps; ++i) {
        int cz = int(std::round(z0 + i * dz));
        int cy = int(std::round(y0 + i * dy));
        int cx = int(std::round(x0 + i * dx));

        for (int dz = -radius; dz <= radius; ++dz) {
            int zz = cz + dz; if (zz < 0 || zz >= nz) continue;
            for (int dy = -radius; dy <= radius; ++dy) {
                int yy = cy + dy; if (yy < 0 || yy >= ny) continue;
                for (int dx = -radius; dx <= radius; ++dx) {
                    int xx = cx + dx; if (xx < 0 || xx >= nx) continue;

                    double d2 = dx*dx + dy*dy + dz*dz;
                    int idx = (zz * ny + yy) * nx + xx;

                    if (d2 <= inner_r2 && !occ[idx]) {
                        labels[idx] = lumen_label;
                        occ[idx] = 1;
                        length_counter += 1.0;
                    } else if (d2 <= outer_r2 && !occ[idx]) {
                        labels[idx] = wall_label;
                        occ[idx] = 1;
                        length_counter += 1.0;
                    }
                }
            }
        }
    }
}
