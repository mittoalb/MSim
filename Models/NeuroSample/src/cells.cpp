#include "cells.h"
#include "geometry.h"
#include <vector>
#include <array>
#include <random>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <memory>


#include <queue>
#include <unordered_map>
#include <cstring>


//──────────────────────────────────────────────────────────────────────────────
// 6‐neighborhood offsets, needed by multiple routines:
static const int N6[6][3] = {
  {+1,0,0}, {-1,0,0},
  {0,+1,0}, {0,-1,0},
  {0,0,+1}, {0,0,-1}
};
//──────────────────────────────────────────────────────────────────────────────
// Forward‐declare the helper
static void compute_local_normal(
    const uint8_t* occ, int nz, int ny, int nx,
    int z, int y, int x,
    double &dz, double &dy, double &dx
);

//──────────────────────────────────────────────────────────────────────────────
inline void safe_advance_rng(int& idx, int advance, int rng_len) {
    idx = (idx + advance < rng_len) ? idx + advance : 0;
}
//──────────────────────────────────────────────────────────────────────────────
// 1D Gaussian kernel for macro-region warping
std::vector<double> gaussian_kernel(int radius, double sigma) {
    int size = 2 * radius + 1;
    std::vector<double> kernel(size);
    double sum = 0.0;
    for (int i = 0; i < size; ++i) {
        double x = i - radius;
        kernel[i] = std::exp(-(x * x) / (2 * sigma * sigma));
        sum += kernel[i];
    }
    for (auto &v : kernel)
        v /= sum;
    return kernel;
}

//──────────────────────────────────────────────────────────────────────────────
// carve 3D Gaussian‐blurred noise into macro‐regions
void add_macroregions(
    uint8_t *labels,
    uint8_t *occ,               // occupancy array (0 = free, 1 = occupied)
    int nz, int ny, int nx,
    int macro_regions,
    double region_smoothness
) {
    // 1) clear
    std::fill(occ, occ + nz * ny * nx, uint8_t(0));

    // 2) generate uniform noise
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    std::vector<double> noise(nz * ny * nx);
    for (auto &v : noise) v = dist(gen);

    // 3) build 1D Gaussian kernel
    int radius = std::min( int(region_smoothness * 3), std::min({nz,ny,nx})/10 );
    auto kernel = gaussian_kernel(radius, region_smoothness);

    // helper lambda to index
    auto idx3 = [&](int z,int y,int x){ return (z*ny + y)*nx + x; };

    // 4) separable blur in X
    {
        std::vector<double> tmp(nx);
        for (int z = 0; z < nz; ++z) {
            for (int y = 0; y < ny; ++y) {
                for (int x = 0; x < nx; ++x) {
                    double sum = 0;
                    for (int k = -radius; k <= radius; ++k) {
                        int xx = std::clamp(x + k, 0, nx - 1);
                        sum += noise[idx3(z,y,xx)] * kernel[k + radius];
                    }
                    tmp[x] = sum;
                }
                for (int x = 0; x < nx; ++x)
                    noise[idx3(z,y,x)] = tmp[x];
            }
        }
    }

    // 5) separable blur in Y
    {
        std::vector<double> tmp(ny);
        for (int z = 0; z < nz; ++z) {
            for (int x = 0; x < nx; ++x) {
                for (int y = 0; y < ny; ++y) {
                    double sum = 0;
                    for (int k = -radius; k <= radius; ++k) {
                        int yy = std::clamp(y + k, 0, ny - 1);
                        sum += noise[idx3(z,yy,x)] * kernel[k + radius];
                    }
                    tmp[y] = sum;
                }
                for (int y = 0; y < ny; ++y)
                    noise[idx3(z,y,x)] = tmp[y];
            }
        }
    }

    // 6) separable blur in Z
    {
        std::vector<double> tmp(nz);
        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                for (int z = 0; z < nz; ++z) {
                    double sum = 0;
                    for (int k = -radius; k <= radius; ++k) {
                        int zz = std::clamp(z + k, 0, nz - 1);
                        sum += noise[idx3(zz,y,x)] * kernel[k + radius];
                    }
                    tmp[z] = sum;
                }
                for (int z = 0; z < nz; ++z)
                    noise[idx3(z,y,x)] = tmp[z];
            }
        }
    }

    // 7) quantize noise into equal‐width bins → labels 1..macro_regions
    for (int z = 0; z < nz; ++z) {
        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                int i = idx3(z,y,x);
                double v = noise[i];        // in [0,1)
                int bin = std::min(int(v * macro_regions), macro_regions - 1);
                labels[i] = uint8_t(bin + 1);
                occ[i]    = 1;
            }
        }
    }
}


//──────────────────────────────────────────────────────────────────────────────
// Modified draw_axon_tube with local bundling and curvature
std::array<int,3> draw_axon_tube(
    uint8_t* labels, uint8_t* occ,
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
) {
    double pz=z0, py=y0, px=x0;
    double vx=dx, vy=dy, vz=dz;
    double m = std::sqrt(vx*vx+vy*vy+vz*vz)+1e-6;
    vx/=m; vy/=m; vz/=m;
    int zi=z0, yi=y0, xi=x0;

    for(int s=0; s<steps; ++s) {
        // convergence pull
        if(converge_pts) {
            for(auto& tgt: *converge_pts) {
                double tz=tgt[0]-pz, ty=tgt[1]-py, tx=tgt[2]-px;
                double d = std::sqrt(tx*tx+ty*ty+tz*tz);
                if(d<converge_radius && d>1e-6) {
                    double w = (converge_strength>=1.0?1.0:converge_strength);
                    vx = (1-w)*vx + w*(tx/d);
                    vy = (1-w)*vy + w*(ty/d);
                    vz = (1-w)*vz + w*(tz/d);
                    m = std::sqrt(vx*vx+vy*vy+vz*vz)+1e-6;
                    vx/=m; vy/=m; vz/=m;
                    break;
                }
            }
        }

        // Brownian jitter
        if(rng_index+3 < rng_len) {
            double rx = rng_vals[rng_index++] - 0.5;
            double ry = rng_vals[rng_index++] - 0.5;
            double rz = rng_vals[rng_index++] - 0.5;
            vx = persist*vx + jitter*rx;
            vy = persist*vy + jitter*ry;
            vz = persist*vz + jitter*rz;
            m = std::sqrt(vx*vx+vy*vy+vz*vz)+1e-6;
            vx/=m; vy/=m; vz/=m;
        }

        // advance
        double pz1=pz+vz, py1=py+vy, px1=px+vx;
        carve_cylinder_segment(
            labels, occ, nz, ny, nx,
            pz, py, px,
            pz1, py1, px1,
            radius,
            wall_label, lumen_label,
            total_length
        );
        pz=pz1; py=py1; px=px1;
        zi=int(std::round(pz));
        yi=int(std::round(py));
        xi=int(std::round(px));
        if(zi<0||zi>=nz||yi<0||yi>=ny||xi<0||xi>=nx) break;
    }
    return {zi, yi, xi};
}

//──────────────────────────────────────────────────────────────────────────────
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
) {
    struct Node {
        int z, y, x, depth, idx;
        double dx, dy, dz, radius;
    };

    std::vector<Node> stack;
    stack.reserve(64);

    // Initial random direction
    double dx = rng_vals[rng_index++] - 0.5;
    double dy = rng_vals[rng_index++] - 0.5;
    double dz = rng_vals[rng_index++] - 0.5;
    double norm = std::sqrt(dx * dx + dy * dy + dz * dz) + 1e-6;
    dx /= norm; dy /= norm; dz /= norm;

    stack.push_back({z0, y0, x0, 0, rng_index, dx, dy, dz, double(base_radius)});

    while (!stack.empty()) {
        Node n = stack.back(); stack.pop_back();
        if (n.depth >= max_depth || n.idx + 3 > rng_len) continue;

        // Normalize direction
        double norm = std::sqrt(n.dx * n.dx + n.dy * n.dy + n.dz * n.dz) + 1e-6;
        double bdx = n.dx / norm, bdy = n.dy / norm, bdz = n.dz / norm;

        double pos[3] = {double(n.z), double(n.y), double(n.x)};
        double dir[3] = {bdz, bdy, bdx};
        double tmax = 1e9;
        for (int i = 0; i < 3; ++i) {
            double d = dir[i], p = pos[i];
            int lim = (i == 0 ? nz : i == 1 ? ny : nx);
            if (d > 0) tmax = std::min(tmax, ((lim - 1) - p) / d);
            else if (d < 0) tmax = std::min(tmax, -p / d);
        }
        int length = std::min(int(tmax), 50);

        double pz = pos[0], py = pos[1], px = pos[2];
        for (int i = 0; i < length; ++i) {
            if ((i % 5) == 0) {
                bdx += (rng_vals[n.idx] - 0.5) * 1.0;
                bdy += (rng_vals[n.idx + 1] - 0.5) * 1.0;
                bdz += (rng_vals[n.idx + 2] - 0.5) * 1.0;
                n.idx += 3;
                double m2 = std::sqrt(bdx * bdx + bdy * bdy + bdz * bdz) + 1e-6;
                bdx /= m2; bdy /= m2; bdz /= m2;
            }

            pz += bdz;
            py += bdy;
            px += bdx;

            int zi = int(std::round(pz));
            int yi = int(std::round(py));
            int xi = int(std::round(px));
            if (zi < 0 || zi >= nz || yi < 0 || yi >= ny || xi < 0 || xi >= nx) break;
            int off = (zi * ny + yi) * nx + xi;
            if (!occ[off]) {
                carve_ball(labels, occ, nz, ny, nx, zi, yi, xi, int(n.radius), label, total_length);
            }
        }

        // Branching logic
        if (n.radius > 1.0) {
            int nb = std::min(2 + (n.depth % max_branch_base), max_branch_base);
            double child_r = n.radius * 0.6;
            for (int b = 0; b < nb; ++b) {
                int base = n.idx + b * 3;
                if (base + 3 > rng_len) break;
                double ndx = bdx + (rng_vals[base] - 0.5) * 1.0;
                double ndy = bdy + (rng_vals[base + 1] - 0.5) * 1.0;
                double ndz = bdz + (rng_vals[base + 2] - 0.5) * 1.0;
                stack.push_back({int(pz), int(py), int(px), n.depth + 1, base, ndx, ndy, ndz, child_r});
            }
        }
    }
}

//──────────────────────────────────────────────────────────────────────────────
void branch_axons(
    uint8_t* labels,
    uint8_t* occ,
    int nz, int ny, int nx,
    int z, int y, int x,
    int depth, int idx,
    double dx, double dy, double dz,
    int radius,
    int max_depth,
    const float* rng_vals, int rng_len,
    double& total_length,
    int max_branches_base,
    uint8_t label 
) {
    struct Node {
        int z, y, x, depth, idx;
        double dx, dy, dz;
        int radius;
    };
    std::vector<Node> stack;
    stack.reserve(64);
    stack.push_back({z, y, x, depth, idx, dx, dy, dz, radius});

    while (!stack.empty()) {
        auto n = stack.back();
        stack.pop_back();
        if (n.depth >= max_depth || n.idx + 3 > rng_len) continue;

        double norm = std::sqrt(n.dx * n.dx + n.dy * n.dy + n.dz * n.dz) + 1e-6;
        double bdx = n.dx / norm, bdy = n.dy / norm, bdz = n.dz / norm;

        double pz = n.z, py = n.y, px = n.x;
        int length = 20;
        for (int i = 0; i < length; ++i) {
            if ((i % 5) == 0 && n.idx + 3 <= rng_len) {
                bdx += (rng_vals[n.idx++] - 0.5) * 0.6;
                bdy += (rng_vals[n.idx++] - 0.5) * 0.6;
                bdz += (rng_vals[n.idx++] - 0.5) * 0.6;
                double m = std::sqrt(bdx * bdx + bdy * bdy + bdz * bdz) + 1e-6;
                bdx /= m; bdy /= m; bdz /= m;
            }

            pz += bdz;
            py += bdy;
            px += bdx;

            int zi = int(std::round(pz));
            int yi = int(std::round(py));
            int xi = int(std::round(px));
            if (zi < 0 || zi >= nz || yi < 0 || yi >= ny || xi < 0 || xi >= nx)
                break;

            int off = (zi * ny + yi) * nx + xi;
            if (!occ[off]) {
                carve_ball(labels, occ, nz, ny, nx, zi, yi, xi, n.radius, label, total_length);
            }
        }

        // volume bounds check — no branching at border
        int zi2 = int(std::round(pz));
        int yi2 = int(std::round(py));
        int xi2 = int(std::round(px));
        if (zi2 <= 0 || zi2 >= nz - 1 ||
            yi2 <= 0 || yi2 >= ny - 1 ||
            xi2 <= 0 || xi2 >= nx - 1)
            continue;

        // Fibonacci branching
        if (n.radius > 1) {
            int fib0 = 1, fib1 = 1;
            for (int i = 0; i < n.depth + 1; ++i) {
                int t = fib1; fib1 += fib0; fib0 = t;
            }
            int nb = std::min(max_branches_base + (fib1 % max_branches_base), max_branches_base);
            double child_r = n.radius * 0.5;

            for (int b = 0; b < nb; ++b) {
                int base = n.idx + b * 3;
                if (base + 3 > rng_len) break;
                double ndx = bdx + (rng_vals[base]     - 0.5);
                double ndy = bdy + (rng_vals[base + 1] - 0.5);
                double ndz = bdz + (rng_vals[base + 2] - 0.5);
                stack.push_back({
                    zi2, yi2, xi2,
                    n.depth + 1,
                    base,
                    ndx, ndy, ndz,
                    int(child_r)
                });
            }
        }
    }
}


//──────────────────────────────────────────────────────────────────────────────
void connect_somas_with_synapses(
    uint8_t* labels,
    uint8_t* occ,
    int nz, int ny, int nx,
    const std::vector<std::array<int,3>>& centers,
    const float* rng_vals,
    int rng_len,
    int& rng_index,
    double& total_length,
    int axon_steps,
    int axon_radius,
    float jitter,
    float persist,
    int dend_branches,
    float extra_conn_prob,
    uint8_t synapse_label
) {
    for (const auto& soma : centers) {
        int zc = soma[0], yc = soma[1], xc = soma[2];

        // Sample initial random unit direction
        double dx0 = rng_vals[rng_index++] - 0.5;
        double dy0 = rng_vals[rng_index++] - 0.5;
        double dz0 = rng_vals[rng_index++] - 0.5;
        double m0  = std::sqrt(dx0*dx0 + dy0*dy0 + dz0*dz0) + 1e-6;
        dx0 /= m0; dy0 /= m0; dz0 /= m0;

        // Draw the main axon
        auto [za, ya, xa] = draw_axon_tube(
            labels, occ,
            nz, ny, nx,
            zc, yc, xc,
            dx0, dy0, dz0,
            axon_steps,
            axon_radius,
            rng_vals, rng_len, rng_index,
            total_length,
            jitter, persist,
            /* no convergence */ nullptr,
            0.0, 0.0,
            /* wall_label */ 8,
            /* lumen_label */ synapse_label
        );

        // Collateral branches
        for (int b = 0; b < dend_branches; ++b) {
            if (rng_vals[rng_index++] < extra_conn_prob) {
                double bdx = rng_vals[rng_index++] - 0.5;
                double bdy = rng_vals[rng_index++] - 0.5;
                double bdz = rng_vals[rng_index++] - 0.5;
                double mb  = std::sqrt(bdx*bdx + bdy*bdy + bdz*bdz) + 1e-6;
                bdx /= mb; bdy /= mb; bdz /= mb;

                draw_axon_tube(
                    labels, occ,
                    nz, ny, nx,
                    za, ya, xa,
                    bdx, bdy, bdz,
                    axon_steps/2,
                    axon_radius/2,
                    rng_vals, rng_len, rng_index,
                    total_length,
                    jitter, persist,
                    nullptr, 0.0, 0.0,
                    /* wall_label */ 8,
                    /* lumen_label */ synapse_label
                );
            }
        }

        // Stamp synapse label
        if (za>=0 && za<nz && ya>=0 && ya<ny && xa>=0 && xa<nx) {
            size_t idx = size_t(za)*ny*nx + size_t(ya)*nx + size_t(xa);
            labels[idx] = synapse_label;
        }
    }
}


//──────────────────────────────────────────────────────────────────────────────
void connect_glia_to_neurons(
    uint8_t* labels,
    int nz, int ny, int nx,
    const std::vector<std::array<int,3>>& neuron_centers,
    int contact_label,
    int contact_radius
) {
    const int stride_y = nx;
    const int stride_z = ny * nx;

    for (auto& nc : neuron_centers) {
        int zn = nc[0], yn = nc[1], xn = nc[2];
        for (int dz = -contact_radius; dz <= contact_radius; ++dz) {
            for (int dy = -contact_radius; dy <= contact_radius; ++dy) {
                for (int dx = -contact_radius; dx <= contact_radius; ++dx) {
                    int z = zn + dz, y = yn + dy, x = xn + dx;
                    if (z < 0 || z >= nz || y < 0 || y >= ny || x < 0 || x >= nx)
                        continue;
                    auto idx = z*stride_z + y*stride_y + x;
                    // assume glia processes are labeled GLIA_PROC_LABEL (8)
                    if (labels[idx] == 8) {
                        labels[idx] = contact_label;
                    }
                }
            }
        }
    }
}


// Main functions to add neuron
//──────────────────────────────────────────────────────────────────────────────
double add_neurons(
    uint8_t *labels,
    uint8_t *occ,
    int nz, int ny, int nx,
    const std::array<double, 3> &voxel_size,
    int num_cells,
    const std::array<double, 2> &cell_radius_range,
    const std::array<double, 2> &axon_dia_range,
    int max_depth,
    std::vector<std::array<int,3>>& centers
) {
    // Clear occupancy
    std::fill(occ, occ + nz*ny*nx, uint8_t(0));
    double total_length = 0.0;

    // Voxel size average
    double dz = voxel_size[0], dy = voxel_size[1], dx = voxel_size[2];
    double px = (dz + dy + dx) / 3.0;

    // RNG for placement and diameters
    std::mt19937_64 gen(std::random_device{}());
    std::uniform_int_distribution<int> Uz(0, nz-1), Uy(0, ny-1), Ux(0, nx-1);
    std::uniform_real_distribution<double> Ur_cell(cell_radius_range[0], cell_radius_range[1]);
    std::uniform_real_distribution<double> Ur_axon(axon_dia_range[0], axon_dia_range[1]);
    std::uniform_real_distribution<float> Urf(0.0f, 1.0f);

    // Pre-generate RNG buffer for axon
    constexpr int RNG_LEN = 1'000'000;
    auto rng_vals = std::make_unique<float[]>(RNG_LEN);
    for (int i = 0; i < RNG_LEN; ++i) {
        rng_vals[i] = Urf(gen);
    }
    int rng_index = 0;

    // Labels
    constexpr uint8_t SOMA_LABEL       = 5;
    constexpr uint8_t NUCLEUS_LABEL    = 7;
    constexpr uint8_t DEND_LABEL       = 7;
    constexpr uint8_t AXON_WALL_LABEL  = 8;
    constexpr uint8_t AXON_LUMEN_LABEL = 10;
    const int AXON_STEPS = 5000;

    for (int i = 0; i < num_cells; ++i) {
        // 1) Place soma & nucleus
        int zc = Uz(gen), yc = Uy(gen), xc = Ux(gen);
        centers.push_back({zc, yc, xc});

        int R_soma = std::max(4, int(std::round(Ur_cell(gen) / px)));
        int R_nucl = std::max(1, int(R_soma * 0.3));
        carve_ball(labels, occ, nz, ny, nx, zc, yc, xc, R_soma, SOMA_LABEL, total_length);
        carve_ball(labels, occ, nz, ny, nx, zc, yc, xc, R_nucl, NUCLEUS_LABEL, total_length);

        // 2) Grow dendrites
        int num_dend = std::clamp(int(R_soma * 0.5), 2, 6);
        int depth    = std::clamp(int(R_soma * 0.4), 2, 6);
        int fanout   = std::clamp(10 - R_soma/2, 2, 6);
        int R_dend   = std::max(1, int(R_soma * 0.4));
        for (int d = 0; d < num_dend; ++d) {
            grow_dendrites_from(
                labels, occ, nz, ny, nx,
                zc, yc, xc,
                R_dend, depth, fanout,
                rng_vals.get(), RNG_LEN, rng_index,
                total_length, DEND_LABEL
            );
            rng_index += 64;
        }

        // 3) Axon radius
        double axon_dia = Ur_axon(gen);
        int R_axon = std::clamp(int(std::round((axon_dia/px)/2.0)), 1, 6);

        // 4) Initial direction
        double dx0 = (Urf(gen) - 0.5)*0.2;
        double dy0 = 1.0 + (Urf(gen) - 0.5)*0.2;
        double dz0 = (Urf(gen) - 0.5)*0.2;
        double m0  = std::sqrt(dx0*dx0 + dy0*dy0 + dz0*dz0) + 1e-6;
        dx0/=m0; dy0/=m0; dz0/=m0;

        // 5) Convergence targets
        std::vector<std::array<double,3>> targets = {
            {50.0,10.0,20.0},
            {80.0,40.0,60.0}
        };
        double conv_rad = 30.0, conv_str = 1.0;

        // 6) Grow axon
        auto [zi, yi, xi] = draw_axon_tube(
            labels, occ, nz, ny, nx,
            zc, yc, xc,
            dx0, dy0, dz0,
            AXON_STEPS, R_axon,
            rng_vals.get(), RNG_LEN, rng_index,
            total_length,
            0.2, 0.8,
            &targets, conv_rad, conv_str,
            AXON_WALL_LABEL, AXON_LUMEN_LABEL
        );
        rng_index += 64;

        // 7) Collateral dendrite at axon tip
        grow_dendrites_from(
            labels, occ, nz, ny, nx,
            zi, yi, xi,
            R_dend, max_depth, 2,
            rng_vals.get(), RNG_LEN, rng_index,
            total_length, DEND_LABEL
        );
        rng_index += 64;
    }

    return total_length;
}

// Main functions to glial cells
//──────────────────────────────────────────────────────────────────────────────
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
) {
    std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<int> Dz(0, nz - 1), Dy(0, ny - 1), Dx(0, nx - 1);
    std::uniform_int_distribution<int> Dr(glia_radius_min, glia_radius_max);
    std::uniform_real_distribution<float> Drf(0.0f, 1.0f);

    const uint8_t GLIA_LABEL = 9;
    const uint8_t GLIA_PROC_LABEL = 8;

    int rng_index = 0;
    int placed = 0;
    int max_attempts = num_glia * 30;

    for (int attempt = 0; attempt < max_attempts && placed < num_glia; ++attempt) {
        int zc = Dz(gen), yc = Dy(gen), xc = Dx(gen);
        int rz = Dr(gen), ry = Dr(gen), rx = Dr(gen);

        safe_advance_rng(rng_index, 64, rng_len);
        if (!can_place_ellipsoid(occ, nz, ny, nx, zc, yc, xc, rz, ry, rx))
            continue;

        // Carve soma
        carve_ellipsoid(labels, occ, nz, ny, nx, zc, yc, xc, rz, ry, rx, GLIA_LABEL, total_length);

        // Create 3–5 dendrite-like short branches
        int num_procs = 3 + int(rng_vals[rng_index++ % rng_len] * 3);
        int root_radius = std::max(1, int(0.3 * std::min({rz, ry, rx})));

        for (int i = 0; i < num_procs; ++i) {
            double dx = Drf(gen) - 0.5;
            double dy = Drf(gen) - 0.5;
            double dz = Drf(gen) - 0.5;
            double norm = std::sqrt(dx * dx + dy * dy + dz * dz) + 1e-6;
            dx /= norm; dy /= norm; dz /= norm;

            grow_dendrites_from(
                labels, occ, nz, ny, nx,
                zc, yc, xc,
                root_radius,
                dend_depth,
                dend_branches,
                rng_vals, rng_len,
                rng_index,
                total_length,
                GLIA_PROC_LABEL
            );

            safe_advance_rng(rng_index, 64, rng_len);
        }

        ++placed;
    }
}


double add_glial(
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
    int& rng_index,
    double& total_length
) {
    // RNG for soma placement
    std::mt19937_64 gen(std::random_device{}());
    std::uniform_int_distribution<int> Uz(0, nz-1), Uy(0, ny-1), Ux(0, nx-1);
    std::uniform_int_distribution<int> Ur(glia_radius_min, glia_radius_max);

    constexpr uint8_t GLIA_LABEL      = 9;
    constexpr uint8_t GLIA_PROC_LABEL = 8;

    int placed = 0;
    int max_attempts = num_glia * 30;

    for (int attempt = 0; attempt < max_attempts && placed < num_glia; ++attempt) {
        // 1) Random center + ellipsoid radii
        int zc = Uz(gen), yc = Uy(gen), xc = Ux(gen);
        int rz = Ur(gen), ry = Ur(gen), rx = Ur(gen);

        // 2) Check occupancy
        if (!can_place_ellipsoid(occ, nz, ny, nx, zc, yc, xc, rz, ry, rx))
            continue;

        // 3) Carve glial soma
        carve_ellipsoid(
            labels, occ, nz, ny, nx,
            zc, yc, xc,
            rz, ry, rx,
            GLIA_LABEL,
            total_length
        );
        ++placed;

        // 4) Grow short processes
        int num_procs = 3 + int(rng_vals[rng_index] * 3);
        safe_advance_rng(rng_index, 1, rng_len);

        int root_radius = std::max(1, int(0.3 * std::min({rz, ry, rx})));

        for (int p = 0; p < num_procs; ++p) {
            grow_dendrites_from(
                labels, occ, nz, ny, nx,
                zc, yc, xc,
                root_radius,
                dend_depth,
                dend_branches,
                rng_vals, rng_len, rng_index,
                total_length,
                GLIA_PROC_LABEL
            );
            safe_advance_rng(rng_index, 64, rng_len);
        }
    }

    return total_length;
}

/// Thin a binary mask in-place by iteratively removing end-points
static void binary_thin(uint8_t* mask, int nz, int ny, int nx) {
    bool removed_any;
    do {
        removed_any = false;
        std::vector<size_t> to_remove;
        for (int z = 1; z < nz-1; ++z) {
            for (int y = 1; y < ny-1; ++y) {
                for (int x = 1; x < nx-1; ++x) {
                    size_t idx = size_t(z)*ny*nx + size_t(y)*nx + x;
                    if (!mask[idx]) continue;
                    int count = 0;
                    for (auto &d : N6) {
                        int zz = z + d[0], yy = y + d[1], xx = x + d[2];
                        size_t nidx = size_t(zz)*ny*nx + size_t(yy)*nx + xx;
                        if (mask[nidx]) ++count;
                    }
                    if (count == 1)  // end-point
                        to_remove.push_back(idx);
                }
            }
        }
        for (auto idx : to_remove) {
            mask[idx] = 0;
            removed_any = true;
        }
    } while (removed_any);
}

// ----------------------------------------------------------------------------
// Helper: compute local normal by central differences on occ[]
static void compute_local_normal(
    const uint8_t* occ,
    int nz, int ny, int nx,
    int z, int y, int x,
    double &dz, double &dy, double &dx
) {
    auto idx = [&](int zz,int yy,int xx){
        return size_t(zz)*ny*nx + size_t(yy)*nx + xx;
    };
    // clamp at boundaries:
    int zp = std::min(z+1, nz-1), zm = std::max(z-1, 0);
    int yp = std::min(y+1, ny-1), ym = std::max(y-1, 0);
    int xp = std::min(x+1, nx-1), xm = std::max(x-1, 0);

    dz = (occ[idx(zp,y,x)] - occ[idx(zm,y,x)]) * 0.5;
    dy = (occ[idx(z,y+1,x)] - occ[idx(z,y-1,x)]) * 0.5;
    dx = (occ[idx(z,y,xp)] - occ[idx(z,y,xm)]) * 0.5;
    double n = std::sqrt(dz*dz + dy*dy + dx*dx) + 1e-6;
    dz /= n; dy /= n; dx /= n;
}


// estimate a local normal by PCA on the 6-neighbor shell
static Vec3 estimate_normal(
    const uint8_t* labels, int nz,int ny,int nx,
    int z,int y,int x, uint8_t vessel_val
){
    std::vector<Vec3> pts;
    for(auto &d:N6){
        int zz=z+d[0], yy=y+d[1], xx=x+d[2];
        size_t idx = size_t(zz)*ny*nx + size_t(yy)*nx + xx;
        if (zz>=0&&zz<nz&&yy>=0&&yy<ny&&xx>=0&&xx<nx
            && labels[idx] != vessel_val)
        {
            pts.push_back({double(zz),double(yy),double(xx)});
        }
    }
    // if no background neighbor, just return some arbitrary
    if (pts.empty()) return {0,0,1};
    // compute centroid
    Vec3 C{0,0,0};
    for(auto &p:pts) C = C + p;
    C = C * (1.0/pts.size());
    // covariance on those few pts
    double cov[3][3] = {};
    for(auto &p:pts){
      Vec3 v = p - C;
      cov[0][0]+=v.z*v.z; cov[0][1]+=v.z*v.y; cov[0][2]+=v.z*v.x;
      cov[1][0]+=v.y*v.z; cov[1][1]+=v.y*v.y; cov[1][2]+=v.y*v.x;
      cov[2][0]+=v.x*v.z; cov[2][1]+=v.x*v.y; cov[2][2]+=v.x*v.x;
    }
    // pick the *smallest* eigenvector of cov as normal (simple 3×3 power‐method)
    Vec3 n{1,1,1};
    for(int it=0;it<10;++it){
      Vec3 m = {
        cov[0][0]*n.z + cov[0][1]*n.y + cov[0][2]*n.x,
        cov[1][0]*n.z + cov[1][1]*n.y + cov[1][2]*n.x,
        cov[2][0]*n.z + cov[2][1]*n.y + cov[2][2]*n.x
      };
      double nm = m.norm() + 1e-9;
      n = m * (1.0/nm);
    }
    return n.normalized();
}

// carve one half‐cylinder cell
bool carve_cell_on_wall(
    uint8_t* labels, uint8_t* occ,
    int nz, int ny, int nx,
    int z, int y, int x,
    const Vec3& nd,    // outward normal (unit)
    const Vec3& td,    // tangent along vessel (unit)
    double length,
    double radius,
    uint8_t cell_label
){
    Vec3 C{ double(z), double(y), double(x) };
    Vec3 P0 = C - td * (length * 0.5);
    Vec3 P1 = C + td * (length * 0.5);

    int steps = std::max(3, int(length * 2));
    int R = int(std::ceil(radius));

    // first, test for overlap & bounds
    for(int i = 0; i <= steps; ++i){
        double t = double(i) / steps;
        Vec3 Pi{
            P0.z * (1-t) + P1.z * t,
            P0.y * (1-t) + P1.y * t,
            P0.x * (1-t) + P1.x * t
        };
        int iz = int(std::round(Pi.z));
        int iy = int(std::round(Pi.y));
        int ix = int(std::round(Pi.x));
        for(int dz = -R; dz <= R; ++dz){
            for(int dy = -R; dy <= R; ++dy){
                for(int dx = -R; dx <= R; ++dx){
                    Vec3 Pj{
                        double(iz + dz),
                        double(iy + dy),
                        double(ix + dx)
                    };
                    Vec3 v{ Pj.z - Pi.z, Pj.y - Pi.y, Pj.x - Pi.x };
                    if (v.norm() > radius + 0.5) continue;
                    if (v.dot(nd) < 0) continue;
                    int zz = int(Pj.z), yy = int(Pj.y), xx = int(Pj.x);
                    if (zz<0||zz>=nz||yy<0||yy>=ny||xx<0||xx>=nx) 
                        return false;
                    size_t idx = size_t(zz)*ny*nx + size_t(yy)*nx + xx;
                    if (occ[idx]) return false;
                }
            }
        }
    }

    // now carve into labels & occ
    for(int i = 0; i <= steps; ++i){
        double t = double(i) / steps;
        Vec3 Pi{
            P0.z * (1-t) + P1.z * t,
            P0.y * (1-t) + P1.y * t,
            P0.x * (1-t) + P1.x * t
        };
        int iz = int(std::round(Pi.z));
        int iy = int(std::round(Pi.y));
        int ix = int(std::round(Pi.x));
        for(int dz = -R; dz <= R; ++dz){
            for(int dy = -R; dy <= R; ++dy){
                for(int dx = -R; dx <= R; ++dx){
                    Vec3 Pj{
                        double(iz + dz),
                        double(iy + dy),
                        double(ix + dx)
                    };
                    Vec3 v{ Pj.z - Pi.z, Pj.y - Pi.y, Pj.x - Pi.x };
                    if (v.norm() > radius + 0.5) continue;
                    if (v.dot(nd) < 0) continue;
                    int zz = int(Pj.z), yy = int(Pj.y), xx = int(Pj.x);
                    size_t idx = size_t(zz)*ny*nx + size_t(yy)*nx + xx;
                    labels[idx] = cell_label;
                    occ   [idx] = 1;
                }
            }
        }
    }

    return true;
}


// semi-cylinder carve: we simply sample points along the cylinder axis
void carve_semi_cylinder_wedge(
    uint8_t* labels, uint8_t* /*occ*/,
    int nz,int ny,int nx,
    double cz,double cy,double cx,
    double dz,double dy,double dx,
    double L, double R,
    uint8_t label,
    double& total_length
) {
    // Number of samples along length and around half-circumference
    int nL = std::max(3, int(std::round(L)));
    int nA = std::max(8, int(std::round(R*3)));
    for (int i = 0; i < nL; ++i) {
        double t = (i + 0.5) / nL * L;
        double zx = cz + dz*t,  yx = cy + dy*t,  xx = cx + dx*t;
        for (int a = 0; a < nA; ++a) {
            double theta = M_PI * (a / double(nA));  // half circle
            // pick a perp‐vector (simple cross)
            // choose arbitrary perp basis:
            double ux = -dy, uy = dz, uz = 0;  // roughly perpendicular
            double norm = std::sqrt(ux*ux+uy*uy+uz*uz) + 1e-6;
            ux/=norm; uy/=norm; uz/=norm;
            // second perp = cross(dir, u)
            double vx = dy*uz - dz*uy;
            double vy = dz*ux - dx*uz;
            double vz = dx*uy - dy*ux;
            // point on semicircle
            double px = zx + R*( ux*std::cos(theta) + vx*std::sin(theta) );
            double py = yx + R*( uy*std::cos(theta) + vy*std::sin(theta) );
            double pz = xx + R*( uz*std::cos(theta) + vz*std::sin(theta) );
            int iz = int(std::round(px));
            int iy = int(std::round(py));
            int ix = int(std::round(pz));
            if (iz<0||iz>=nz||iy<0||iy>=ny||ix<0||ix>=nx) continue;
            size_t idx = size_t(iz)*ny*nx + size_t(iy)*nx + ix;
            labels[idx] = label;
            total_length += 1.0;
        }
    }
}

// scatter cells onto every vessel-wall voxel
int add_endothelial_cells_direct(
    uint8_t* labels,
    uint8_t* occ,
    int nz, int ny, int nx,
    uint8_t vessel_wall_label,
    uint8_t cell_label,
    uint8_t nucleus_label,
    double max_cell_length,
    double max_cell_radius,
    int seed
) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> Ld(1, max_cell_length);
    std::uniform_real_distribution<double> Rd(0.5, max_cell_radius);

    double total_length = 0;
    int count = 0;

    for (int z=0; z<nz; ++z) {
      for (int y=0; y<ny; ++y) {
        for (int x=0; x<nx; ++x) {
          size_t idx = size_t(z)*ny*nx + size_t(y)*nx + x;
          if (labels[idx] != vessel_wall_label) continue;
          // random cell size
          double L = Ld(rng), R = Rd(rng);
          // pick a random tangent direction in 3D:
          double theta = rng() / double(rng.max()) * M_PI*2;
          double phi   = rng() / double(rng.max()) * M_PI;
          double dz = std::cos(phi) * std::sin(theta);
          double dy = std::sin(phi) * std::sin(theta);
          double dx = std::cos(theta);
          // carve the semi-cylinder (cell body)
          carve_semi_cylinder_wedge(
            labels, occ, nz,ny,nx,
            z,y,x, dz,dy,dx,
            L, R,
            cell_label, total_length
          );
          // optionally stamp a little nucleus sphere at the base:
          carve_ellipsoid(
            labels, occ, nz,ny,nx,
            z,y,x,
            int(std::round(R*0.5)),
            int(std::round(R*0.5)),
            int(std::round(R*0.5)),
            nucleus_label,
            total_length
          );
          ++count;
        }
      }
    }

    return count;
}

// ─────────────────────────────────────────────────────────────────────────────
int add_schwann_cells(
    uint8_t* labels,
    uint8_t* occ,
    int nz, int ny, int nx,
    uint8_t axon_label,
    uint8_t schwann_label,
    bool myelinated,
    double radius,
    double thickness
) {
    int count = 0;
    double dummy_len = 0.0;

    for (int z = 1; z < nz - 1; ++z) {
        for (int y = 1; y < ny - 1; ++y) {
            for (int x = 1; x < nx - 1; ++x) {
                size_t idx = (z * ny + y) * nx + x;
                if (labels[idx] != axon_label) continue;
                if (occ[idx] && labels[idx] != axon_label) continue;
                if (myelinated) {
                    carve_hollow_ellipsoid(
                        labels, occ,
                        nz, ny, nx,
                        z, y, x,
                        0.0, 0.0, 1.0,           // orientation vector
                        radius, thickness,
                        schwann_label,
                        dummy_len
                    );
                } else {
                    carve_ellipsoid(
                        labels, occ,
                        nz, ny, nx,
                        z, y, x,
                        radius, radius, radius,
                        schwann_label,
                        dummy_len
                    );
                }

                count++;
            }
        }
    }

    return count;
}
