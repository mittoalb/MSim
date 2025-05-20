#include "cells.h"
#include "geometry.h"
#include <vector>
#include <array>
#include <random>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <memory>


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
// iterative jitter + Fibonacci branching for axons
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
        int length = 20;  // fixed branch length
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

        // Fibonacci branching
        if (n.radius > 1) {
            int fib0 = 1, fib1 = 1;
            for (int i = 0; i < n.depth + 1; ++i) {
                int t = fib1;
                fib1 += fib0;
                fib0 = t;
            }
            int nb = std::min(max_branches_base + (fib1 % max_branches_base), max_branches_base);
            double child_r = n.radius * 0.5;
            for (int b = 0; b < nb; ++b) {
                int base = n.idx + b * 3;
                if (base + 3 > rng_len) break;
                double ndx = bdx + (rng_vals[base] - 0.5);
                double ndy = bdy + (rng_vals[base + 1] - 0.5);
                double ndz = bdz + (rng_vals[base + 2] - 0.5);
                stack.push_back({int(pz), int(py), int(px), n.depth + 1, base,
                                 ndx, ndy, ndz, int(child_r)});
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