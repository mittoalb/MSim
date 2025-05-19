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
    double jitter
) {
    double px = x0, py = y0, pz = z0;
    double dx = dx0, dy = dy0, dz = dz0;

    auto idx3 = [&](int z, int y, int x) { return (z * ny + y) * nx + x; };

    for (int s = 0; s < steps && rng_index + 3 < rng_len; ++s) {
        dx += dx * 0.2 + (rng_vals[rng_index++] - 0.5) * jitter;
        dy += dy * 0.2 + (rng_vals[rng_index++] - 0.5) * jitter;
        dz += dz * 0.2 + (rng_vals[rng_index++] - 0.5) * jitter;
        double norm = std::sqrt(dx * dx + dy * dy + dz * dz) + 1e-6;
        dx /= norm; dy /= norm; dz /= norm;

        px += dx; py += dy; pz += dz;
        int xi = int(std::round(px));
        int yi = int(std::round(py));
        int zi = int(std::round(pz));
        if (zi < 0 || zi >= nz || yi < 0 || yi >= ny || xi < 0 || xi >= nx)
            break;

        carve_ball(labels, occ, nz, ny, nx, zi, yi, xi, radius, /*AXON*/ 5, total_length);
    }

    return {int(std::round(pz)), int(std::round(py)), int(std::round(px))};
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
void connect_cells_with_dendrites(
    uint8_t* labels,
    uint8_t* occ,
    double& total_length,
    int nz, int ny, int nx,
    const std::vector<std::array<int, 3>>& centers,
    int dend_radius,
    int dend_depth,
    int max_branches,
    const float* rng_vals,
    int rng_len,
    int& rng_index
) {
    int n = (int)centers.size();
    std::vector<double> d2(n * n);

    // Compute pairwise squared distances
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j) {
            int dz = centers[j][0] - centers[i][0];
            int dy = centers[j][1] - centers[i][1];
            int dx = centers[j][2] - centers[i][2];
            d2[i * n + j] = d2[j * n + i] = double(dz * dz + dy * dy + dx * dx);
        }

    // MST using Prim's algorithm
    std::vector<bool> used(n, false);
    used[0] = true;
    std::vector<std::pair<int, int>> edges;
    for (int k = 0; k < n - 1; ++k) {
        double best = 1e300;
        int u_best = -1, v_best = -1;
        for (int u = 0; u < n; ++u) if (used[u]) {
            for (int v = 0; v < n; ++v) if (!used[v] && d2[u * n + v] < best) {
                best = d2[u * n + v];
                u_best = u;
                v_best = v;
            }
        }
        edges.emplace_back(u_best, v_best);
        used[v_best] = true;
    }

    // Grow dendrites between each connected cell pair
    for (auto& e : edges) {
        auto& A = centers[e.first];
        auto& B = centers[e.second];

        double dx = B[2] - A[2];
        double dy = B[1] - A[1];
        double dz = B[0] - A[0];
        double norm = std::sqrt(dx*dx + dy*dy + dz*dz) + 1e-6;
        dx /= norm; dy /= norm; dz /= norm;

        grow_dendrites_from(
            labels, occ, nz, ny, nx,
            A[0], A[1], A[2],
            dend_radius,
            dend_depth,
            max_branches,
            rng_vals, rng_len,
            rng_index,
            total_length,
            /* label = */ 7
        );
        rng_index += 128;
    }
}

// Main functions to add neuron
//──────────────────────────────────────────────────────────────────────────────
double add_neurons(
    uint8_t* labels,
    uint8_t* occ,
    int nz, int ny, int nx,
    const std::array<double, 3>& voxel_size,
    int num_cells,
    const std::array<double, 2>& cell_radius_range,
    const std::array<double, 2>& axon_dia_range,
    int max_depth
) {
    double total_length = 0.0;

    double dz = voxel_size[0], dy = voxel_size[1], dx = voxel_size[2];
    double px = (dz + dy + dx) / 3.0;

    std::mt19937_64 gen(std::random_device{}());
    std::uniform_int_distribution<int> Dz(0, nz - 1), Dy(0, ny - 1), Dx(0, nx - 1);
    std::uniform_real_distribution<double>
        Dr_cell(cell_radius_range[0], cell_radius_range[1]),
        Dr_axon(axon_dia_range[0], axon_dia_range[1]);
    std::uniform_real_distribution<float> Dr(0.0f, 1.0f);

    const int RNG_LEN = 1'000'000;
    std::unique_ptr<float[]> rng_vals(new float[RNG_LEN]);
    for (int i = 0; i < RNG_LEN; ++i) rng_vals[i] = Dr(gen);

    int rng_index = 0;
    std::vector<std::array<int, 3>> centers;
    centers.reserve(num_cells);

    int max_retries = 1000;
    int placed = 0;

    while (placed < num_cells && max_retries-- > 0) {
        int zc = Dz(gen), yc = Dy(gen), xc = Dx(gen);
        double soma_diam = Dr_cell(gen);
        int rz = std::max(2, int(std::round(soma_diam / dz / 2.0)));
        int ry = std::max(2, int(std::round(soma_diam / dy / 2.0)));
        int rx = std::max(2, int(std::round(soma_diam / dx / 2.0)));

        if (!can_place_ellipsoid(occ, nz, ny, nx, zc, yc, xc, rz, ry, rx)) continue;

        centers.push_back({zc, yc, xc});
        carve_ellipsoid(labels, occ, nz, ny, nx, zc, yc, xc, rz, ry, rx, /*CELL*/ 6, total_length);
        carve_ellipsoid(labels, occ, nz, ny, nx, zc, yc, xc, rz / 2, ry / 2, rx / 2, /*NUCL*/ 2, total_length);

        // Dendrites from soma
        int NUM_DENDRITES = std::clamp(int((rz + ry + rx) / 3.0), 2, 6);
        int dend_depth = std::clamp(int(rz * 0.5), 2, 12);
        int dend_fanout = std::clamp(10 - rz, 2, 6);
        int R_dend = std::max(1, int((rz + ry + rx) / 3.0 * 0.4));

        for (int d = 0; d < NUM_DENDRITES; ++d) {
            grow_dendrites_from(
                labels, occ, nz, ny, nx,
                zc, yc, xc,
                R_dend,
                dend_depth,
                dend_fanout,
                rng_vals.get(), RNG_LEN,
                rng_index,
                total_length,
                /*label=*/7
            );
            rng_index += 64;
        }

        // Axon trunk
        double axon_dia_um = Dr_axon(gen);
        int R_axon = std::clamp(int(std::round(axon_dia_um / px / 2.0)), 1, 6);

        double dx0 = Dr(gen) - 0.5;
        double dy0 = Dr(gen) - 0.5;
        double dz0 = Dr(gen) - 0.5;
        double norm = std::sqrt(dx0 * dx0 + dy0 * dy0 + dz0 * dz0) + 1e-6;
        dx0 /= norm; dy0 /= norm; dz0 /= norm;

        const int AXON_STEPS = 5000;
        auto [za, ya, xa] = draw_axon_tube(
            labels, occ, nz, ny, nx,
            zc, yc, xc,
            dx0, dy0, dz0,
            AXON_STEPS,
            R_axon,
            rng_vals.get(), RNG_LEN,
            rng_index,
            total_length,
            0.3  // jitter
        );
        rng_index += 64;

        // Dendrites at axon terminal
        grow_dendrites_from(
            labels, occ, nz, ny, nx,
            za, ya, xa,
            R_dend,
            max_depth,
            2,
            rng_vals.get(), RNG_LEN,
            rng_index,
            total_length,
            /*label=*/7
        );
        rng_index += 64;

        placed++;
    }

    // Dendritic connections between neurons (MST)
    connect_cells_with_dendrites(
        labels, occ, total_length,
        nz, ny, nx,
        centers,
        /*dend_radius=*/2,
        /*dend_depth=*/6,
        /*max_branches=*/2,
        rng_vals.get(), RNG_LEN,
        rng_index
    );

    return total_length;
}

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

    for (int attempt = 0; attempt < num_glia * 20 && placed < num_glia; ++attempt) {
        int zc = Dz(gen), yc = Dy(gen), xc = Dx(gen);
        int rz = Dr(gen), ry = Dr(gen), rx = Dr(gen);

        if (!can_place_ellipsoid(occ, nz, ny, nx, zc, yc, xc, rz, ry, rx))
            continue;

        // Carve glial soma as ellipsoid
        carve_ellipsoid(labels, occ, nz, ny, nx, zc, yc, xc, rz, ry, rx, GLIA_LABEL, total_length);

        // Grow 3–5 short branched dendrite-like arbors
        int num_procs = 3 + (rng_vals[rng_index++ % rng_len] * 3);
        for (int i = 0; i < num_procs; ++i) {
            double dx = Drf(gen) - 0.5;
            double dy = Drf(gen) - 0.5;
            double dz = Drf(gen) - 0.5;
            double norm = std::sqrt(dx*dx + dy*dy + dz*dz) + 1e-6;
            dx /= norm; dy /= norm; dz /= norm;

            grow_dendrites_from(
                labels, occ, nz, ny, nx,
                zc, yc, xc,
                std::max(1, int(0.3 * std::min({rz, ry, rx}))),
                dend_depth,
                dend_branches,
                rng_vals, rng_len,
                rng_index,
                total_length,
                GLIA_PROC_LABEL
            );
            rng_index += 64;
        }

        ++placed;
    }
}
