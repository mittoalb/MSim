#include "cells.h"
#include "cball.h"
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
// carve warped horizontal layers, skipping occupied voxels
//──────────────────────────────────────────────────────────────────────────────
// carve warped horizontal layers, skipping occupied voxels
void add_macroregions(
    uint8_t *labels,
    uint8_t *occ,               // occupancy array (0 = free, 1 = occupied)
    int nz, int ny, int nx,
    int macro_regions,
    double region_smoothness
) {
    std::fill(occ, occ + nz * ny * nx, uint8_t(0));  // clear as 0
    int radius = int(region_smoothness * 10);
    auto kernel = gaussian_kernel(radius, region_smoothness);

    // 1) generate per‐slice random warp2d[z][x]
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    std::vector<std::vector<double>> warp2d(nz, std::vector<double>(nx));
    for (int z = 0; z < nz; ++z)
        for (int x = 0; x < nx; ++x)
            warp2d[z][x] = dist(gen);

    // 2) blur in X
    std::vector<double> temp(nx);
    for (int z = 0; z < nz; ++z) {
        for (int x = 0; x < nx; ++x) {
            double v = 0.0;
            for (int k = -radius; k <= radius; ++k) {
                int xx = std::clamp(x + k, 0, nx - 1);
                v += warp2d[z][xx] * kernel[k + radius];
            }
            temp[x] = v;
        }
        warp2d[z].swap(temp);
    }

    // 3) **NEW**: blur that result in Z so layers meander vertically
    std::vector<double> tempz(nz);
    for (int x = 0; x < nx; ++x) {
        // compute column warp2d[:,x] → tempz
        for (int z = 0; z < nz; ++z) {
            double v = 0.0;
            for (int k = -radius; k <= radius; ++k) {
                int zz = std::clamp(z + k, 0, nz - 1);
                v += warp2d[zz][x] * kernel[k + radius];
            }
            tempz[z] = v;
        }
        // write back blurred column
        for (int z = 0; z < nz; ++z) {
            warp2d[z][x] = tempz[z];
        }
    }

    // 4) carve your layers using the now-2D+Z‐blurred warp field
    double layer_thick = double(ny) / macro_regions;
    for (int z = 0; z < nz; ++z) {
        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                int idx = (z * ny + y) * nx + x;
                if (occ[idx]) continue;
                double warp = (warp2d[z][x] - 0.5) * (layer_thick * 0.4);
                for (int r = 0; r < macro_regions; ++r) {
                    double lo = r * layer_thick + warp;
                    double hi = (r + 1) * layer_thick + warp;
                    if (y >= lo && y < hi) {
                        labels[idx] = uint8_t(r + 1);
                        occ[idx]    = 1;  // mark occupied
                        break;
                    }
                }
            }
        }
    }
}

//──────────────────────────────────────────────────────────────────────────────
// iterative jitter + Fibonacci branching for axons
void branch_axons(
    uint8_t *labels,
    uint8_t *occ,               // occupancy array (0 = free, 1 = occupied)
    int nz, int ny, int nx,
    int z, int y, int x,
    int depth, int idx,
    double dx, double dy, double dz,
    int radius, int max_depth,
    const float *rng_vals, int rng_len,
    double &total_length
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

        // normalize
        double norm = std::sqrt(n.dx * n.dx + n.dy * n.dy + n.dz * n.dz) + 1e-6;
        double bdx = n.dx / norm, bdy = n.dy / norm, bdz = n.dz / norm;

        // how far forward before leaving volume
        double pos[3] = {double(n.z), double(n.y), double(n.x)};
        double dir[3] = {bdz, bdy, bdx};
        double tmax = 1e9;
        for (int i = 0; i < 3; ++i) {
            double d = dir[i], p = pos[i];
            int lim = (i == 0 ? nz : i == 1 ? ny : nx);
            if (d > 0) tmax = std::min(tmax, ((lim - 1) - p) / d);
            else if (d < 0) tmax = std::min(tmax, -p / d);
        }
        int length = std::min(int(tmax), 80);

        double pz = pos[0], py = pos[1], px = pos[2];
        int zi = 0, yi = 0, xi = 0;
        for (int i = 0; i < length; ++i) {
            if ((i % 5) == 0) {
                bdx += (rng_vals[n.idx] - 0.5) * 1.0;
                bdy += (rng_vals[n.idx + 1] - 0.5) * 1.5;
                bdz += (rng_vals[n.idx + 2] - 0.5) * 1.0;
                n.idx += 3;
                double m2 = std::sqrt(bdx * bdx + bdy * bdy + bdz * bdz) + 1e-6;
                bdx /= m2;
                bdy /= m2;
                bdz /= m2;
            }

            pz += bdz;
            py += bdy;
            px += bdx;
            zi = int(std::round(pz));
            yi = int(std::round(py));
            xi = int(std::round(px));
            if (zi < 0 || zi >= nz || yi < 0 || yi >= ny || xi < 0 || xi >= nx) break;
            int off = (zi * ny + yi) * nx + xi;
            if (!occ[off]) {
                carve_ball(labels, occ, nz, ny, nx, zi, yi, xi, n.radius, /*AXON*/ 5, total_length);
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
            int nb = std::min(2 + (fib1 % 6), 5);
            double child_r = n.radius * 0.4;
            for (int b = 0; b < nb; ++b) {
                int base = n.idx + b * 3;
                if (base + 3 > rng_len) break;
                double ndx = bdx + (rng_vals[base] - 0.5) * 1.0;
                double ndy = bdy + (rng_vals[base + 1] - 0.5) * 1.0;
                double ndz = bdz + (rng_vals[base + 2] - 0.5) * 1.0;
                stack.push_back({zi, yi, xi, n.depth + 1, base, ndx, ndy, ndz, int(child_r)});
            }
        }
    }
}

//──────────────────────────────────────────────────────────────────────────────
// straight trunk + branch
void draw_axons(
    uint8_t *labels,
    uint8_t *occ,               // occupancy array (0 = free, 1 = occupied)
    int nz, int ny, int nx,
    int z0, int y0, int x0,
    int z1, int y1, int x1,
    int max_depth,
    int base_radius,
    const float *rng_vals, int rng_len,
    double &total_length
) {
    int dz_ = z1 - z0, dy_ = y1 - y0, dx_ = x1 - x0;
    int L = int(std::ceil(std::sqrt(dz_ * dz_ + dy_ * dy_ + dx_ * dx_))) + 1;

    double pz = z0, py = y0, px = x0;
    double step_z = double(dz_) / (L - 1),
           step_y = double(dy_) / (L - 1),
           step_x = double(dx_) / (L - 1);

    for (int t = 0; t < L; ++t) {
        int zi = int(std::round(pz)),
            yi = int(std::round(py)),
            xi = int(std::round(px));
        int off = (zi * ny + yi) * nx + xi;
        if (!occ[off]) {
            carve_ball(labels, occ, nz, ny, nx, zi, yi, xi, base_radius, /*AXON*/ 5, total_length);
        }
        pz += step_z;
        py += step_y;
        px += step_x;
    }

    // now branch
    branch_axons(
        labels, occ, nz, ny, nx,
        z1, y1, x1,
        1, 0,
        step_x, step_y, step_z,
        base_radius, max_depth,
        rng_vals, rng_len,
        total_length
    );
}

//──────────────────────────────────────────────────────────────────────────────
// MST + draw_axons
void connect_cells(
    uint8_t *labels,
    uint8_t *occ,  // occupancy array (0 = free, 1 = occupied)
    double &total_length,
    int nz, int ny, int nx,
    const std::vector<std::array<int, 3>> &centers,
    int max_depth,
    double axon_dia_px,
    const float *rng_vals,
    int rng_len
) {
    int n = (int)centers.size();
    std::vector<double> d2(n * n);
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j) {
            int dz = centers[j][0] - centers[i][0];
            int dy = centers[j][1] - centers[i][1];
            int dx = centers[j][2] - centers[i][2];
            d2[i * n + j] = d2[j * n + i] = double(dz * dz + dy * dy + dx * dx);
        }

    std::vector<bool> used(n, false);
    used[0] = true;
    std::vector<std::pair<int, int>> edges;
    for (int k = 0; k < n - 1; ++k) {
        double best = 1e300;
        int bu = 0, bv = 1;
        for (int u = 0; u < n; ++u)
            if (used[u]) {
                for (int v = 0; v < n; ++v)
                    if (!used[v] && d2[u * n + v] < best) {
                        best = d2[u * n + v];
                        bu = u;
                        bv = v;
                    }
            }
        edges.emplace_back(bu, bv);
        used[bv] = true;
    }

    int R_ax = std::max(1, int(std::round(axon_dia_px / 2.0)));
    for (auto &e : edges) {
        auto &A = centers[e.first], &B = centers[e.second];
        draw_axons(
            labels, occ, nz, ny, nx,
            A[0], A[1], A[2], B[0], B[1], B[2],
            max_depth, R_ax,
            rng_vals, rng_len,
            total_length
        );
    }
}

//──────────────────────────────────────────────────────────────────────────────
// place bodies & nuclei, then MST-join with axons
double add_neurons(
    uint8_t *labels,
    uint8_t *occ,               // occupancy array (0 = free, 1 = occupied)
    int nz, int ny, int nx,
    const std::array<double, 3> &voxel_size,
    int num_cells,
    const std::array<double, 2> &cell_radius_range,
    const std::array<double, 2> &axon_dia_range,
    int max_depth
) {
    std::fill(occ, occ + nz * ny * nx, uint8_t(0));  // clear as 0
    double total_length = 0.0;

    double dz = voxel_size[0], dy = voxel_size[1], dx = voxel_size[2];
    double px = (dz + dy + dx) / 3.0;
    std::mt19937_64 gen(std::random_device{}());
    std::uniform_int_distribution<int> Dz(0, nz - 1), Dy(0, ny - 1), Dx(0, nx - 1);
    std::uniform_real_distribution<double>
        Dr_cell(cell_radius_range[0], cell_radius_range[1]),
        Dr_axon(axon_dia_range[0], axon_dia_range[1]);
    const int RNG_LEN = 1'000'000;
    std::unique_ptr<float[]> rng_vals(new float[RNG_LEN]);
    std::uniform_real_distribution<float> Dr(0.0f, 1.0f);
    for (int i = 0; i < RNG_LEN; ++i) rng_vals[i] = Dr(gen);

    std::vector<std::array<int, 3>> centers;
    centers.reserve(num_cells);
    for (int i = 0; i < num_cells; ++i) {
        int zc = Dz(gen), yc = Dy(gen), xc = Dx(gen);
        centers.push_back({zc, yc, xc});
        int R_px = std::max(1, int(std::round(Dr_cell(gen) / px)));
        carve_ball(labels, occ, nz, ny, nx, zc, yc, xc, R_px, /*CELL*/ 6, total_length);
        int r_in = std::max(1, int(R_px * 0.3));
        carve_ball(labels, occ, nz, ny, nx, zc, yc, xc, r_in, /*NUCL*/ 2, total_length);
    }

    double ax_px = Dr_axon(gen) / px;
    connect_cells(labels, occ, total_length,
        nz, ny, nx,
        centers, max_depth,
        ax_px,
        rng_vals.get(), RNG_LEN);

    return total_length;
}

