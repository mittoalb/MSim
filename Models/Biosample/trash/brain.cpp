#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <vector>
#include <array>
#include <random>
#include <cmath>
#include <cstdint>


namespace py = pybind11;

// Label constants
static constexpr uint8_t CELL_VAL    = 5;
static constexpr uint8_t NUCLEUS_VAL = 7;
static constexpr uint8_t AXON_VAL    = 8;
static constexpr uint8_t VESSEL_VAL  = 5;

//----------------------------------------------------------------------------------------------------------------
//──────────────────────────────────────────────────────────────────────────────
// carve_ball: fill a sphere at (z0,y0,x0), skip already‐occupied voxels
// and increment length_counter for each newly carved voxel.
//──────────────────────────────────────────────────────────────────────────────
inline void carve_ball(
    uint8_t*   labels,
    bool*      occ,
    int        nz, int ny, int nx,
    int        z0, int y0, int x0,
    int        radius,
    uint8_t    label,
    double&    length_counter
) {
    int rr = radius * radius;
    for (int dz = -radius; dz <= radius; ++dz) {
        int zz = z0 + dz;
        if (zz < 0 || zz >= nz) continue;
        int ddz = dz * dz;
        for (int dy = -radius; dy <= radius; ++dy) {
            int yy = y0 + dy;
            if (yy < 0 || yy >= ny) continue;
            int ddy = dy * dy;
            for (int dx = -radius; dx <= radius; ++dx) {
                if (ddz + ddy + dx*dx > rr) continue;
                int xx = x0 + dx;
                if (xx < 0 || xx >= nx) continue;
                int idx = (zz * ny + yy) * nx + xx;
                if (!occ[idx]) {
                    labels[idx] = label;
                    occ[idx]    = true;
                    length_counter += 1.0;
                }
            }
        }
    }
}
//----------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------
// branch_axons: iterative jitter+Fibonacci branching with occupancy + length counting
//----------------------------------------------------------------------------------------------------------------
void branch_axons(
    uint8_t*    labels,
    bool*       occ,
    int         nz, int ny, int nx,
    int         z,  int y,  int x,
    int         depth,
    int         idx,
    double      dx, double dy, double dz,
    int         radius,
    int         max_depth,
    const float* rng_vals,
    int         rng_len,
    double&     total_length
) {
    struct Node {
        int     z, y, x;
        int     depth, idx;
        double  dx, dy, dz;
        int     radius;
    };
    std::vector<Node> stack;
    stack.reserve(64);
    stack.push_back({z, y, x, depth, idx, dx, dy, dz, radius});

    // temp coords for carving & for child push
    int zi = 0, yi = 0, xi = 0;

    while (!stack.empty()) {
        Node n = stack.back(); stack.pop_back();
        // stop if too deep or ran out of random numbers
        if (n.depth >= max_depth || n.idx + 3 > rng_len) continue;

        // normalize direction
        double norm = std::sqrt(n.dx*n.dx + n.dy*n.dy + n.dz*n.dz) + 1e-6;
        double dx1 = n.dx / norm,
               dy1 = n.dy / norm,
               dz1 = n.dz / norm;

        // compute how far we can go before hitting a boundary
        double t_max = std::numeric_limits<double>::infinity();
        double pos[3] = { double(n.z), double(n.y), double(n.x) };
        double dir[3] = { dz1, dy1, dx1 };  // note: z is first index
        for (int i = 0; i < 3; ++i) {
            double d = dir[i], p = pos[i];
            int lim = (i == 0 ? nz : i == 1 ? ny : nx);
            if (d > 0)      t_max = std::min(t_max, ((lim - 1) - p) / d);
            else if (d < 0) t_max = std::min(t_max, -p / d);
        }
        int length = std::min(int(t_max), 80);

        // step along this branch
        double pz = pos[0], py = pos[1], px = pos[2];
        for (int i = 0; i < length; ++i) {
            // jitter every 5 steps
            if ((i % 5) == 0) {
                dx1 += (rng_vals[n.idx]   - 0.5) * 1.0;
                dy1 += (rng_vals[n.idx+1] - 0.5) * 1.5;
                dz1 += (rng_vals[n.idx+2] - 0.5) * 1.0;
                n.idx += 3;
                double m2 = std::sqrt(dx1*dx1 + dy1*dy1 + dz1*dz1) + 1e-6;
                dx1 /= m2; dy1 /= m2; dz1 /= m2;
            }

            // advance
            pz += dz1; py += dy1; px += dx1;
            zi = int(std::round(pz));
            yi = int(std::round(py));
            xi = int(std::round(px));

            if (zi >= 0 && zi < nz && yi >= 0 && yi < ny && xi >= 0 && xi < nx) {
                carve_ball(
                  labels, occ,
                  nz, ny, nx,
                  zi, yi, xi,
                  n.radius,
                  AXON_VAL,
                  total_length
                );
            }
        }

        // Fibonacci‐style sparse branching
        if (n.radius > 1) {
            // build Fib number
            int fib0 = 1, fib1 = 1;
            for (int i = 0; i < n.depth + 1; ++i) {
                int t = fib1; fib1 += fib0; fib0 = t;
            }
            int nb = std::min(2 + (fib1 % 6), 5);
            for (int b = 0; b < nb; ++b) {
                int base = n.idx + b*3;
                if (base + 3 > rng_len) break;
                double ndx = dx1 + (rng_vals[base]   - 0.5) * 2.0;
                double ndy = dy1 + (rng_vals[base+1] - 0.5) * 2.0;
                double ndz = dz1 + (rng_vals[base+2] - 0.5) * 2.0;
                int cr = std::max(1, int(n.radius * (0.5 + 0.4 * rng_vals[base+2])));
                stack.push_back({ zi, yi, xi,
                                  n.depth+1, base,
                                  ndx, ndy, ndz,
                                  cr });
            }
        }
    }
}

//----------------------------------------------------------------------------------------------------------------
// draw_axons: straight trunk + branch with occupancy & length counting
//----------------------------------------------------------------------------------------------------------------
void draw_axons(
    uint8_t*    labels,
    bool*       occ,
    int         nz, int ny, int nx,
    int         z0, int y0, int x0,
    int         z1, int y1, int x1,
    int         max_depth,
    int         base_radius,
    const float* rng_vals,
    int         rng_len,
    double&     total_length
) {
    // 1) carve straight trunk along the line
    int dz_ = z1 - z0, dy_ = y1 - y0, dx_ = x1 - x0;
    int L = int(std::ceil(std::sqrt(dz_*dz_ + dy_*dy_ + dx_*dx_))) + 1;

    double pz     = z0,
           py     = y0,
           px     = x0;
    double step_z = double(dz_) / double(L - 1),
           step_y = double(dy_) / double(L - 1),
           step_x = double(dx_) / double(L - 1);

    for (int t = 0; t < L; ++t) {
        int zi = int(std::round(pz)),
            yi = int(std::round(py)),
            xi = int(std::round(px));
        carve_ball(
            labels, occ,
            nz, ny, nx,
            zi, yi, xi,
            base_radius,   // radius
            AXON_VAL,      // label
            total_length   // increments when carving new voxels
        );
        pz += step_z;
        py += step_y;
        px += step_x;
    }

    // 2) now branch off from the end of the trunk
    branch_axons(
        labels, occ,
        nz, ny, nx,
        z1, y1, x1,      // start coordinates at trunk end
        1,               // initial depth
        0,               // initial RNG index
        step_x, step_y, step_z,
        base_radius,
        max_depth,
        rng_vals, rng_len,
        total_length     // pass down the same length counter
    );
}



//----------------------------------------------------------------------------------------------------------------
//──────────────────────────────────────────────────────────────────────────────
// connect_cells: MST + draw_axons with occupancy & length counting
//──────────────────────────────────────────────────────────────────────────────
//----------------------------------------------------------------------------------------------------------------
// connect_cells: MST + draw_axons (with occupancy & length counting)
void connect_cells(
    uint8_t*                                labels,
    bool*                                   occ,
    double&                                 length_counter,
    int                                     nz,
    int                                     ny,
    int                                     nx,
    const std::vector<std::array<int,3>>&   centers,
    int                                     max_depth,
    double                                  axon_dia_px,
    const float*                            rng_vals,
    int                                     rng_len
) {
    int n = (int)centers.size();
    // 1) compute all‐pairs squared distances
    std::vector<double> d2(n * n);
    for(int i = 0; i < n; ++i) {
        for(int j = i+1; j < n; ++j) {
            int dz = centers[j][0] - centers[i][0];
            int dy = centers[j][1] - centers[i][1];
            int dx = centers[j][2] - centers[i][2];
            double dist2 = double(dz)*dz + double(dy)*dy + double(dx)*dx;
            d2[i*n + j] = d2[j*n + i] = dist2;
        }
    }

    // 2) Prim’s MST
    std::vector<bool> used(n,false);
    used[0] = true;
    std::vector<std::pair<int,int>> edges;
    edges.reserve(n-1);
    for(int k = 0; k < n-1; ++k) {
        double best = 1e300;
        int bu = 0, bv = 1;
        for(int u = 0; u < n; ++u) if(used[u]) {
            for(int v = 0; v < n; ++v) if(!used[v]) {
                double d = d2[u*n + v];
                if(d < best) {
                    best = d;
                    bu = u; bv = v;
                }
            }
        }
        edges.emplace_back(bu,bv);
        used[bv] = true;
    }

    // 3) carve each edge with draw_axons (passing occ + length_counter)
    int R_ax = std::max(1, int(std::round(axon_dia_px / 2.0)));
    for (auto &e : edges) {
        const auto &A = centers[e.first];
        const auto &B = centers[e.second];
        draw_axons(
            /*labels=*/        labels,
            /*occ=*/           occ,
            /*nz=*/            nz,
            /*ny=*/            ny,
            /*nx=*/            nx,
            /*z0,y0,x0=*/      A[0], A[1], A[2],
            /*z1,y1,x1=*/      B[0], B[1], B[2],
            /*max_depth=*/     max_depth,
            /*base_radius=*/   R_ax,
            /*rng_vals=*/      rng_vals,
            /*rng_len=*/       rng_len,
            /*total_length=*/  length_counter
        );
    }
}



//----------------------------------------------------------------------------------------------------------------
//──────────────────────────────────────────────────────────────────────────────
// add_neurons: bodies, nuclei, and full axon with occupancy + length count
//──────────────────────────────────────────────────────────────────────────────
//----------------------------------------------------------------------------------------------------------------
// add_neurons: bodies, nuclei, and full axon (with occupancy + length counting)
double add_neurons(
    py::array_t<uint8_t, py::array::c_style> labels,
    py::array_t<bool,    py::array::c_style> occupied,
    std::array<double,3>                    voxel_size,
    int                                     num_cells,
    std::array<double,2>                    cell_radius_range,
    std::array<double,2>                    axon_dia_range,
    int                                     max_depth
) {
    // 1) Grab raw pointers + dims
    auto bufL = labels.request();
    auto bufO = occupied.request();
    uint8_t* ptrL = static_cast<uint8_t*>(bufL.ptr);
    bool*    ptrO = static_cast<bool   *>(bufO.ptr);
    int nz = bufL.shape[0], ny = bufL.shape[1], nx = bufL.shape[2];

    // 2) Clear occupancy + length counter
    std::fill(ptrO, ptrO + nz*ny*nx, false);
    double total_length = 0.0;

    // 3) Setup RNG for bodies + axons
    double dz = voxel_size[0], dy = voxel_size[1], dx = voxel_size[2];
    double px = (dz + dy + dx) / 3.0;
    std::mt19937_64 gen(std::random_device{}());
    std::uniform_int_distribution<int> dist_z(0, nz-1),
                                     dist_y(0, ny-1),
                                     dist_x(0, nx-1);
    std::uniform_real_distribution<double> 
        dist_cellR(cell_radius_range[0], cell_radius_range[1]),
        dist_axonD(axon_dia_range[0],   axon_dia_range[1]);
    const int rng_len = 1'000'000;
    std::unique_ptr<float[]> rng_vals(new float[rng_len]);
    std::uniform_real_distribution<float> dist_f(0.0f, 1.0f);
    for(int i = 0; i < rng_len; ++i) 
        rng_vals[i] = dist_f(gen);

    // 4) Place bodies + nuclei
    std::vector<std::array<int,3>> centers;
    centers.reserve(num_cells);
    for(int i = 0; i < num_cells; ++i) {
        int zc = dist_z(gen), yc = dist_y(gen), xc = dist_x(gen);
        centers.push_back({zc,yc,xc});

        int R_px = std::max(1, int(std::round(dist_cellR(gen)/px)));
        carve_ball(ptrL, ptrO, nz, ny, nx,
                   zc, yc, xc, R_px, CELL_VAL,    total_length);

        int r_inner = std::max(1, int(R_px * 0.3));
        carve_ball(ptrL, ptrO, nz, ny, nx,
                   zc, yc, xc, r_inner, NUCLEUS_VAL, total_length);
    }

    // 5) Grow axons along MST
    double ax_d = dist_axonD(gen) / px;
    connect_cells(
        ptrL, ptrO, total_length,
        nz, ny, nx,
        centers,
        max_depth,
        ax_d,
        rng_vals.get(), rng_len
    );

    // 6) Return total carved length
    return total_length;
}


//----------------------------------------------------------------------------------------------------------------
// Gaussian blur 1D helper (for macroregions warp)
static std::vector<double> gaussian_kernel(int radius, double sigma) {
    int size = 2*radius+1;
    std::vector<double> kernel(size);
    double sum = 0.0;
    for(int i=0;i<size;++i) {
        double x = i - radius;
        kernel[i] = std::exp(-(x*x)/(2*sigma*sigma));
        sum += kernel[i];
    }
    for(auto &v:kernel) v /= sum;
    return kernel;
}

//----------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------
// add_macroregions: warp and assign N horizontal layers, with occupancy
//------------------------------------------------------------------------------
//──────────────────────────────────────────────────────────────────────────────
// add_macroregions: warp & assign macro-regions without overwriting occupied
//──────────────────────────────────────────────────────────────────────────────
void add_macroregions(
    py::array_t<uint8_t, py::array::c_style> labels,
    py::array_t<bool,    py::array::c_style> occupied,
    int                                     macro_regions,
    double                                  region_smoothness
) {
    // 1) Grab raw pointers + dims
    auto bufL = labels.request();
    auto bufO = occupied.request();
    uint8_t* ptr   = static_cast<uint8_t*>(bufL.ptr);
    bool*    occ   = static_cast<bool   *>(bufO.ptr);
    int nz = bufL.shape[0], ny = bufL.shape[1], nx = bufL.shape[2];

    // 2) Clear occupancy mask
    std::fill(occ, occ + nz*ny*nx, false);

    // 3) Build 1D Gaussian kernel for warping
    int radius = int(region_smoothness * 3);
    auto kernel = gaussian_kernel(radius, region_smoothness);

    // 4) Generate per-slice random noise → warp2d[z][x]
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    std::vector<std::vector<double>> warp2d(nz, std::vector<double>(nx));
    for (int z = 0; z < nz; ++z)
      for (int x = 0; x < nx; ++x)
        warp2d[z][x] = dist(gen);

    // 5) Blur each row in X
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

    // 6) Assign warped horizontal layers, skipping already occupied voxels
    double layer_thick = double(ny) / macro_regions;
    for (int z = 0; z < nz; ++z) {
      for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
          int idx = (z * ny + y) * nx + x;
          if (occ[idx]) continue;  // do not overwrite anything already carved

          double warp = (warp2d[z][x] - 0.5) * (layer_thick * 0.4);
          for (int r = 0; r < macro_regions; ++r) {
            double lo = r * layer_thick + warp;
            double hi = (r + 1) * layer_thick + warp;
            if (y >= lo && y < hi) {
              ptr[idx] = uint8_t(r + 1);
              occ[idx] = true;       // mark this voxel occupied
              break;
            }
          }
        }
      }
    }
}

//──────────────────────────────────────────────────────────────────────────────
// draw_vessels: straight trunk + branching that never flips backwards,
// avoids carving into already‐occupied voxels, and returns its own length.
//──────────────────────────────────────────────────────────────────────────────
//------------------------------------------------------------------------------
// draw_vessels: carve a straight trunk + forward‐biased jitter & sparse branching,
//                avoid re‐carving occupied voxels, and report total carved length.
//------------------------------------------------------------------------------
double draw_vessels(
    uint8_t*    mask,
    bool*       occ,
    int         nz, int ny, int nx,
    int         root_z, int root_y, int root_x,
    int         max_depth,
    int         base_radius,
    const float* rng_vals,
    int         rng_len,
    double      init_dx,
    double      init_dy,
    double      init_dz,

    // tuning from JSON
    int         TRUNK_LEN,
    int         JITTER_INTERVAL,
    int         MAX_BRANCHES_BASE,
    double      RADIUS_DECAY
) {
    auto idx3 = [&](int z,int y,int x){ return (z*ny + y)*nx + x; };
    double total_length = 0.0;

    // 1) carve straight trunk
    double pz = root_z, py = root_y, px = root_x;
    double mag0 = std::sqrt(init_dx*init_dx + init_dy*init_dy + init_dz*init_dz) + 1e-6;
    double fdx = init_dx/mag0, fdy = init_dy/mag0, fdz = init_dz/mag0;

    for(int t = 0; t < TRUNK_LEN; ++t) {
        int zi = int(std::round(pz)),
            yi = int(std::round(py)),
            xi = int(std::round(px));
        if (zi<0||zi>=nz||yi<0||yi>=ny||xi<0||xi>=nx) break;

        int off = idx3(zi,yi,xi);
        if (mask[off] != 0) {
            // deflect slightly on collision
            double jf = (rng_vals[t % rng_len] - 0.5) * 0.3;
            fdx += jf; fdy += jf; fdz += jf;
            double m = std::sqrt(fdx*fdx + fdy*fdy + fdz*fdz) + 1e-6;
            fdx/=m; fdy/=m; fdz/=m;
        } else {
            carve_ball(mask, occ,
                       nz, ny, nx,
                       zi, yi, xi,
                       base_radius, /*label=*/7,
                       total_length);
        }
        pz += fdz; py += fdy; px += fdx;
    }

    // 2) start branching at trunk end
    int ez = int(std::round(pz)),
        ey = int(std::round(py)),
        ex = int(std::round(px));
    if (ez<0||ez>=nz||ey<0||ey>=ny||ex<0||ex>=nx)
        return total_length;

    struct Node { int z,y,x, depth, idx; double dx,dy,dz, radius; };
    std::vector<Node> stack;
    stack.reserve(64);
    stack.push_back({ez,ey,ex, 0, 0, fdx, fdy, fdz, double(base_radius)});

    int zi2=0, yi2=0, xi2=0;
    while(!stack.empty()) {
        Node n = stack.back(); stack.pop_back();
        if (n.depth >= max_depth || n.idx + 3 > rng_len) continue;

        // re-normalize branch direction
        double norm = std::sqrt(n.dx*n.dx + n.dy*n.dy + n.dz*n.dz) + 1e-6;
        double bdx = n.dx/norm, bdy = n.dy/norm, bdz = n.dz/norm;

        // compute forward‐limit
        double tmax = 1e9;
        double pos[3] = { double(n.z), double(n.y), double(n.x) };
        double dir[3] = { bdz, bdy, bdx };
        for (int i = 0; i < 3; ++i) {
            double d = dir[i], p = pos[i];
            int lim = (i==0? nz : i==1? ny : nx);
            if      (d > 0) tmax = std::min(tmax, ((lim-1)-p)/d);
            else if (d < 0) tmax = std::min(tmax, -p/d);
        }
        int length = std::min(int(tmax), 80);

        // walk this branch
        double p2[3] = { pos[0], pos[1], pos[2] };
        for (int i = 0; i < length; ++i) {
            if ((i % JITTER_INTERVAL) == 0) {
                bdx += (rng_vals[n.idx]   - 0.5) * 0.7;
                bdy += (rng_vals[n.idx+1] - 0.5) * 0.7;
                bdz += (rng_vals[n.idx+2] - 0.5) * 0.7;
                n.idx += 3;
                double m2 = std::sqrt(bdx*bdx + bdy*bdy + bdz*bdz) + 1e-6;
                bdx/=m2; bdy/=m2; bdz/=m2;
                // enforce forward bias
                double dot = bdx*fdx + bdy*fdy + bdz*fdz;
                if (dot < 0) {
                    bdx -= 2*dot*fdx;
                    bdy -= 2*dot*fdy;
                    bdz -= 2*dot*fdz;
                    double m3 = std::sqrt(bdx*bdx + bdy*bdy + bdz*bdz) + 1e-6;
                    bdx/=m3; bdy/=m3; bdz/=m3;
                }
            }

            p2[0] += bdz; p2[1] += bdy; p2[2] += bdx;
            zi2 = int(std::round(p2[0]));
            yi2 = int(std::round(p2[1]));
            xi2 = int(std::round(p2[2]));
            if (zi2<0||zi2>=nz||yi2<0||yi2>=ny||xi2<0||xi2>=nx) break;

            int off2 = idx3(zi2, yi2, xi2);
            if (mask[off2] == 0) {
                carve_ball(mask, occ,
                           nz, ny, nx,
                           zi2, yi2, xi2,
                           int(n.radius), /*label=*/7,
                           total_length);
            } else {
                // deflect on collision
                double jf = (rng_vals[n.idx % rng_len] - 0.5) * 0.3;
                bdx += jf; bdy += jf; bdz += jf;
                double m4 = std::sqrt(bdx*bdx + bdy*bdy + bdz*bdz) + 1e-6;
                bdx/=m4; bdy/=m4; bdz/=m4;
            }
        }

        // Fibonacci‐style sparse branching
        if (n.radius > 1.0) {
            int fib0 = 1, fib1 = 1;
            for (int i = 0; i < n.depth + 1; ++i) {
                int t = fib1; fib1 += fib0; fib0 = t;
            }
            int nb = std::min(MAX_BRANCHES_BASE + (fib1 % MAX_BRANCHES_BASE),
                               MAX_BRANCHES_BASE);
            double child_r = n.radius * RADIUS_DECAY;
            for (int b = 0; b < nb; ++b) {
                int base = n.idx + b*3;
                if (base + 3 > rng_len) break;
                double ndx = bdx + (rng_vals[base]   - 0.5) * 1.0;
                double ndy = bdy + (rng_vals[base+1] - 0.5) * 1.0;
                double ndz = bdz + (rng_vals[base+2] - 0.5) * 1.0;
                stack.push_back({ zi2, yi2, xi2,
                                  n.depth+1, base,
                                  ndx, ndy, ndz,
                                  child_r });
            }
        }
    }

    return total_length;
}


//──────────────────────────────────────────────────────────────────────────────
// add_vessels: now returns total_length across all faces
//──────────────────────────────────────────────────────────────────────────────
double add_vessels(
    py::array_t<uint8_t, py::array::c_style> labels,
    int num_vessels,
    int max_depth,
    int vessel_radius,
    int trunk_len,
    int jitter_interval,
    int max_branches,
    double radius_decay,
    int seed
) {
    // 1) grab label volume and dimensions
    auto bufL = labels.request();
    int nz = bufL.shape[0], ny = bufL.shape[1], nx = bufL.shape[2];
    uint8_t* ptrL = static_cast<uint8_t*>(bufL.ptr);

    // 2) allocate and clear occupancy mask
    std::vector<bool> occ(nz * ny * nx, false);

    // 3) RNG setup
    std::mt19937                        gen(seed);
    std::uniform_int_distribution<int>   Dz(0, nz-1),
                                         Dy(0, ny-1),
                                         Dx(0, nx-1);
    std::uniform_real_distribution<float> Dr(0.0f, 1.0f);

    const int rng_len = 50000;
    std::vector<float> rng_vals(rng_len);

    double grand_total = 0.0;
    int per_face = num_vessels / 6,
        rem      = num_vessels % 6;

    // 4) for each face, plant 'count' roots
    for (int face = 0; face < 6; ++face) {
        int count = per_face + (face < rem ? 1 : 0);
        for (int i = 0; i < count; ++i) {
            // refill random jitter buffer
            for (int j = 0; j < rng_len; ++j)
                rng_vals[j] = Dr(gen);

            // pick a random point on that face and an initial direction
            int rz, ry, rx;
            double init_dx = 0.0, init_dy = 0.0, init_dz = 0.0;
            switch (face) {
                case 0: rz = 0;      ry = Dy(gen); rx = Dx(gen); init_dz = +1; break;
                case 1: rz = nz - 1; ry = Dy(gen); rx = Dx(gen); init_dz = -1; break;
                case 2: rz = Dz(gen); ry = 0;      rx = Dx(gen); init_dy = +1; break;
                case 3: rz = Dz(gen); ry = ny - 1; rx = Dx(gen); init_dy = -1; break;
                case 4: rz = Dz(gen); ry = Dy(gen); rx = 0;      init_dx = +1; break;
                default: rz = Dz(gen); ry = Dy(gen); rx = nx - 1; init_dx = -1; break;
            }

            // call updated draw_vessels (which takes occ.data()) and sum up its length
            grand_total += draw_vessels(
                ptrL,                    // voxel labels
                occ.data(),              // occupancy mask
                nz, ny, nx,              // dimensions
                rz, ry, rx,              // root position
                max_depth,               // recursion depth
                vessel_radius,           // starting radius
                rng_vals.data(),         // jitter values
                rng_len,                 // their length
                init_dx, init_dy, init_dz,  // initial direction
                trunk_len,               // straight trunk length
                jitter_interval,         // jitter frequency
                max_branches,            // max children per branch
                radius_decay             // radius decay factor
            );
        }
    }

    return grand_total;
}


PYBIND11_MODULE(brain, m) {
    m.doc() = "High-performance C++ backend for brain geometry generation";

    // ───────────── add_macroregions (4 args) ─────────────
    m.def("add_macroregions", &add_macroregions,
          py::arg("labels"),       // uint8 volume
          py::arg("occupied"),     // bool mask, same shape
          py::arg("macro_regions"),
          py::arg("region_smoothness"),
          R"doc(
Replace space with warped horizontal macro-regions,
skipping any voxel already marked occupied.
)doc");

    // ───────────── add_neurons (7 args, returns total axon length) ─────────────
    m.def("add_neurons", &add_neurons,
          py::arg("labels"),            // uint8 volume
          py::arg("occupied"),          // bool mask, same shape
          py::arg("voxel_size"),        // std::array<double,3>
          py::arg("num_cells"),
          py::arg("cell_radius_range"), // std::array<double,2>
          py::arg("axon_dia_range"),    // std::array<double,2>
          py::arg("max_depth"),
          R"doc(
Place spheroidal cell bodies and nuclei, then grow axons along
a minimal spanning tree—avoiding overlaps—and return total axon length.
)doc");

    // ───────────── add_vessels (9 args, returns total vessel length) ─────────────
    m.def("add_vessels", &add_vessels,
          py::arg("labels"),         // uint8 volume
          py::arg("num_vessels"),
          py::arg("max_depth"),
          py::arg("vessel_radius"),
          py::arg("trunk_len"),
          py::arg("jitter_interval"),
          py::arg("max_branches"),
          py::arg("radius_decay"),
          py::arg("seed"),
          R"doc(
Generate vascular trees with collision checks (internal mask),
and return total vessel length (in voxels).
)doc");
}

