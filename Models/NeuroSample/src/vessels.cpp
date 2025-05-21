#include "vessels.h"
#include "geometry.h"
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>

//──────────────────────────────────────────────────────────────────────────────
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
    int         MAX_BRANCH_LEN
) {
    constexpr uint8_t WALL_LABEL = 7;
    constexpr uint8_t LUMEN_LABEL = 8;

    double total_length = 0.0;

    double mag0 = std::sqrt(init_dx*init_dx + init_dy*init_dy + init_dz*init_dz) + 1e-6;
    double fdx = init_dx / mag0, fdy = init_dy / mag0, fdz = init_dz / mag0;

    double pz = root_z, py = root_y, px = root_x;
    for (int t = 0; t < TRUNK_LEN; ++t) {
        if (t % JITTER_INTERVAL == 0) {
            double jf = (rng_vals[t % rng_len] - 0.5) * 0.3;
            fdx += jf; fdy += jf; fdz += jf;
            double m = std::sqrt(fdx*fdx + fdy*fdy + fdz*fdz) + 1e-6;
            fdx /= m; fdy /= m; fdz /= m;
        }
        double z0 = pz, y0 = py, x0 = px;
        pz += fdz; py += fdy; px += fdx;
        carve_cylinder_segment(mask, occ, nz, ny, nx,
            z0, y0, x0,
            pz, py, px,
            base_radius,
            WALL_LABEL, LUMEN_LABEL,
            total_length
        );
    }

    int ez = int(std::round(pz)), ey = int(std::round(py)), ex = int(std::round(px));
    if (ez < 0 || ez >= nz || ey < 0 || ey >= ny || ex < 0 || ex >= nx)
        return total_length;

    struct Node {
        int    z,y,x, depth, idx;
        double dx,dy,dz, radius;
    };
    std::vector<Node> stack;
    stack.reserve(1024);
    stack.push_back({ez,ey,ex, 0, 0, fdx,fdy,fdz, double(base_radius)});

    while (!stack.empty()) {
        Node n = stack.back(); stack.pop_back();
        if (n.depth >= max_depth || n.idx + 3 > rng_len) continue;

        double norm = std::sqrt(n.dx*n.dx + n.dy*n.dy + n.dz*n.dz) + 1e-6;
        double bdx = n.dx / norm, bdy = n.dy / norm, bdz = n.dz / norm;

        double pos[3] = {double(n.z), double(n.y), double(n.x)};
        double dir[3] = {bdz, bdy, bdx};
        double tmax = 1e9;
        for (int i = 0; i < 3; ++i) {
            double d = dir[i], p = pos[i];
            int L = (i==0? nz : (i==1? ny : nx));
            if      (d > 0) tmax = std::min(tmax, ((L-1)-p)/d);
            else if (d < 0) tmax = std::min(tmax, -p/d);
        }
        int length = std::min(int(tmax), MAX_BRANCH_LEN);

        double p2[3] = {pos[0],pos[1],pos[2]};
        int rng_i = n.idx;
        for (int i = 0; i < length; ++i) {
            if (i % JITTER_INTERVAL == 0) {
                bdx += (rng_vals[rng_i]     - 0.5)*0.7;
                bdy += (rng_vals[rng_i + 1] - 0.5)*0.7;
                bdz += (rng_vals[rng_i + 2] - 0.5)*0.7;
                rng_i += 3;
                double m2 = std::sqrt(bdx*bdx + bdy*bdy + bdz*bdz) + 1e-6;
                bdx /= m2; bdy /= m2; bdz /= m2;
            }
            double z0 = p2[0], y0 = p2[1], x0 = p2[2];
            double z1 = z0 + bdz, y1 = y0 + bdy, x1 = x0 + bdx;
            carve_cylinder_segment(mask, occ, nz, ny, nx,
                z0,y0,x0, z1,y1,x1,
                int(n.radius), WALL_LABEL, LUMEN_LABEL, total_length);
            p2[0]=z1; p2[1]=y1; p2[2]=x1;
        }

        int zi2 = int(std::round(p2[0])), yi2 = int(std::round(p2[1])), xi2 = int(std::round(p2[2]));
        if (zi2<1||zi2>=nz-1 || yi2<1||yi2>=ny-1 || xi2<1||xi2>=nx-1) continue;

        if (n.radius > 0.1) {
            double child_r = n.radius / std::sqrt(2.0);
            for (int b = 0; b < 2; ++b) {
                int base = n.idx + b*3;
                if (base + 3 > rng_len) break;
                double ndx = bdx + (rng_vals[base]     - 0.5);
                double ndy = bdy + (rng_vals[base + 1] - 0.5);
                double ndz = bdz + (rng_vals[base + 2] - 0.5);
                stack.push_back({zi2, yi2, xi2, n.depth + 1, base, ndx, ndy, ndz, child_r});
            }
        }
    }

    return total_length;
}


//──────────────────────────────────────────────────────────────────────────────
double add_vessels(
    uint8_t* labels,
    int      nz, int ny, int nx,
    int      num_vessels,
    int      max_depth,
    int      vessel_radius_avg,
    double   vessel_radius_jitter,
    int      trunk_len,
    int      jitter_interval,
    int      max_branches,    // ← kept here for config compatibility (but not used)
    int      branch_len,
    double   radius_decay,    // ← kept here for config compatibility (but not used)
    int      seed
) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<> Dz(0,nz-1), Dy(0,ny-1), Dx(0,nx-1);
    std::uniform_real_distribution<float> Dr(0.0f,1.0f);
    std::uniform_real_distribution<> Rdist(
        vessel_radius_avg*(1.0 - vessel_radius_jitter),
        vessel_radius_avg*(1.0 + vessel_radius_jitter)
    );

    const int rng_len = 50000;
    std::vector<float>      rng_vals(rng_len);
    std::vector<uint8_t>    occ(nz*ny*nx, 0);

    double grand_total = 0.0;
    int per_face = num_vessels / 6, rem = num_vessels % 6;

    for (int face = 0; face < 6; ++face) {
        int count = per_face + (face < rem);
        for (int i = 0; i < count; ++i) {
            // refill RNG
            for (int j = 0; j < rng_len; ++j)
                rng_vals[j] = Dr(gen);

            // sample per-vessel start radius
            int base_radius = int(std::round(Rdist(gen)));

            // pick a random entry point on one of the 6 faces
            int rz, ry, rx;
            double dx = 0, dy = 0, dz = 0;
            switch (face) {
                case 0: rz = 0;      ry = Dy(gen); rx = Dx(gen); dz = +1; break;
                case 1: rz = nz-1;   ry = Dy(gen); rx = Dx(gen); dz = -1; break;
                case 2: rz = Dz(gen); ry = 0;      rx = Dx(gen); dy = +1; break;
                case 3: rz = Dz(gen); ry = ny-1;   rx = Dx(gen); dy = -1; break;
                case 4: rz = Dz(gen); ry = Dy(gen); rx = 0;      dx = +1; break;
                default:rz = Dz(gen); ry = Dy(gen); rx = nx-1;   dx = -1; break;
            }

            // final call — now matches draw_vessels correctly (18 args)
            grand_total += draw_vessels(
                labels, occ.data(),
                nz, ny, nx,
                rz, ry, rx,
                max_depth,
                base_radius,
                rng_vals.data(), rng_len,
                dx, dy, dz,
                trunk_len,
                jitter_interval,
                branch_len
            );
        }
    }

    return grand_total;
}
