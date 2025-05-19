#include <cuda_runtime.h>
#include <cstdint>

// Each thread carves one sphere at (z0,y0,x0) with radius r
extern "C"
__global__ void carve_ball_kernel(
    uint8_t* __restrict__ mask,
    int nz, int ny, int nx,
    const int4* __restrict__ centers,  // (z,y,x,r)
    int n_centers,
    uint8_t label
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_centers) return;

    int4 c = centers[idx];
    int z0 = c.x, y0 = c.y, x0 = c.z, r = c.w;
    int rr = r*r;

    for (int dz = -r; dz <= r; dz++) {
        int zz = z0 + dz;
        if (zz < 0 || zz >= nz) continue;
        int dz2 = dz*dz;
        for (int dy = -r; dy <= r; dy++) {
            int yy = y0 + dy;
            if (yy < 0 || yy >= ny) continue;
            int dy2 = dy*dy;
            for (int dx = -r; dx <= r; dx++) {
                if (dz2 + dy2 + dx*dx <= rr) {
                    int xx = x0 + dx;
                    if (xx >= 0 && xx < nx) {
                        int off = (zz*ny + yy)*nx + xx;
                        mask[off] = label;
                    }
                }
            }
        }
    }
}
