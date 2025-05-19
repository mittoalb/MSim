#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cuda_runtime.h>
#include <cstdint>
namespace py = pybind11;

// forward‐declare the CUDA kernel
extern "C"
__global__ void carve_ball_kernel(
    uint8_t*       mask,
    int            nz,
    int            ny,
    int            nx,
    const int4*    centers,
    int            n_centers,
    uint8_t        label
);

void carve_ball(
    py::array_t<uint8_t, py::array::c_style> mask,
    py::array_t<int,    py::array::c_style> centers,
    uint8_t label
) {
    auto m = mask.request();
    auto c = centers.request();
    uint8_t* ptr_mask = static_cast<uint8_t*>(m.ptr);
    int4*    ptr_c    = reinterpret_cast<int4*>(c.ptr);

    int nz = int(m.shape[0]);
    int ny = int(m.shape[1]);
    int nx = int(m.shape[2]);
    int n_centers = int(c.shape[0]);

    int threads = 256;
    int blocks  = (n_centers + threads - 1) / threads;
    carve_ball_kernel<<<blocks,threads>>>(
        ptr_mask, nz, ny, nx,
        ptr_c, n_centers,
        label
    );
    cudaDeviceSynchronize();
}

PYBIND11_MODULE(carlext, m) {
    m.def("carve_ball", &carve_ball,
          "carve_ball(mask(z,y,x), centers(N,4 int), label) → GPU‐filled spheres");
}
