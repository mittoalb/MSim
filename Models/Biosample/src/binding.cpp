#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "vessels.h"
#include "cells.h"
#include "cball.h"

namespace py = pybind11;

PYBIND11_MODULE(brain, m) {
    m.doc() = "High-performance C++ backend for brain geometry generation";

    //──────────────────────────────────────────────────────────────────────────
    // draw_vessels: carve one vascular tree from a root
    m.def("draw_vessels",
        [](py::array_t<uint8_t, py::array::c_style | py::array::forcecast> mask,
           py::array_t<uint8_t, py::array::c_style | py::array::forcecast> occ,
           int nz, int ny, int nx,
           int root_z, int root_y, int root_x,
           int max_depth,
           int base_radius,
           py::array_t<float, py::array::c_style | py::array::forcecast> rng_vals,
           int rng_len,
           double init_dx, double init_dy, double init_dz,
           int TRUNK_LEN, int JITTER_INTERVAL,
           int MAX_BRANCHES_BASE, int MAX_BRANCH_LEN,
           double RADIUS_DECAY)
        {
            auto mb = mask.request();
            auto ob = occ.request();
            auto rb = rng_vals.request();
            if (mb.ndim != 3 || ob.ndim != 3 || rb.ndim != 1)
                throw std::runtime_error("mask, occ must be 3-D and rng_vals 1-D");

            return draw_vessels(
                static_cast<uint8_t*>(mb.ptr),
                static_cast<uint8_t*>(ob.ptr),
                nz, ny, nx,
                root_z, root_y, root_x,
                max_depth,
                base_radius,
                static_cast<const float*>(rb.ptr), rng_len,
                init_dx, init_dy, init_dz,
                TRUNK_LEN, JITTER_INTERVAL,
                MAX_BRANCHES_BASE, MAX_BRANCH_LEN,
                RADIUS_DECAY
            );
        },
        py::arg("mask"),           py::arg("occ"),
        py::arg("nz"),             py::arg("ny"),             py::arg("nx"),
        py::arg("root_z"),         py::arg("root_y"),         py::arg("root_x"),
        py::arg("max_depth"),      py::arg("base_radius"),
        py::arg("rng_vals"),       py::arg("rng_len"),
        py::arg("init_dx"),        py::arg("init_dy"),        py::arg("init_dz"),
        py::arg("TRUNK_LEN"),      py::arg("JITTER_INTERVAL"),
        py::arg("MAX_BRANCHES_BASE"), py::arg("MAX_BRANCH_LEN"),
        py::arg("RADIUS_DECAY")
    );

    //──────────────────────────────────────────────────────────────────────────
    // add_vessels: plant many roots into labels[] with per-vessel radius jitter
    m.def("add_vessels",
        [](py::array_t<uint8_t, py::array::c_style | py::array::forcecast> labels,
           int num_vessels,
           int max_depth,
           int vessel_radius_avg,
           double vessel_radius_jitter,
           int trunk_len,
           int jitter_interval,
           int max_branches,
           int branch_len,
           double radius_decay,
           int seed)
        {
            auto lb = labels.request();
            if (lb.ndim != 3)
                throw std::runtime_error("labels must be a 3-D array");
            int nz = lb.shape[0], ny = lb.shape[1], nx = lb.shape[2];

            return add_vessels(
                static_cast<uint8_t*>(lb.ptr),
                nz, ny, nx,
                num_vessels,
                max_depth,
                vessel_radius_avg,
                vessel_radius_jitter,
                trunk_len,
                jitter_interval,
                max_branches,
                branch_len,
                radius_decay,
                seed
            );
        },
        py::arg("labels"),               py::arg("num_vessels"),
        py::arg("max_depth"),            py::arg("vessel_radius_avg"),
        py::arg("vessel_radius_jitter"), py::arg("trunk_len"),
        py::arg("jitter_interval"),      py::arg("max_branches"),
        py::arg("branch_len"),           py::arg("radius_decay"),
        py::arg("seed")
    );

    //──────────────────────────────────────────────────────────────────────────
    // add_macroregions: carve warped macro regions via Gaussian warp
    m.def("add_macroregions",
        [](py::array_t<uint8_t, py::array::c_style | py::array::forcecast> labels,
           py::array_t<uint8_t, py::array::c_style | py::array::forcecast> occ,
           int macro_regions,
           double region_smoothness)
        {
            auto lb = labels.request(), ob = occ.request();
            if (lb.ndim != 3 || ob.ndim != 3)
                throw std::runtime_error("labels and occ must be 3-D arrays");
            int nz = lb.shape[0], ny = lb.shape[1], nx = lb.shape[2];
            add_macroregions(
                static_cast<uint8_t*>(lb.ptr),
                static_cast<uint8_t*>(ob.ptr),
                nz, ny, nx,
                macro_regions,
                region_smoothness
            );
        },
        py::arg("labels"), py::arg("occ"),
        py::arg("macro_regions"), py::arg("region_smoothness")
    );

    //──────────────────────────────────────────────────────────────────────────
    // add_neurons: place cells & connect with axons
    m.def("add_neurons",
        [](py::array_t<uint8_t, py::array::c_style | py::array::forcecast> labels,
           py::array_t<uint8_t, py::array::c_style | py::array::forcecast> occ,
           std::array<double,3> voxel_size,
           int num_cells,
           std::array<double,2> cell_radius_range,
           std::array<double,2> axon_dia_range,
           int max_depth)
        {
            auto lb = labels.request(), ob = occ.request();
            if (lb.ndim != 3 || ob.ndim != 3)
                throw std::runtime_error("labels and occ must be 3-D arrays");
            int nz = lb.shape[0], ny = lb.shape[1], nx = lb.shape[2];
            return add_neurons(
                static_cast<uint8_t*>(lb.ptr),
                static_cast<uint8_t*>(ob.ptr),
                nz, ny, nx,
                voxel_size,
                num_cells,
                cell_radius_range,
                axon_dia_range,
                max_depth
            );
        },
        py::arg("labels"),   py::arg("occ"),
        py::arg("voxel_size"),
        py::arg("num_cells"),
        py::arg("cell_radius_range"),
        py::arg("axon_dia_range"),
        py::arg("max_depth")
    );
}

