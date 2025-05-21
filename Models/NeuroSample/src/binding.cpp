#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "vessels.h"
#include "cells.h"
#include "geometry.h"

namespace py = pybind11;

PYBIND11_MODULE(brain, m) {
    m.doc() = "High-performance C++ backend for brain geometry generation";
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
    m.def("connect_glia_to_neurons",
        [](
        py::array_t<uint8_t> labels,
        std::vector<std::array<int,3>> neuron_centers,
        int contact_label,
        int contact_radius
        ){
        auto lb = labels.request();
        uint8_t* ptr = static_cast<uint8_t*>(lb.ptr);
        int nz = lb.shape[0], ny = lb.shape[1], nx = lb.shape[2];
        connect_glia_to_neurons(ptr, nz, ny, nx,
                                neuron_centers,
                                contact_label,
                                contact_radius);
        },
        py::arg("labels"),
        py::arg("neuron_centers"),
        py::arg("contact_label")  = 12,
        py::arg("contact_radius") = 5
    );


    //──────────────────────────────────────────────────────────────────────────
    m.def("add_neurons",
        [](
            py::array_t<uint8_t, py::array::c_style|py::array::forcecast> labels,
            py::array_t<uint8_t, py::array::c_style|py::array::forcecast> occ,
            std::tuple<double,double,double> voxel_size,
            int num_cells,
            std::tuple<double,double> cell_radius_range,
            std::tuple<double,double> axon_dia_range,
            int max_depth
        ) {
            auto lb = labels.request(), ob = occ.request();
            if (lb.ndim != 3 || ob.ndim != 3)
                throw std::runtime_error("labels and occ must be 3-D");
            uint8_t* lbp = static_cast<uint8_t*>(lb.ptr);
            uint8_t* obp = static_cast<uint8_t*>(ob.ptr);
            int nz = lb.shape[0], ny = lb.shape[1], nx = lb.shape[2];

            std::array<double,3> vs = {
                std::get<0>(voxel_size),
                std::get<1>(voxel_size),
                std::get<2>(voxel_size)
            };
            std::array<double,2> cr = {
                std::get<0>(cell_radius_range),
                std::get<1>(cell_radius_range)
            };
            std::array<double,2> ar = {
                std::get<0>(axon_dia_range),
                std::get<1>(axon_dia_range)
            };

            // call C++ with the new centers arg
            std::vector<std::array<int,3>> centers;
            double total_length = add_neurons(
                lbp, obp,
                nz, ny, nx,
                vs,
                num_cells,
                cr,
                ar,
                max_depth,
                centers       // ← pass the vector
            );

            // build Python list of tuples
            py::list py_centers;
            for (auto &c : centers) {
                py_centers.append(py::make_tuple(c[0], c[1], c[2]));
            }

            return py::make_tuple(total_length, py_centers);
        },
        py::arg("labels"),
        py::arg("occ"),
        py::arg("voxel_size"),
        py::arg("num_cells"),
        py::arg("cell_radius_range"),
        py::arg("axon_dia_range"),
        py::arg("max_depth")
    );

    //──────────────────────────────────────────────────────────────────────────
        m.def("add_glial",
        [](
        py::array_t<uint8_t> labels,
        py::array_t<uint8_t> occ,
        int num_glia,
        int glia_radius_min,
        int glia_radius_max,
        int dend_depth,
        int dend_branches,
        py::array_t<float, py::array::c_style|py::array::forcecast> rng_vals
        ) {
        // Unpack labels/occ dims
        auto lb = labels.request(), ob = occ.request();
        uint8_t* lbp = static_cast<uint8_t*>(lb.ptr);
        uint8_t* obp = static_cast<uint8_t*>(ob.ptr);
        int nz = lb.shape[0], ny = lb.shape[1], nx = lb.shape[2];

        // Unpack RNG buffer
        auto rb = rng_vals.request();
        const float* rptr = static_cast<const float*>(rb.ptr);
        int rng_len = rb.size;

        int rng_index = 0;
        double total_length = 0.0;

        // Call C++ implementation
        double out_length = add_glial(
            lbp, obp,
            nz, ny, nx,
            num_glia,
            glia_radius_min,
            glia_radius_max,
            dend_depth,
            dend_branches,
            rptr, rng_len,
            rng_index,
            total_length
        );

        return out_length;
        },
        py::arg("labels"),
        py::arg("occ"),
        py::arg("num_glia"),
        py::arg("glia_radius_min"),
        py::arg("glia_radius_max"),
        py::arg("dend_depth"),
        py::arg("dend_branches"),
        py::arg("rng_vals")
    );


    //──────────────────────────────────────────────────────────────────────────────
        m.def("connect_somas_with_synapses",
        [](py::array_t<uint8_t, py::array::c_style | py::array::forcecast> labels,
        py::array_t<uint8_t, py::array::c_style | py::array::forcecast> occ,
        std::vector<std::array<int, 3>> centers,
        py::array_t<float,   py::array::c_style | py::array::forcecast> rng_vals,
        int rng_len,
        int axon_radius,           // tube radius
        int axon_steps,            // number of segments
        float jitter       = 0.3f, // σ: random‐perturb strength
        float persist      = 0.9f, // α: direction‐persistence
        int dend_branches  = 2,    // how many side‐branches to attempt
        float extra_connection_prob = 0.01f, // chance per branch
        uint8_t wall_label = 8,
        uint8_t synapse_label = 7
        ) {
            // Request buffers
            auto lb = labels.request(), ob = occ.request(), rb = rng_vals.request();
            if (lb.ndim != 3 || ob.ndim != 3 || rb.ndim != 1)
                throw std::runtime_error("labels/occ must be 3-D; rng_vals must be 1-D");

            int nz = lb.shape[0], ny = lb.shape[1], nx = lb.shape[2];
            double total_length = 0.0;
            int rng_index = 0;

            // Call the C++ implementation
            connect_somas_with_synapses(
                static_cast<uint8_t*>(lb.ptr),
                static_cast<uint8_t*>(ob.ptr),
                nz, ny, nx,
                centers,
                static_cast<const float*>(rb.ptr), rng_len, rng_index,
                total_length,
                axon_steps,           // 11: number of segments
                axon_radius,          // 12: tube radius
                jitter,               // 13: σ
                persist,              // 14: α
                dend_branches,        // 15: collateral branches
                extra_connection_prob,// 16: chance per branch
                synapse_label         // 17: label at endpoint
            );

            return total_length;
        },
        // Arg names and defaults for Python
        py::arg("labels"),
        py::arg("occ"),
        py::arg("centers"),
        py::arg("rng_vals"),
        py::arg("rng_len"),
        py::arg("axon_radius") = 2,
        py::arg("axon_steps")  = 5,
        py::arg("jitter")      = 0.3f,
        py::arg("persist")     = 0.9f,
        py::arg("dend_branches") = 2,
        py::arg("extra_connection_prob") = 0.01f,
        py::arg("wall_label")  = 8,
        py::arg("synapse_label") = 7
    );

    m.def("add_endothelial_cells_direct",
    [](py::array_t<uint8_t> labels,
       py::array_t<uint8_t> occ,
       uint8_t vessel_wall_label,
       uint8_t cell_label,
       uint8_t nucleus_label,
       double max_L = 5.0,
       double max_R = 2.0,
       int seed = 42
    ){
      auto lb = labels.request(), ob = occ.request();
      int nz = lb.shape[0], ny = lb.shape[1], nx = lb.shape[2];
      auto ptrL = (uint8_t*)lb.ptr;
      auto ptrO = (uint8_t*)ob.ptr;
      return add_endothelial_cells_direct(
        ptrL, ptrO, nz,ny,nx,
        vessel_wall_label, cell_label, nucleus_label,
        max_L, max_R, seed
      );
    },
    py::arg("labels"), py::arg("occ"),
    py::arg("vessel_wall_label"),
    py::arg("cell_label"), py::arg("nucleus_label"),
    py::arg("max_cell_length")=5.0,
    py::arg("max_cell_radius")=2.0,
    py::arg("seed")=42
    );

    m.def("add_schwann_cells",
        [](py::array_t<uint8_t> labels,
        py::array_t<uint8_t> occ,
        uint8_t axon_label,
        uint8_t schwann_label,
        bool myelinated,
        double radius,
        double thickness) {

            auto lb = labels.request();
            auto ob = occ.request();

            return add_schwann_cells(
                static_cast<uint8_t*>(lb.ptr),
                static_cast<uint8_t*>(ob.ptr),
                lb.shape[0], lb.shape[1], lb.shape[2],
                axon_label, schwann_label,
                myelinated,
                radius, thickness
            );
        },
        py::arg("labels"),
        py::arg("occ"),
        py::arg("axon_label"),
        py::arg("schwann_label"),
        py::arg("myelinated") = true,
        py::arg("radius") = 4.0,
        py::arg("thickness") = 1.5
    );

}

