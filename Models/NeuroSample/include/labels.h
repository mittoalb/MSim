#pragma once
#include <cstdint>

/// All the semantic label values used in the simulation
namespace Labels {
    constexpr uint8_t BACKGROUND        = 0;
    constexpr uint8_t OCCUPIED          = 1;    // occupancy mask
    constexpr uint8_t VESSEL_LUMEN      = 5;    // or whatever you chose
    constexpr uint8_t VESSEL_WALL       = 8;
    constexpr uint8_t NEURON_SOMA       = 6;
    constexpr uint8_t NEURON_NUCL        = 2;
    constexpr uint8_t AXON_WALL         = 8;    // reuse VESSEL_WALL? or distinct
    constexpr uint8_t AXON_SYNAPSE      = 10;
    constexpr uint8_t GLIA_CELL_BODY    = 9;
    constexpr uint8_t GLIA_PROCESS      = 7;
    constexpr uint8_t ENDO_CELL_BASE    = 12;   // endothelial bodies start here
    constexpr uint8_t ENDO_NUC_BASE     = 100;  // endothelial nuclei start here
}
