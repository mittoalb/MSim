#!/usr/bin/env python3
"""
Simple test without streaming - just verify projection works
"""

import numpy as np
from msim.geometry import simulate_projection_series

print("="*70)
print("SIMPLE PROJECTION TEST (No Streaming)")
print("="*70)

# Create test volume
print("Creating test volume...")
test_volume = np.random.randint(0, 5, (30, 30, 30), dtype=np.int32)

# Create material lookup table
test_lookup = {
    0: {'composition': {}, 'density': 0.0},  # Air
    1: {'composition': {'H': 2, 'O': 1}, 'density': 1.0},  # Water
    2: {'composition': {'Ca': 1, 'C': 1, 'O': 3}, 'density': 2.71},  # CaCO3
    3: {'composition': {'Al': 1}, 'density': 2.7},  # Aluminum
    4: {'composition': {'Cu': 1}, 'density': 8.96},  # Copper
}

# Voxel size
voxel_size = (0.5, 0.5, 0.5)

# Config as DICTIONARY
config = {
    "ENERGY_KEV": 23.0,
    "DETECTOR_DIST": 0.3,
    "DETECTOR_PIXEL_SIZE": 0.5e-6,
    "PAD": 50,
    "ENABLE_PHASE": True,
    "ENABLE_ABSORPTION": True,
    "ENABLE_SCATTER": True,
    "ADD_RANDOM_PHASE": False,
    "INCIDENT_PHOTONS": 1e6,
    "DETECTOR_EFFICIENCY": 0.8,
    "DARK_CURRENT": 10,
    "READOUT_NOISE": 5,
    "ENABLE_PHOTON_NOISE": True,
}

# Just 3 angles for quick test
angles = np.arange(0,180,1)#[0, 45, 90]

print(f"Simulating {len(angles)} projections...")

try:
    projections = simulate_projection_series(
        test_volume,
        test_lookup,
        voxel_size,
        angles,
        tilt_deg=0,
        config=config,
        stream_pv='TEST:MINIMAL'  # No streaming
    )
    
    print("\n" + "="*70)
    print("✓ TEST PASSED")
    print("="*70)
    print(f"Generated {len(projections)} projections")
    print(f"Projection shape: {projections[0].shape}")
    print(f"Data range: [{projections.min():.1f}, {projections.max():.1f}]")
    
except Exception as e:
    print("\n" + "="*70)
    print("✗ TEST FAILED")
    print("="*70)
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()