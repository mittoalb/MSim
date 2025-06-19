# MSim - Modular X-ray Simulation Toolkit

A GPU-accelerated X-ray simulation package for tomography and laminography with realistic photon statistics and dose calculation.

## Features

- **GPU-accelerated wave propagation** using CuPy for high-performance simulation
- **Multi-physics modeling** including phase contrast, absorption, and coherent scattering
- **Tomography and laminography** simulation with arbitrary tilt angles
- **Realistic detector response** with photon counting statistics and noise
- **Radiation dose calculation** with beam attenuation modeling
- **Multi-material phantoms** with accurate X-ray properties from XRayLib
- **Flexible data formats** supporting N5/Zarr multiscale volumes

## Installation

### Requirements

- Python ≥ 3.9
- CUDA-compatible GPU
- CuPy (CUDA toolkit)

### Dependencies

```bash
pip install cupy-cuda12x numpy scipy h5py z5py xraylib
```

### Install from source

```bash
git clone <repository-url>
cd msim
pip install -e .
```

## Quick Start

### 1. Generate a test phantom

```bash
python generate_phantom.py bone
```

### 2. Create simulation configuration

```json
{
    "ENERGY_KEV": 23.0,
    "DETECTOR_DIST": 0.3,
    "DETECTOR_PIXEL_SIZE": 0.5e-6,
    "PAD": 50,
    "ENABLE_PHASE": true,
    "ENABLE_ABSORPTION": true,
    "ENABLE_SCATTER": true,
    "INCIDENT_PHOTONS": 1e6,
    "DETECTOR_EFFICIENCY": 0.8,
    "ENABLE_PHOTON_NOISE": true
}
```

### 3. Run simulation

```python
from msim.interface import quick_tomography, quick_laminography

# Tomography scan
projections, dose_stats = quick_tomography(
    "phantom_bone.zarr", 
    "phantom_bone.json", 
    n_projections=180,
    calculate_dose=True
)

# Laminography scan  
projections, dose_stats = quick_laminography(
    "phantom_bone.zarr",
    "phantom_bone.json",
    tilt_deg=45,
    n_projections=360,
    calculate_dose=True
)
```

## Advanced Usage

### Custom scan parameters

```python
from msim.interface import XRayScanner
import numpy as np

scanner = XRayScanner("config.json")
scanner.load_volume("phantom.zarr", "phantom.json")

# Custom angle sequence
angles = np.linspace(0, 180, 90)
projections, dose_stats = scanner.tomography_scan(
    angles, 
    "custom_scan.h5",
    calculate_dose=True
)
```

### Dose analysis only

```python
from msim.interface import analyze_dose_only

dose_map, dose_stats = analyze_dose_only(
    "phantom_bone.zarr",
    "phantom_bone.json"
)

# Print dose statistics
for label, stats in dose_stats.items():
    print(f"{stats['material_name']}: {stats['mean_dose_gy']:.2e} Gy")
```

### Parameter studies

```python
# Test different photon counts
for photon_count in [1e4, 1e5, 1e6, 1e7]:
    scanner.config["INCIDENT_PHOTONS"] = photon_count
    proj, dose = scanner.tomography_scan(angles, f"scan_{photon_count:.0e}.h5")
```

## Configuration Parameters

### Geometry
- `ENERGY_KEV`: X-ray energy in keV
- `DETECTOR_DIST`: Sample-to-detector distance (m)
- `DETECTOR_PIXEL_SIZE`: Detector pixel size (m)
- `PAD`: Padding for wave propagation

### Physics
- `ENABLE_PHASE`: Enable phase contrast
- `ENABLE_ABSORPTION`: Enable absorption contrast  
- `ENABLE_SCATTER`: Enable coherent scattering
- `ADD_RANDOM_PHASE`: Add random phase for numerical stability

### Photon Statistics
- `INCIDENT_PHOTONS`: Photons per detector pixel
- `DETECTOR_EFFICIENCY`: Quantum efficiency (0-1)
- `DARK_CURRENT`: Dark counts per pixel
- `READOUT_NOISE`: Electronic noise (RMS)
- `ENABLE_PHOTON_NOISE`: Include shot noise

## Phantom Generation

### Available phantom types

```bash
python generate_phantom.py sphere         # Simple nested spheres
python generate_phantom.py cylinder       # Cylindrical features
python generate_phantom.py bone           # Complex bone with implants
python generate_phantom.py microstructure # Fine structures and fibers
python generate_phantom.py dose_test      # Optimized for dose testing
```

### Custom phantoms

```python
from generate_phantom import generate_phantom

# Create custom phantom
zarr_path, json_path = generate_phantom(
    "bone", 
    shape=(100, 128, 128), 
    voxel_size=(0.5, 0.5, 0.5)
)
```

## Material Database

MSim includes accurate material properties for dose calculation:

- **Biological**: soft tissue, fat, muscle, cortical/trabecular bone, marrow
- **Medical**: iodine contrast, titanium implants
- **Industrial**: aluminum, iron, lead, carbon fiber, gold nanoparticles

Materials are defined with:
- Chemical composition (elements and stoichiometry)
- Mass density (g/cm³)
- Automatic X-ray properties via XRayLib

## Output Formats

### HDF5 Structure
```
scan.h5
├── exchange/
│   └── data          # (n_projections, height, width) float32
├── dose/             # Optional dose data
│   └── dose_map      # (nz, ny, nx) dose in Gray
├── angles            # Rotation angles (degrees)
└── metadata          # Energy, geometry, photon parameters
```

### Zarr/N5 Volumes
- Multiscale pyramids for efficient visualization
- Neuroglancer-compatible metadata
- Material lookup tables included

## Physics Implementation

### Wave Propagation
- **Fresnel diffraction** using angular spectrum method
- **Multi-slice propagation** through 3D volumes
- **Complex wavefront** tracking (amplitude + phase)

### X-ray Interactions
- **Phase contrast**: Refractive index from XRayLib
- **Absorption**: Mass attenuation coefficients
- **Coherent scattering**: Klein-Nishina + Thomson cross sections

### Detector Response
- **Photon counting statistics** (Poisson noise)
- **Quantum efficiency** and electronic noise
- **Realistic detector parameters**

### Dose Calculation
- **Beam attenuation** through materials
- **Energy absorption coefficients** from XRayLib
- **Material-specific dose maps** in Gray

## Performance

### GPU Acceleration
- **CuPy arrays** for all computations
- **Memory-efficient** volume handling
- **Batch processing** for multiple projections

### Typical Performance
- **100×128×128 volume**: ~1-2 seconds per projection (RTX 4090)
- **Memory usage**: ~2× volume size in GPU RAM
- **Scaling**: Linear with number of projections

## Validation

### Physics Accuracy
- **Quantitative agreement** with Beer's law (0.2% error)
- **Phase contrast validation** against analytical solutions
- **Dose calculations** verified with reference materials

### Test Suite
```bash
python test_simulation_physics.py  # Comprehensive physics tests
python test_rotation.py           # Geometry validation  
```

## Examples

See `examples/` directory for:
- Basic tomography and laminography
- Parameter studies (photon count, energy, geometry)
- Material-specific dose analysis
- Custom phantom generation
- Reconstruction workflows

## API Reference

### Core Classes
- `XRayScanner`: Main simulation interface
- `GPUVolumeManager`: Memory-efficient volume handling

### Key Functions
- `projection()`: Core wave propagation simulation
- `quick_tomography()`, `quick_laminography()`: One-line scans
- `calculate_dose_map()`: Radiation dose computation
- `generate_phantom()`: Test phantom creation

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Run tests (`python -m pytest tests/`)
4. Commit changes (`git commit -m 'Add amazing feature'`)
5. Push to branch (`git push origin feature/amazing-feature`)
6. Open Pull Request

