# MSim - Modular X-ray Simulation Toolkit

A GPU-accelerated X-ray simulation package for tomography and laminography with realistic photon statistics, dose calculation, and real-time streaming.

## Features

- **GPU-accelerated wave propagation** using CuPy for high-performance simulation
- **Multi-physics modeling** including phase contrast, absorption, and coherent scattering
- **Tomography and laminography** simulation with arbitrary tilt angles
- **Real-time PV streaming** to EPICS viewers (ImageJ, CS-Studio, etc.)
- **Realistic detector response** with photon counting statistics and noise
- **Radiation dose calculation** with beam attenuation modeling
- **Multi-material phantoms** with accurate X-ray properties from XRayLib
- **Flexible data formats** supporting N5/Zarr multiscale volumes

## Installation

Create a conda environment

```bash
conda create -n MSIM python=3.13 cupy z5py pvaccess
```

### Requirements

- Python ≥ 3.9
- CUDA-compatible GPU
- CuPy (CUDA toolkit)
- PVAccess (for streaming, optional)

### Dependencies

```bash
pip install numpy scipy z5py xraylib pvaccess
```

### Install from source

```bash
git clone <repository-url>
cd msim
pip install -e .
```

### Compiling the CUDA code

This will create the required CUDA libraries.

```bash
cd MSim/utils
./compile.sh
```

You may need to export the path in your environment:

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/your/msim/path/msim/cuda
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
from msim.simulator import quick_tomography, quick_laminography

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

## Real-Time PV Streaming

MSim can stream projections to EPICS PVAccess viewers as they are computed, enabling real-time monitoring and feedback.

### Viewer Setup

**ImageJ:**
1. Install EPICS AD Viewer plugin
2. Plugins → EPICS → EPICS AD Viewer
3. Connect to PV name (e.g., `SIM:TOMO`)

**CS-Studio:**
1. Add Image widget
2. Set PV name to your stream
3. Configure color mapping

**Command line:**
```bash
# Monitor PV updates
pvget -m SIM:TOMO

# List active PVs
pvlist
```

### Basic Streaming Example

```python
from msim.geometry import simulate_projection_series
import numpy as np

# Your phantom data
volume = ...  # Load your volume
lookup = ...  # Material lookup
config = {...}  # Simulation config

# Stream to PV as projections are computed
angles = np.linspace(0, 180, 90)
projections = simulate_projection_series(
    volume, lookup, voxel_size,
    angles_deg=angles,
    tilt_deg=0,
    config=config,
    stream_pv="SIM:TOMO"  # Stream to this PV
)

# Keep server alive for viewing
from msim.geometry import keep_streaming_alive
keep_streaming_alive()  # Press Ctrl+C to stop
```

### Streaming with XRayScanner

```python
from msim.simulator import XRayScanner

scanner = XRayScanner("config.json")
scanner.load_volume("phantom.zarr", "phantom.json")

# Note: Streaming requires using simulate_projection_series directly
from msim.geometry import simulate_projection_series

projections = simulate_projection_series(
    scanner.volume, scanner.lookup, scanner.voxel_size,
    angles_deg=np.linspace(0, 180, 90),
    tilt_deg=0,
    config=scanner.config,
    stream_pv="SIM:TOMO"
)
```

### Advanced Streaming

```python
# Multiple PVs for different geometries
tomo_proj = simulate_projection_series(
    volume, lookup, voxel_size,
    angles, tilt_deg=0, config=config,
    stream_pv="SIM:TOMO"
)

lamino_proj = simulate_projection_series(
    volume, lookup, voxel_size,
    angles, tilt_deg=45, config=config,
    stream_pv="SIM:LAMINO"
)

# List active streams
from msim.geometry import list_active_pvs
list_active_pvs()

# Cleanup when done
from msim.geometry import cleanup_streaming
cleanup_streaming()
```

### Streaming Tips

- **PV names**: Use descriptive names like `SIM:TOMO`, `SIM:LAMINO`, `TEST:BRAIN`
- **Viewer connection**: Start your viewer and connect before running simulation
- **Network**: Ensure firewall allows PVAccess traffic (default ports: 5075-5076)
- **Performance**: Streaming adds minimal overhead (~1-2ms per frame)
- **Multiple viewers**: Multiple viewers can connect to the same PV simultaneously

## Advanced Usage

### Custom scan parameters

```python
from msim.simulator import XRayScanner
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
from msim.simulator import analyze_dose_only

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

### Fast Mouse Brain Phantom

Generate anatomically realistic mouse brain phantoms:

```bash
# Standard resolution (200×160×120 @ 50µm) - Fast!
python generate_fast_brain.py

# High resolution (400×320×240 @ 25µm)
python generate_fast_brain.py --high-res

# With cortical layers and detailed vasculature
python generate_fast_brain.py --enable-layers --enable-vessels
```

Features:
- 24 anatomical brain regions
- Realistic tissue compositions (ICRP-based)
- Hierarchical vascular network
- Cortical layering (optional)
- Generation time: 10-30 seconds (standard) or 2-3 minutes (high-res)

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
- **Brain tissues**: gray matter, white matter, hippocampus, cerebellum, etc.
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

### Wave Propagation Theory

MSim implements coherent X-ray wave propagation using the **Fresnel diffraction** formalism with the **angular spectrum method**.

#### Fresnel Propagation

The wavefront propagation between planes separated by distance `z` is given by:

```
ψ(x,y,z) = F⁻¹[F[ψ(x,y,0)] · H(kₓ,kᵧ,z)]
```

where the **Fresnel propagator** is:

```
H(kₓ,kᵧ,z) = exp(-i(kₓ² + kᵧ²)z / 2k₀)
```

with:
- `k₀ = 2π/λ` = wavenumber in vacuum
- `λ = hc/E` = X-ray wavelength
- `kₓ, kᵧ` = spatial frequencies
- `F, F⁻¹` = Fourier transform operators

#### Multi-slice Method

The 3D volume is propagated slice-by-slice with thickness `Δz`:

```
ψₙ₊₁ = F⁻¹[F[ψₙ · T(x,y)] · H(kₓ,kᵧ,Δz)]
```

where `T(x,y)` is the **transmission function** for slice `n`.

### X-ray Matter Interaction

#### Complex Refractive Index

The X-ray refractive index for materials is:

```
n = 1 - δ - iβ
```

where:
- `δ` = real part (phase shift)
- `β` = imaginary part (absorption)

From XRayLib, these are calculated as:
```
δ = 1 - Re[n(E,ρ)]
β = Im[n(E,ρ)]
```

#### Transmission Function

The transmission through a material slice of thickness `Δz` is:

```
T(x,y) = exp(ik₀δ(x,y)Δz) · exp(-μₐᵦₛ(x,y)Δz/2)
```

where:
- **Phase term**: `exp(ik₀δΔz)` causes phase advance
- **Absorption term**: `exp(-μₐᵦₛΔz/2)` attenuates amplitude

#### Linear Attenuation Coefficients

From XRayLib cross-sections:

```
μₐᵦₛ = ρ(σₜₒₜₐₗ - σᵣₐᵧₗₑᵢgₕ) × 100
μₛcₐₜ = ρ × σᵣₐᵧₗₑᵢgₕ × 100
```

where:
- `ρ` = material density (g/cm³)
- `σₜₒₜₐₗ` = total cross-section (cm²/g)
- `σᵣₐᵧₗₑᵢgₕ` = Rayleigh scattering cross-section (cm²/g)
- Factor of 100 converts to cm⁻¹

### Coherent Scattering Model

#### Klein-Nishina + Thomson Scattering

The differential scattering cross-section combines:

**Thomson scattering:**
```
dσᵧ/dΩ = rₑ²(1 + cos²θ)/2
```

**Klein-Nishina correction:**
```
dσₖₙ/dΩ = (rₑ²/2) × (1 + cos²θ)/[1 + α(1-cosθ)]² × [1 + α(1-cosθ) - α²(1-cosθ)²/((1+cos²θ)(1+α(1-cosθ)))]
```

where:
- `rₑ = 2.818 × 10⁻¹⁵ m` = classical electron radius
- `α = E/(mₑc²) = E/511` keV = photon energy ratio
- `θ` = scattering angle

#### Point Spread Function

The 2D scattering PSF is:

```
PSF(r) = ∫ (dσ/dΩ)(θ) × 2πsinθ dθ
```

projected onto the detector plane:
```
r = θ × D / pᵢₓₑₗ
```

where `D` = detector distance, `pᵢₓₑₗ` = pixel size.

#### Scattering Implementation

For each slice, the intensity is split:
```
Iᵤₙₛcₐₜₜₑᵣₑ𝒹 = I × exp(-μₛcₐₜ × Δz)
Iₛcₐₜₜₑᵣₑ𝒹 = I × [1 - exp(-μₛcₐₜ × Δz)]
```

The scattered intensity is convolved with the PSF:
```
Iᵦₗᵤᵣᵣₑ𝒹 = Iₛcₐₜₜₑᵣₑ𝒹 ⊗ PSF
```

Total intensity: `Iₜₒₜₐₗ = Iᵤₙₛcₐₜₜₑᵣₑ𝒹 + Iᵦₗᵤᵣᵣₑ𝒹`

Wavefront reconstruction:
```
ψₙₑ𝓌 = √Iₜₒₜₐₗ × exp(i∠ψₒₗ𝒹)
```

### Detector Response Model

#### Photon Statistics

**Incident photon flux:**
```
Φᵢₙc = Iₙₒᵣₘₐₗᵢᵤₑ𝒹 × N₀
```

where `N₀` = incident photons per pixel.

**Detected photons:**
```
Nᵈᵉᵗ = Poisson(Φᵢₙc × ηᵈᵉᵗ)
```

where `ηᵈᵉᵗ` = detector quantum efficiency.

**Noise sources:**
```
Nₜₒₜₐₗ = Nᵈᵉᵗ + Poisson(Nᵈₐᵣₖ) + Normal(0, σᵣₑₐ𝒹ₒᵤₜ)
```

where:
- `Nᵈₐᵣₖ` = dark current counts
- `σᵣₑₐ𝒹ₒᵤₜ` = readout noise (RMS)

### Radiation Dose Calculation

#### Beam Attenuation Model

The beam intensity decreases according to **Beer's law**:

```
I(z) = I₀ × exp(-∫₀ᶻ μₜₒₜₐₗ(z') dz')
```

For discrete voxels:
```
I(z) = I₀ × ∏ᵢ₌₀ᶻ⁻¹ exp(-μₜₒₜₐₗ(i) × Δz)
```

#### Energy Absorption

**Energy absorbed per voxel:**
```
Eₐᵦₛ = I(z) × μₑₙ(z) × Δz × Eₚₕₒₜₒₙ
```

where:
- `μₑₙ` = mass energy absorption coefficient (cm²/g)
- `Eₚₕₒₜₒₙ = hν` = photon energy (J)

#### Dose Calculation

**Absorbed dose (Gray):**
```
D = Eₐᵦₛ / m = Eₐᵦₛ / (ρ × Vᵥₒₓₑₗ)
```

where:
- `m` = mass of material in voxel (kg)
- `ρ` = material density (kg/m³)
- `Vᵥₒₓₑₗ` = voxel volume (m³)
- `1 Gy = 1 J/kg`

### Geometry Models

#### Tomography Geometry

**Parallel beam:** X-rays along z-axis, sample rotated by angle `θ`:

```
Rotation matrix: R(θ) = [cos(θ)  -sin(θ)  0]
                        [sin(θ)   cos(θ)  0]
                        [0        0       1]
```

#### Laminography Geometry

**Tilted beam:** Sample tilted by `α`, then rotated by `θ`:

```
Combined rotation: R(α,θ) = R_y(α) × R_z(θ)
                          = [cos(α)cos(θ)  -sin(θ)  sin(α)cos(θ)]
                            [cos(α)sin(θ)   cos(θ)  sin(α)sin(θ)]
                            [-sin(α)        0       cos(α)      ]
```

where:
- `α` = tilt angle from vertical
- `θ` = rotation angle around tilt axis

### Numerical Implementation

#### Sampling Requirements

**Fresnel number:** `F = a²/(λz)`

For proper sampling: `F < 1`

**Pixel size constraint:**
```
Δx ≤ √(λz/2)
```

#### Padding Strategy

To avoid wraparound artifacts:
```
Nₚₐ𝒹 = N + 2 × PAD
```

where `PAD ≥ 50` pixels recommended.

#### Memory Optimization

**GPU memory usage:**
```
M_GPU ≈ 6 × N_voxels × 4 bytes
```

for complex64 arrays (original + rotated + material maps + buffers).

### Validation Metrics

#### Quantitative Accuracy

**Beer's law validation:**
```
T_analytical = exp(-μ × t)
T_simulated = I_sim / I₀
Error = |T_simulated - T_analytical| / T_analytical
```

Target: `Error < 1%` for uniform materials.

#### Phase Contrast Validation

**Edge enhancement factor:**
```
EEF = (I_edge - I_background) / I_background
```

Compare with analytical Fresnel diffraction for simple edges.

## Performance

### GPU Acceleration
- **CuPy arrays** for all computations
- **Memory-efficient** volume handling
- **Batch processing** for multiple projections
- **PV streaming** with minimal overhead (<2ms per frame)

### Typical Performance
- **100×128×128 volume**: ~1-2 seconds per projection (RTX 4090)
- **Memory usage**: ~2× volume size in GPU RAM
- **Scaling**: Linear with number of projections
- **Streaming**: Real-time capable for typical volumes

## Validation

### Physics Accuracy
- **Quantitative agreement** with Beer's law (0.2% error)
- **Phase contrast validation** against analytical solutions
- **Dose calculations** verified with reference materials

### Test Suite
```bash
python test_simulation_physics.py  # Comprehensive physics tests
python test_rotation.py           # Geometry validation
python test_streaming.py          # PV streaming tests
```

## Examples

See `examples/` directory for:
- Basic tomography and laminography
- Real-time PV streaming
- Parameter studies (photon count, energy, geometry)
- Material-specific dose analysis
- Custom phantom generation
- Mouse brain phantom simulation
- Reconstruction workflows

## Troubleshooting

### PV Streaming Issues

**Viewer not connecting:**
```bash
# Check if PV is available
pvget SIM:TOMO

# List all PVs
pvlist | grep SIM

# Check firewall (Linux)
sudo firewall-cmd --add-port=5075-5076/tcp
```

**Image not displaying:**
- Check image normalization in viewer settings
- Try auto-scale or adjust min/max manually
- Verify data is being updated: watch frame counter

**Performance issues:**
- Reduce volume size or padding
- Disable detailed features (layers, vessels)
- Check GPU memory usage

### Common Errors

**"Config object not subscriptable":**
- Ensure `config` is a dictionary, not a class object
- Use `config = {...}` not `config = Config()`

**GPU out of memory:**
- Reduce volume size
- Reduce PAD value
- Clear GPU memory: `cp.get_default_memory_pool().free_all_blocks()`

## API Reference

### Core Classes
- `XRayScanner`: Main simulation interface
- `GPUVolumeManager`: Memory-efficient volume handling
- `StreamingManager`: PV streaming coordination

### Key Functions
- `projection()`: Core wave propagation simulation
- `simulate_projection_series()`: Series with optional streaming
- `quick_tomography()`, `quick_laminography()`: One-line scans
- `calculate_dose_map()`: Radiation dose computation
- `generate_phantom()`: Test phantom creation
- `keep_streaming_alive()`: Maintain PV server

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Run tests (`python -m pytest tests/`)
4. Commit changes (`git commit -m 'Add amazing feature'`)
5. Push to branch (`git push origin feature/amazing-feature`)
6. Open Pull Request


## Acknowledgments

- XRayLib for material properties
- Allen Mouse Brain Atlas for anatomical data
- ICRP Publication 110 for tissue compositions
- EPICS community for PVAccess protocol