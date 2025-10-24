#!/usr/bin/env python3
"""
FAST realistic mouse brain phantom generator.
Optimized for speed while maintaining anatomical accuracy.
"""

import numpy as np
import json
import os
import shutil
import z5py
from scipy.ndimage import distance_transform_edt

# Set to False to skip slow features
ENABLE_CORTICAL_LAYERS = False  # Saves ~10 seconds
ENABLE_DETAILED_VASCULATURE = False  # Saves ~30 seconds

def save_multiscale_zarr(data, codes, out_dir, voxel_size, n_scales=4):
    """Fast zarr save."""
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    f = z5py.File(out_dir, use_zarr_format=True)
    curr = data.copy()
    datasets = []
    for lvl in range(n_scales):
        path = str(lvl)
        chunks = (64, 64, 64)
        f.create_dataset(name=path, data=curr, chunks=chunks, compression='raw')
        scale = 2 ** lvl
        datasets.append({
            "path": path,
            "coordinateTransformations": [
                {"type": "scale", "scale": [scale * v for v in voxel_size]}
            ]
        })
        if lvl < n_scales - 1:
            curr = curr[::2, ::2, ::2]
    
    lookup = {v: {'alias': k} for k, v in codes.items()}
    f.attrs['lookup'] = lookup
    f.attrs['voxel_size'] = voxel_size
    multiscale_meta = {
        "version": "0.4",
        "axes": [{"name": "z", "type": "space"}, {"name": "y", "type": "space"}, {"name": "x", "type": "space"}],
        "datasets": datasets
    }
    f.attrs['multiscales'] = [multiscale_meta]

def create_metadata_json(lookup_dict, voxel_size, output_path):
    """Create metadata JSON."""
    metadata = {"voxel_size": list(voxel_size), "lookup": lookup_dict}
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)

def draw_tube_fast(volume, start, end, radius, label, n_points=20):
    """Fast vectorized tube drawing."""
    points = np.linspace(start, end, n_points)
    for pt in points:
        z, y, x = pt.astype(int)
        r = int(radius) + 1
        zs, ys, xs = np.ogrid[-r:r+1, -r:r+1, -r:r+1]
        sphere = zs**2 + ys**2 + xs**2 <= radius**2
        
        z_start, z_end = max(0, z-r), min(volume.shape[0], z+r+1)
        y_start, y_end = max(0, y-r), min(volume.shape[1], y+r+1)
        x_start, x_end = max(0, x-r), min(volume.shape[2], x+r+1)
        
        z_slice = slice(max(0, r-z), min(sphere.shape[0], r+volume.shape[0]-z))
        y_slice = slice(max(0, r-y), min(sphere.shape[1], r+volume.shape[1]-y))
        x_slice = slice(max(0, r-x), min(sphere.shape[2], r+volume.shape[2]-x))
        
        volume[z_start:z_end, y_start:y_end, x_start:x_end][sphere[z_slice, y_slice, x_slice]] = label

def create_fast_mouse_brain(shape=(200, 160, 120), voxel_size=(0.05, 0.05, 0.05)):
    """
    Fast realistic mouse brain - same anatomy, optimized generation.
    Default: 200×160×120 @ 50µm = 10×8×6mm
    Use shape=(400,320,240) and voxel_size=(0.025,0.025,0.025) for high-res
    """
    nz, ny, nx = shape
    volume = np.zeros(shape, dtype=np.int32)
    
    cz, cy, cx = nz // 2, ny // 2, nx // 2
    
    print(f"Generating {shape[0]}×{shape[1]}×{shape[2]} brain...")
    
    # Coordinate grids
    zz, yy, xx = np.meshgrid(np.arange(nz) - cz, np.arange(ny) - cy, np.arange(nx) - cx, indexing='ij')
    
    # === 1. Brain envelope ===
    print("  Creating brain envelope...")
    brain_a, brain_b, brain_c = nz * 0.45, ny * 0.38, nx * 0.37
    brain_mask = (zz**2 / brain_a**2 + yy**2 / brain_b**2 + xx**2 / brain_c**2) <= 1
    volume[brain_mask] = 1  # gray matter
    
    # === 2. Meninges ===
    brain_dilated = (zz**2 / (brain_a*1.03)**2 + yy**2 / (brain_b*1.03)**2 + xx**2 / (brain_c*1.03)**2) <= 1
    volume[brain_dilated & ~brain_mask] = 23
    
    # === 3. Olfactory bulbs ===
    print("  Adding olfactory bulbs...")
    bulb_z, bulb_sep, bulb_r, bulb_l = nz * 0.38, nx * 0.16, nx * 0.10, nz * 0.12
    for side in [-1, 1]:
        bulb_mask = ((zz - bulb_z + cz)**2 / bulb_l**2 + (yy + ny * 0.05)**2 / bulb_r**2 + 
                     (xx - side * bulb_sep)**2 / bulb_r**2) <= 1
        bulb_mask &= (zz > bulb_z - bulb_l - cz)
        volume[bulb_mask] = 10
    
    # === 4. Cerebellum ===
    print("  Adding cerebellum...")
    cereb_z, cereb_y = -nz * 0.30, -ny * 0.12
    cereb_a, cereb_b, cereb_c = nz * 0.18, ny * 0.22, nx * 0.37
    cereb_mask = ((zz - cereb_z)**2 / cereb_a**2 + (yy - cereb_y)**2 / cereb_b**2 + xx**2 / cereb_c**2) <= 1
    cereb_mask &= (yy > cereb_y - cereb_b * 0.8)
    volume[cereb_mask] = 9
    
    # === 5. White matter (corpus callosum + internal capsule) ===
    print("  Adding white matter...")
    cc_mask = (np.abs(yy - ny * 0.08) < ny * 0.04) & (np.abs(xx) < nx * 0.40) & (np.abs(zz) < nz * 0.35)
    cc_mask &= brain_mask & ~cereb_mask
    volume[cc_mask] = 2
    
    for side in [-1, 1]:
        ic_mask = (np.abs(xx - side * nx * 0.15) < nx * 0.08) & (np.abs(yy - ny * 0.02) < ny * 0.15) & (np.abs(zz) < nz * 0.25)
        volume[ic_mask & brain_mask] = 2
    
    # === 6. Ventricles ===
    print("  Adding ventricles...")
    for side in [-1, 1]:
        lv_mask = (np.abs(xx - side * nx * 0.14) < nx * 0.07) & (np.abs(yy - ny * 0.08) < ny * 0.10) & (np.abs(zz) < nz * 0.22)
        volume[lv_mask & brain_mask] = 3
    
    third_v = (np.abs(xx) < nx * 0.02) & (np.abs(yy) < ny * 0.10) & (np.abs(zz) < nz * 0.15)
    volume[third_v & brain_mask] = 3
    
    fourth_v = (np.abs(xx) < nx * 0.08) & (np.abs(yy + ny * 0.05) < ny * 0.05) & (np.abs(zz + nz * 0.25) < nz * 0.08)
    volume[fourth_v & brain_mask] = 3
    
    # === 7. Hippocampus (simplified C-shape) ===
    print("  Adding hippocampus...")
    for side in [-1, 1]:
        for t in np.linspace(-1, 1, 30):
            z_pos = t * nz * 0.22
            curve_x = side * nx * 0.24 + side * nx * 0.06 * (t**2)
            curve_y = -ny * 0.08 - ny * 0.10 * np.abs(t)
            hipp_mask = ((zz - z_pos)**2 + (yy - curve_y)**2 + (xx - curve_x)**2) < (nx * 0.055)**2
            volume[hipp_mask & brain_mask] = 4
    
    # === 8. Striatum ===
    print("  Adding striatum...")
    for side in [-1, 1]:
        str_mask = ((zz - nz * 0.08)**2 / (nz * 0.16)**2 + (yy - ny * 0.05)**2 / (ny * 0.14)**2 + 
                    (xx - side * nx * 0.18)**2 / (nx * 0.11)**2) < 1
        volume[str_mask & brain_mask] = 5
    
    # === 9. Thalamus ===
    print("  Adding thalamus...")
    for side in [-1, 1]:
        thal_mask = ((zz - nz * 0.02)**2 / (nz * 0.13)**2 + yy**2 / (ny * 0.12)**2 + 
                     (xx - side * nx * 0.10)**2 / (nx * 0.09)**2) < 1
        volume[thal_mask & brain_mask] = 6
    
    # === 10. Hypothalamus ===
    hypo_mask = (np.abs(xx) < nx * 0.12) & (np.abs(yy + ny * 0.10) < ny * 0.08) & (np.abs(zz) < nz * 0.10)
    volume[hypo_mask & brain_mask] = 7
    
    # === 11. Amygdala ===
    for side in [-1, 1]:
        amyg_mask = ((zz - nz * 0.10)**2 + (yy + ny * 0.12)**2 + (xx - side * nx * 0.20)**2) < (nx * 0.06)**2
        volume[amyg_mask & brain_mask] = 19
    
    # === 12. Brainstem ===
    print("  Adding brainstem...")
    # Midbrain
    mid_mask = (yy**2 + xx**2) < (nx * 0.18)**2
    mid_mask &= (np.abs(zz + nz * 0.15) < nz * 0.10)
    volume[mid_mask & brain_mask] = 8
    
    # Superior/Inferior colliculus
    sc_mask = (np.abs(yy - ny * 0.02) < ny * 0.08) & (np.abs(xx) < nx * 0.15) & (np.abs(zz + nz * 0.13) < nz * 0.06)
    volume[sc_mask & brain_mask] = 17
    
    ic_mask = (np.abs(yy) < ny * 0.06) & (np.abs(xx) < nx * 0.12) & (np.abs(zz + nz * 0.19) < nz * 0.05)
    volume[ic_mask & brain_mask] = 18
    
    # Substantia nigra
    for side in [-1, 1]:
        sn_mask = (np.abs(xx - side * nx * 0.08) < nx * 0.05) & (np.abs(yy + ny * 0.05) < ny * 0.04) & (np.abs(zz + nz * 0.15) < nz * 0.06)
        volume[sn_mask & brain_mask] = 16
    
    # Pons + Medulla
    pons_mask = (yy**2 + xx**2) < (nx * 0.16)**2
    pons_mask &= (np.abs(zz + nz * 0.28) < nz * 0.08)
    volume[pons_mask & brain_mask] = 8
    
    med_mask = (yy**2 + xx**2) < (nx * 0.13)**2
    med_mask &= (np.abs(zz + nz * 0.38) < nz * 0.10)
    volume[med_mask & brain_mask] = 8
    
    # === 13. Cortical layers (optional, slow) ===
    if ENABLE_CORTICAL_LAYERS:
        print("  Adding cortical layers (slow)...")
        dist = distance_transform_edt(brain_mask)
        cortical_depth = int(40 * voxel_size[0] / 0.025)  # Scale with resolution
        
        volume[(dist <= cortical_depth * 0.10) & (volume == 1)] = 11
        volume[(dist > cortical_depth * 0.10) & (dist <= cortical_depth * 0.30) & (volume == 1)] = 12
        volume[(dist > cortical_depth * 0.30) & (dist <= cortical_depth * 0.45) & (volume == 1)] = 13
        volume[(dist > cortical_depth * 0.45) & (dist <= cortical_depth * 0.70) & (volume == 1)] = 14
        volume[(dist > cortical_depth * 0.70) & (dist <= cortical_depth) & (volume == 1)] = 15
    
    # === 14. Vasculature ===
    if ENABLE_DETAILED_VASCULATURE:
        print("  Adding detailed vasculature (slow)...")
        # MCA
        for side in [-1, 1]:
            start = np.array([nz*0.1 + cz, cy, side*nx*0.25 + cx])
            end = np.array([nz*0.3 + cz, ny*0.35 + cy, side*nx*0.40 + cx])
            draw_tube_fast(volume, start, end, 2.0, 20, n_points=30)
        
        # ACA
        start = np.array([nz*0.05 + cz, cy, cx])
        end = np.array([nz*0.25 + cz, ny*0.35 + cy, cx])
        draw_tube_fast(volume, start, end, 1.5, 20, n_points=20)
        
        # Capillaries (reduced number)
        np.random.seed(42)
        for _ in range(30):
            valid = np.argwhere((volume > 0) & (volume < 20))
            if len(valid) > 0:
                idx = valid[np.random.randint(len(valid))]
                direction = np.random.randn(3)
                direction = direction / np.linalg.norm(direction) * 15
                end = idx + direction
                draw_tube_fast(volume, idx, end, 0.6, 21, n_points=10)
    else:
        print("  Adding simple vasculature (fast)...")
        # Simple fast vessels
        for side in [-1, 1]:
            # MCA approximation
            z_line = np.linspace(nz*0.1 + cz, nz*0.3 + cz, 20).astype(int)
            y_line = np.linspace(cy, ny*0.35 + cy, 20).astype(int)
            x_line = np.linspace(side*nx*0.25 + cx, side*nx*0.40 + cx, 20).astype(int)
            for z, y, x in zip(z_line, y_line, x_line):
                if 0 <= z < nz and 0 <= y < ny and 0 <= x < nx:
                    volume[z, y-1:y+2, x-1:x+2] = 20
        
        # Random capillaries (very simple)
        np.random.seed(42)
        for _ in range(50):
            z = np.random.randint(0, nz)
            y = np.random.randint(0, ny)
            x = np.random.randint(0, nx)
            if volume[z, y, x] > 0 and volume[z, y, x] < 20:
                volume[z, y, x] = 21
    
    print("✓ Brain structure complete")
    return volume

def get_realistic_tissue_compositions():
    """Accurate tissue compositions."""
    return {
        "air": {"composition": {}, "density": 0.0012},
        "gray_matter": {"composition": {"H": 10.7, "C": 14.5, "N": 2.2, "O": 7.2, "P": 0.4, "S": 0.2}, "density": 1.039},
        "white_matter": {"composition": {"H": 10.8, "C": 13.4, "N": 1.8, "O": 7.1, "P": 0.35, "S": 0.25}, "density": 1.041},
        "csf": {"composition": {"H": 11.0, "O": 8.84, "Na": 0.003, "Cl": 0.004}, "density": 1.007},
        "hippocampus": {"composition": {"H": 10.6, "C": 14.6, "N": 2.3, "O": 7.1, "P": 0.42, "S": 0.19}, "density": 1.040},
        "striatum": {"composition": {"H": 10.7, "C": 14.4, "N": 2.2, "O": 7.2, "P": 0.41, "S": 0.20}, "density": 1.038},
        "thalamus": {"composition": {"H": 10.7, "C": 14.3, "N": 2.3, "O": 7.2, "P": 0.40, "S": 0.19}, "density": 1.037},
        "hypothalamus": {"composition": {"H": 10.8, "C": 14.2, "N": 2.2, "O": 7.3, "P": 0.39, "S": 0.18}, "density": 1.036},
        "brainstem": {"composition": {"H": 10.7, "C": 14.0, "N": 2.1, "O": 7.3, "P": 0.38, "S": 0.21}, "density": 1.038},
        "cerebellum": {"composition": {"H": 10.6, "C": 14.3, "N": 2.2, "O": 7.2, "P": 0.40, "S": 0.19}, "density": 1.039},
        "olfactory_bulb": {"composition": {"H": 10.8, "C": 14.4, "N": 2.3, "O": 7.1, "P": 0.41, "S": 0.18}, "density": 1.037},
        "cortical_layer_I": {"composition": {"H": 11.0, "C": 14.5, "N": 2.3, "O": 7.2, "P": 0.4, "S": 0.2}, "density": 1.035},
        "cortical_layer_II_III": {"composition": {"H": 10.8, "C": 14.6, "N": 2.2, "O": 7.1, "P": 0.42, "S": 0.18}, "density": 1.038},
        "cortical_layer_IV": {"composition": {"H": 10.7, "C": 14.7, "N": 2.2, "O": 7.0, "P": 0.43, "S": 0.17}, "density": 1.040},
        "cortical_layer_V": {"composition": {"H": 10.6, "C": 14.5, "N": 2.3, "O": 7.1, "P": 0.44, "S": 0.19}, "density": 1.041},
        "cortical_layer_VI": {"composition": {"H": 10.7, "C": 14.4, "N": 2.2, "O": 7.2, "P": 0.41, "S": 0.20}, "density": 1.039},
        "substantia_nigra": {"composition": {"H": 10.5, "C": 14.2, "N": 2.2, "O": 7.3, "P": 0.39, "S": 0.20, "Fe": 0.015}, "density": 1.042},
        "superior_colliculus": {"composition": {"H": 10.7, "C": 14.3, "N": 2.2, "O": 7.2, "P": 0.40, "S": 0.19}, "density": 1.038},
        "inferior_colliculus": {"composition": {"H": 10.7, "C": 14.3, "N": 2.2, "O": 7.2, "P": 0.40, "S": 0.19}, "density": 1.038},
        "amygdala": {"composition": {"H": 10.7, "C": 14.4, "N": 2.2, "O": 7.2, "P": 0.41, "S": 0.19}, "density": 1.039},
        "arterial_blood": {"composition": {"H": 10.2, "C": 11.0, "N": 3.3, "O": 7.5, "Fe": 0.046}, "density": 1.060},
        "capillary_blood": {"composition": {"H": 10.2, "C": 11.0, "N": 3.3, "O": 7.4, "Fe": 0.046}, "density": 1.058},
        "venous_blood": {"composition": {"H": 10.2, "C": 11.0, "N": 3.3, "O": 7.3, "Fe": 0.046}, "density": 1.057},
        "meninges": {"composition": {"H": 9.4, "C": 20.7, "N": 6.2, "O": 6.2, "S": 0.5, "Ca": 0.1}, "density": 1.130}
    }

def generate_mouse_brain_fast(high_res=False):
    """
    Generate mouse brain phantom quickly.
    
    Args:
        high_res: If True, use 400×320×240 @ 25µm (slower but better quality)
                  If False, use 200×160×120 @ 50µm (fast, good quality)
    """
    print("="*70)
    print("FAST REALISTIC MOUSE BRAIN PHANTOM GENERATOR")
    print("="*70)
    
    if high_res:
        shape = (400, 320, 240)
        voxel_size = (0.025, 0.025, 0.025)
        print("Mode: HIGH RESOLUTION (400×320×240 @ 25µm)")
        print("Warning: This will take 2-3 minutes")
    else:
        shape = (200, 160, 120)
        voxel_size = (0.05, 0.05, 0.05)
        print("Mode: STANDARD RESOLUTION (200×160×120 @ 50µm)")
        print("Expected time: 10-30 seconds")
    
    print(f"Cortical layers: {'ENABLED' if ENABLE_CORTICAL_LAYERS else 'DISABLED (saves time)'}")
    print(f"Detailed vessels: {'ENABLED' if ENABLE_DETAILED_VASCULATURE else 'DISABLED (saves time)'}")
    print("="*70)
    
    import time
    t0 = time.time()
    
    # Generate
    volume = create_fast_mouse_brain(shape, voxel_size)
    
    # Materials
    materials = get_realistic_tissue_compositions()
    
    # Codes
    codes = {
        "air": 0, "gray_matter": 1, "white_matter": 2, "csf": 3,
        "hippocampus": 4, "striatum": 5, "thalamus": 6, "hypothalamus": 7,
        "brainstem": 8, "cerebellum": 9, "olfactory_bulb": 10,
        "cortical_layer_I": 11, "cortical_layer_II_III": 12, "cortical_layer_IV": 13,
        "cortical_layer_V": 14, "cortical_layer_VI": 15,
        "substantia_nigra": 16, "superior_colliculus": 17, "inferior_colliculus": 18,
        "amygdala": 19, "arterial_blood": 20, "capillary_blood": 21, "venous_blood": 22,
        "meninges": 23
    }
    
    lookup = {str(i): materials[name] for name, i in codes.items()}
    
    # Save
    suffix = "_hires" if high_res else ""
    zarr_path = f"phantom_mouse_brain_fast{suffix}.zarr"
    json_path = f"phantom_mouse_brain_fast{suffix}.json"
    
    print("\nSaving files...")
    save_multiscale_zarr(volume, codes, zarr_path, voxel_size)
    create_metadata_json(lookup, voxel_size, json_path)
    
    t1 = time.time()
    
    # Stats
    print("\n" + "="*70)
    print(f"✓ COMPLETED IN {t1-t0:.1f} SECONDS")
    print("="*70)
    print(f"Shape: {volume.shape}")
    print(f"Size: {np.array(shape) * np.array(voxel_size)} mm")
    print(f"Files: {zarr_path}, {json_path}")
    
    print("\nStructure distribution:")
    for name, label in sorted(codes.items(), key=lambda x: x[1]):
        count = np.sum(volume == label)
        if count > 0:
            pct = 100 * count / volume.size
            print(f"  {label:2d}: {name:25s} {count:8d} voxels ({pct:5.2f}%)")
    
    print("\n" + "="*70)
    print("Usage:")
    print(f"  from msim.simulator import quick_tomography")
    print(f"  projs, dose = quick_tomography('{zarr_path}', '{json_path}', calculate_dose=True)")
    print("="*70)
    
    return zarr_path, json_path

if __name__ == "__main__":
    import sys
    
    # Parse arguments
    high_res = "--high-res" in sys.argv or "-hr" in sys.argv
    
    if "--enable-layers" in sys.argv:
        ENABLE_CORTICAL_LAYERS = True
    if "--enable-vessels" in sys.argv:
        ENABLE_DETAILED_VASCULATURE = True
    
    print("\nOptions:")
    print("  python generate_fast_brain.py              # Standard res, fast")
    print("  python generate_fast_brain.py --high-res   # High res, slower")
    print("  python generate_fast_brain.py --enable-layers  # Add cortical layers (+10s)")
    print("  python generate_fast_brain.py --enable-vessels # Detailed vessels (+30s)")
    print()
    
    generate_mouse_brain_fast(high_res=high_res)