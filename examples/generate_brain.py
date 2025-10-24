#!/usr/bin/env python3
"""
Generate a mouse brain phantom in Zarr format for X-ray simulation with dose calculation.
Includes anatomically relevant structures: cortex, white matter, ventricles, cerebellum, etc.
"""

import numpy as np
import json
import os
import shutil
import z5py

def save_multiscale_zarr(
    data,
    codes,
    out_dir,
    voxel_size=(0.25, 0.25, 0.1),
    n_scales=3,
    base_chunk=(64, 64, 64),
    logger=None
):
    """Save a 3D volume as a multiscale Zarr file with Neuroglancer-compatible metadata."""
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    f = z5py.File(out_dir, use_zarr_format=True)
    curr = data.copy()
    datasets = []
    for lvl in range(n_scales):
        path = str(lvl)
        if path in f:
            del f[path]
        chunks = tuple(min(c, s) for c, s in zip(base_chunk, curr.shape))
        f.create_dataset(
            name=path,
            data=curr,
            chunks=chunks,
            compression='raw'
        )
        scale = 2 ** lvl
        datasets.append({
            "path": path,
            "coordinateTransformations": [
                {"type": "scale", "scale": [scale] * 3},
                {"type": "translation", "translation": [scale / 2 - 0.5] * 3}
            ]
        })
        curr = curr[::2, ::2, ::2]
    
    lookup = {v: {'alias': k} for k, v in codes.items()}
    f.attrs['lookup'] = lookup
    f.attrs['voxel_size'] = voxel_size
    multiscale_meta = {
        "version": "0.4",
        "axes": [
            {"name": "z", "type": "space"},
            {"name": "y", "type": "space"},
            {"name": "x", "type": "space"}
        ],
        "datasets": datasets,
        "type": "image",
        "metadata": {"voxel_size": voxel_size}
    }
    f.attrs['multiscales'] = [multiscale_meta]
    
    with open(os.path.join(out_dir, "multiscale.json"), "w") as fjson:
        json.dump([multiscale_meta], fjson, indent=2)
    if logger:
        logger.info(f"Saved Neuroglancer-ready multiscale Zarr → {out_dir}")

def create_metadata_json(lookup_dict, voxel_size, output_path):
    """Create a separate JSON metadata file for simulation compatibility."""
    metadata = {
        "voxel_size": list(voxel_size),
        "lookup": lookup_dict
    }
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)

def create_mouse_brain_phantom(shape=(200, 160, 120), voxel_size=(0.05, 0.05, 0.05)):
    """
    Create an anatomically-inspired mouse brain phantom.
    
    Approximate mouse brain dimensions: 10mm (AP) x 8mm (ML) x 6mm (DV)
    With 50µm voxels: 200 x 160 x 120 voxels
    
    Structures (anterior to posterior):
    - Olfactory bulbs
    - Cerebral cortex (gray matter)
    - Corpus callosum and white matter
    - Hippocampus
    - Striatum (caudate-putamen)
    - Thalamus
    - Lateral ventricles
    - Cerebellum
    - Brainstem
    """
    nz, ny, nx = shape  # z: anterior-posterior, y: dorsal-ventral, x: medial-lateral
    volume = np.zeros(shape, dtype=np.int32)
    
    # Center coordinates
    center_z = (nz - 1) / 2.0
    center_y = (ny - 1) / 2.0
    center_x = (nx - 1) / 2.0
    
    # Create coordinate grids
    zz, yy, xx = np.meshgrid(
        np.arange(nz) - center_z,
        np.arange(ny) - center_y,
        np.arange(nx) - center_x,
        indexing='ij'
    )
    
    # === 1. Overall brain shape (ellipsoid) ===
    brain_a = nz * 0.45  # anterior-posterior
    brain_b = ny * 0.40  # dorsal-ventral
    brain_c = nx * 0.38  # medial-lateral
    
    brain_mask = (zz**2 / brain_a**2 + yy**2 / brain_b**2 + xx**2 / brain_c**2) <= 1
    volume[brain_mask] = 1  # gray matter (cortex) default
    
    # === 2. Olfactory bulbs (anterior) ===
    bulb_z_center = nz * 0.35
    bulb_radius = nx * 0.12
    bulb_length = nz * 0.15
    
    for side in [-1, 1]:
        bulb_x_offset = side * nx * 0.15
        bulb_zz = zz - bulb_z_center
        bulb_xx = xx - bulb_x_offset
        bulb_mask = (bulb_zz**2 / bulb_length**2 + bulb_xx**2 / bulb_radius**2 + yy**2 / bulb_radius**2) <= 1
        bulb_mask &= (zz > bulb_z_center - bulb_length) & (zz < bulb_z_center + bulb_length/3)
        volume[bulb_mask] = 1  # gray matter
    
    # === 3. Cerebellum (posterior-ventral) ===
    cereb_z_center = -nz * 0.28
    cereb_y_center = -ny * 0.15
    cereb_a = nz * 0.20
    cereb_b = ny * 0.25
    cereb_c = nx * 0.38
    
    cereb_zz = zz - cereb_z_center
    cereb_yy = yy - cereb_y_center
    cereb_mask = (cereb_zz**2 / cereb_a**2 + cereb_yy**2 / cereb_b**2 + xx**2 / cereb_c**2) <= 1
    volume[cereb_mask] = 8  # cerebellum
    
    # === 4. White matter (corpus callosum and internal capsule) ===
    # Central white matter band
    wm_thickness = ny * 0.15
    wm_width = nx * 0.55
    wm_length = nz * 0.50
    
    wm_mask = (np.abs(yy) <= wm_thickness) & (np.abs(xx) <= wm_width) & (np.abs(zz) <= wm_length)
    wm_mask &= brain_mask & ~cereb_mask
    volume[wm_mask] = 2  # white matter
    
    # === 5. Lateral ventricles (CSF-filled cavities) ===
    for side in [-1, 1]:
        vent_x_offset = side * nx * 0.15
        vent_y_offset = ny * 0.05
        vent_z_length = nz * 0.25
        vent_width = nx * 0.08
        vent_height = ny * 0.12
        
        vent_xx = xx - vent_x_offset
        vent_yy = yy - vent_y_offset
        vent_mask = (np.abs(vent_xx) <= vent_width) & (np.abs(vent_yy) <= vent_height) & (np.abs(zz) <= vent_z_length)
        volume[vent_mask] = 3  # CSF
    
    # === 6. Hippocampus (bilateral, curved structures) ===
    for side in [-1, 1]:
        hipp_x_offset = side * nx * 0.25
        hipp_y_offset = -ny * 0.10
        hipp_z_range = nz * 0.20
        
        for z_pos in np.linspace(-hipp_z_range, hipp_z_range, 40):
            # Curved trajectory
            curve_x = hipp_x_offset + side * 0.05 * nx * (z_pos / hipp_z_range)
            curve_y = hipp_y_offset - 0.08 * ny * np.abs(z_pos / hipp_z_range)
            
            hipp_zz = zz - z_pos
            hipp_yy = yy - curve_y
            hipp_xx = xx - curve_x
            hipp_radius = nx * 0.06
            
            hipp_sphere = (hipp_zz**2 + hipp_yy**2 + hipp_xx**2) <= hipp_radius**2
            volume[hipp_sphere & brain_mask] = 4  # hippocampus
    
    # === 7. Striatum (caudate-putamen, bilateral) ===
    for side in [-1, 1]:
        striatum_x = side * nx * 0.20
        striatum_y = ny * 0.08
        striatum_z = nz * 0.10
        
        striatum_zz = zz - striatum_z
        striatum_yy = yy - striatum_y
        striatum_xx = xx - striatum_x
        
        striatum_a = nz * 0.15
        striatum_b = ny * 0.15
        striatum_c = nx * 0.12
        
        striatum_mask = (striatum_zz**2 / striatum_a**2 + 
                        striatum_yy**2 / striatum_b**2 + 
                        striatum_xx**2 / striatum_c**2) <= 1
        volume[striatum_mask] = 5  # striatum
    
    # === 8. Thalamus (central structure) ===
    thalamus_y = ny * 0.00
    thalamus_z = nz * 0.00
    
    thalamus_yy = yy - thalamus_y
    thalamus_zz = zz - thalamus_z
    
    thalamus_a = nz * 0.12
    thalamus_b = ny * 0.12
    thalamus_c = nx * 0.18
    
    thalamus_mask = (thalamus_zz**2 / thalamus_a**2 + 
                     thalamus_yy**2 / thalamus_b**2 + 
                     xx**2 / thalamus_c**2) <= 1
    volume[thalamus_mask] = 6  # thalamus
    
    # === 9. Brainstem (posterior continuation) ===
    stem_z_center = -nz * 0.35
    stem_radius = nx * 0.15
    stem_length = nz * 0.18
    
    stem_zz = zz - stem_z_center
    stem_mask = (yy**2 + xx**2) <= stem_radius**2
    stem_mask &= (zz < stem_z_center + stem_length/2) & (zz > stem_z_center - stem_length)
    volume[stem_mask] = 7  # brainstem
    
    # === 10. Add some realistic vasculature (blood vessels) - FAST VERSION ===
    np.random.seed(42)
    n_vessels = 12
    for i in range(n_vessels):
        # Random vessel path
        start_z = np.random.uniform(-nz*0.3, nz*0.3)
        start_y = np.random.uniform(-ny*0.2, ny*0.3)
        start_x = np.random.uniform(-nx*0.3, nx*0.3)
        
        dir_z = np.random.uniform(-0.5, 0.5)
        dir_y = np.random.uniform(-0.3, 0.3)
        dir_x = np.random.uniform(-0.3, 0.3)
        norm = np.sqrt(dir_z**2 + dir_y**2 + dir_x**2)
        dir_z, dir_y, dir_x = dir_z/norm, dir_y/norm, dir_x/norm
        
        vessel_length = nz * 0.25
        vessel_radius = 1.0 + i * 0.15
        
        # Create vessel as series of spheres (much faster)
        n_points = 30
        for t in np.linspace(0, vessel_length, n_points):
            v_z = start_z + t * dir_z
            v_y = start_y + t * dir_y
            v_x = start_x + t * dir_x
            
            # Simple sphere distance
            dist_sq = (zz - v_z)**2 + (yy - v_y)**2 + (xx - v_x)**2
            vessel_mask = (dist_sq <= vessel_radius**2) & (volume > 0)
            volume[vessel_mask] = 9  # blood vessels
    
    return volume

def generate_mouse_brain():
    """Generate mouse brain phantom and save files."""
    
    # Material definitions for mouse brain tissues
    materials = {
        "vacuum": {"composition": {}, "density": 0.0},
        "gray_matter": {"composition": {"H": 10.7, "C": 14.5, "N": 2.2, "O": 7.2, "P": 0.4, "S": 0.2}, "density": 1.04},
        "white_matter": {"composition": {"H": 10.8, "C": 13.4, "N": 1.8, "O": 7.1, "P": 0.35, "S": 0.25}, "density": 1.04},
        "csf": {"composition": {"H": 2, "O": 1, "Na": 0.003, "Cl": 0.004}, "density": 1.007},
        "hippocampus": {"composition": {"H": 10.7, "C": 14.5, "N": 2.2, "O": 7.2, "P": 0.4, "S": 0.2}, "density": 1.04},
        "striatum": {"composition": {"H": 10.7, "C": 14.5, "N": 2.2, "O": 7.2, "P": 0.4, "S": 0.2}, "density": 1.04},
        "thalamus": {"composition": {"H": 10.7, "C": 14.5, "N": 2.2, "O": 7.2, "P": 0.4, "S": 0.2}, "density": 1.04},
        "brainstem": {"composition": {"H": 10.7, "C": 14.5, "N": 2.2, "O": 7.2, "P": 0.4, "S": 0.2}, "density": 1.04},
        "cerebellum": {"composition": {"H": 10.7, "C": 14.5, "N": 2.2, "O": 7.2, "P": 0.4, "S": 0.2}, "density": 1.04},
        "blood": {"composition": {"H": 10.2, "C": 11.0, "N": 3.3, "O": 7.5, "Fe": 0.05}, "density": 1.06}
    }
    
    # Generate volume
    shape = (200, 160, 120)  # ~10mm x 8mm x 6mm with 50µm voxels
    voxel_size = (0.05, 0.05, 0.05)  # 50 µm isotropic
    
    volume = create_mouse_brain_phantom(shape, voxel_size)
    
    codes = {
        "vacuum": 0,
        "gray_matter": 1,
        "white_matter": 2,
        "csf": 3,
        "hippocampus": 4,
        "striatum": 5,
        "thalamus": 6,
        "brainstem": 7,
        "cerebellum": 8,
        "blood": 9
    }
    
    lookup = {str(i): materials[name] for name, i in codes.items()}
    
    # Save files
    zarr_path = "phantom_mouse_brain.zarr"
    json_path = "phantom_mouse_brain.json"
    
    save_multiscale_zarr(volume, codes, zarr_path, voxel_size, n_scales=4)
    create_metadata_json(lookup, voxel_size, json_path)
    
    print(f"Generated mouse brain phantom:")
    print(f"  Shape: {volume.shape} voxels")
    print(f"  Physical size: {np.array(shape) * np.array(voxel_size)} mm")
    print(f"  Voxel size: {voxel_size} mm (50 µm isotropic)")
    print(f"  Structures: {list(codes.keys())}")
    print(f"  Unique labels: {np.unique(volume)}")
    print(f"  Files: {zarr_path}, {json_path}")
    
    print(f"\nMaterial properties for dose calculation:")
    for label, props in lookup.items():
        if props.get("composition"):
            comp_str = ', '.join(f"{el}:{amt}" for el, amt in props["composition"].items())
            print(f"  Label {label}: {comp_str}, ρ={props['density']:.3f} g/cm³")
        else:
            print(f"  Label {label}: vacuum, ρ={props['density']:.6f} g/cm³")
    
    print(f"\nLabel mapping:")
    for name, label in codes.items():
        count = np.sum(volume == label)
        percentage = 100 * count / volume.size
        print(f"  {label}: {name:20s} ({count:8d} voxels, {percentage:5.2f}%)")
    
    return zarr_path, json_path

if __name__ == "__main__":
    zarr_path, json_path = generate_mouse_brain()
    
    print(f"\nTo test simulation:")
    print(f"from msim.interface import quick_tomography, analyze_dose_only")
    print(f"projections, dose_stats = quick_tomography('{zarr_path}', '{json_path}', calculate_dose=True)")
    print(f"dose_map, dose_stats = analyze_dose_only('{zarr_path}', '{json_path}')")
