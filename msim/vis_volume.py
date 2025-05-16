#!/usr/bin/env python3
import os
import json
import numpy as np
import z5py
import h5py
from vedo import Volume, Text2D
from vedo.applications import Slicer3DPlotter, RayCastPlotter

def view_volume(volume_path):
    # -- Determine basename and JSON path --
    basename, ext = os.path.splitext(volume_path)
    ext = ext.lower()
    json_path = basename + ".json"

    # -- Open file and load data array --
    if ext in (".h5", ".hdf5", ".hdf"):
        f = h5py.File(volume_path, "r")
        if "exchange/data" in f:
            data = f["exchange/data"][:]
        elif "reconstruction" in f:
            data = f["reconstruction"][:]
        else:
            print(f"Error: '/exchange/data' not found in {volume_path}")
            return
        f.close()
        print(f"Loaded HDF5: {volume_path} → /exchange/data | shape: {data.shape}, dtype: {data.dtype}")

    else:
        # N5/Zarr path
        f = z5py.File(volume_path, use_zarr_format=False)
        # recursive first dataset finder
        def find_first_dataset_z5(group):
            for name, val in group.items():
                if isinstance(val, z5py.Group):
                    sub = find_first_dataset_z5(val)
                    if sub:
                        return f"{name}/{sub}"
                else:
                    return name
            return None

        dataset_path = find_first_dataset_z5(f)
        if not dataset_path:
            print("No dataset found in", volume_path)
            return

        data = f[dataset_path][:]
        print(f"Loaded N5/Zarr: {volume_path} → {dataset_path} | shape: {data.shape}, dtype: {data.dtype}")
        f.close()

    # -- Ensure exactly 3D --
    data = np.squeeze(data)
    if data.ndim != 3:
        print(f"Error: expected 3D array, got shape {data.shape}")
        return

    # -- Handle different bit-depths (float → uint16, signed→unsigned, keep uint8/16/32) --
    if np.issubdtype(data.dtype, np.floating):
        mn, mx = data.min(), data.max()
        data = (data - mn) / (mx - mn) if mx > mn else np.zeros_like(data)
        data = (data * np.iinfo(np.uint16).max).astype(np.uint16)
    elif np.issubdtype(data.dtype, np.signedinteger):
        bits = data.dtype.itemsize * 8
        if bits <= 8:
            data = data.astype(np.uint8)
        elif bits <= 16:
            data = data.astype(np.uint16)
        else:
            data = data.astype(np.uint32)
    elif data.dtype in (np.uint8, np.uint16, np.uint32):
        pass
    else:
        data = data.astype(np.uint16)

    data = np.ascontiguousarray(data)

    # -- Load sidecar JSON metadata if present --
    voxel_size = [1.0, 1.0, 1.0]
    description = "No description"
    if os.path.exists(json_path):
        with open(json_path, "r") as jf:
            meta = json.load(jf)
        voxel_size  = meta.get("voxel_size", voxel_size)
        description = meta.get("description", description)

    # -- Build the Vedo volume actor --
    vol = Volume(data)
    vol.spacing(voxel_size[::-1])  # Vedo expects (dx, dy, dz) reversed
    vol.cmap("bone_r")
    vol.alpha([0, 0.1, 0.3, 0.6, 0.8, 1.0])
    vol.mode(0)

    # -- On-screen labels and state --
    legend     = Text2D("Mode: SLICER — press R to switch", pos="top-left", c="white", s=1.2)
    desc_label = Text2D(description,               pos="bottom-left", c="white", s=1.0)
    state = {"is_slicer": True}

    # -- Mode launchers --
    def launch_slicer():
        slicer = Slicer3DPlotter(vol,
                                 cmaps=["bone_r", "jet", "plasma"],
                                 use_slider3d=False,
                                 bg="black", bg2="black")
        legend.text("Mode: SLICER — press R to switch")
        slicer += legend; slicer += desc_label
        slicer.add_callback("KeyPress", toggle_mode)
        slicer.show(viewup="z"); slicer.close()

    def launch_render():
        renderer = RayCastPlotter(vol, bg="black", bg2="black", axes=7)
        legend.text("Mode: RENDER — press S to switch")
        renderer += legend; renderer += desc_label
        renderer.add_callback("KeyPress", toggle_mode)
        renderer.show(viewup="z"); renderer.close()

    def toggle_mode(evt):
        key = evt.keypress.lower()
        if key == 'r' and state["is_slicer"]:
            state["is_slicer"] = False
            launch_render()
        elif key == 's' and not state["is_slicer"]:
            state["is_slicer"] = True
            launch_slicer()

    # -- Start in slicer mode --
    launch_slicer()
