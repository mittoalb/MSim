#!/usr/bin/env python3
"""
view_volume.py — quick 3-D viewer (HDF5, N5, Zarr) using vedo.

• Mouse-wheel  = change slice (Slicer mode)
• R            = switch to Ray-cast render
• S            = back to Slicer
"""
import os
import sys
import json
import numpy as np
import h5py
import z5py
from vedo import Volume, Text2D
from vedo.applications import Slicer3DPlotter, RayCastPlotter


def _looks_like_zarr(path: str) -> bool:
    """Return True if `path` is a Zarr directory (has .zgroup or .zarray)."""
    return os.path.isdir(path) and (
        os.path.exists(os.path.join(path, ".zgroup"))
        or os.path.exists(os.path.join(path, ".zarray"))
    )


def _find_first_dataset_z5(group: z5py.Group) -> str | None:
    """Depth-first search for the first dataset inside a z5py group."""
    for name, val in group.items():
        if isinstance(val, z5py.Group):
            sub = _find_first_dataset_z5(val)
            if sub:
                return f"{name}/{sub}"
        else:
            return name
    return None


def view_volume(volume_path: str) -> None:
    # ---------------------------------------------------------------- I/O ---
    volume_path = volume_path.rstrip(os.sep)          # remove trailing “/”
    root, ext = os.path.splitext(volume_path)
    ext = ext.lower()
    json_path = root + ".json"

    if ext in (".h5", ".hdf5", ".hdf"):
        with h5py.File(volume_path, "r") as f:
            if "exchange/data" in f:
                data = f["exchange/data"][:]
            elif "reconstruction" in f:
                data = f["reconstruction"][:]
            else:
                print(f"Error: '/exchange/data' not found in {volume_path}")
                return
        print(f"Loaded HDF5: {volume_path} | shape={data.shape}, dtype={data.dtype}")

    else:
        is_zarr = (ext == ".zarr") or _looks_like_zarr(volume_path)
        f = z5py.File(volume_path, use_zarr_format=is_zarr)

        dset_path = _find_first_dataset_z5(f)
        if not dset_path:
            print(f"No dataset found in {volume_path}")
            return
        data = f[dset_path][:]
        f.close()

        kind = "Zarr" if is_zarr else "N5"
        print(f"Loaded {kind}: {volume_path} → {dset_path} | shape={data.shape}, dtype={data.dtype}")

    # ----------------------------------------------------------- preprocess --
    data = np.squeeze(data)
    if data.ndim != 3:
        print(f"Error: expected 3-D array, got shape {data.shape}")
        return

    if np.issubdtype(data.dtype, np.floating):
        mn, mx = data.min(), data.max()
        data = np.zeros_like(data) if mx == mn else (data - mn) / (mx - mn)
        data = (data * np.iinfo(np.uint16).max).astype(np.uint16)
    elif np.issubdtype(data.dtype, np.signedinteger):
        bits = data.dtype.itemsize * 8
        data = data.astype(np.uint8 if bits <= 8 else np.uint16 if bits <= 16 else np.uint32)
    elif data.dtype not in (np.uint8, np.uint16, np.uint32):
        data = data.astype(np.uint16)
    data = np.ascontiguousarray(data)

    # ---------------------------------------------------------- metadata ----
    voxel_size = [1.0, 1.0, 1.0]
    description = "No description"
    if os.path.exists(json_path):
        with open(json_path) as jf:
            meta = json.load(jf)
        voxel_size  = meta.get("voxel_size", voxel_size)
        description = meta.get("description", description)

    # ------------------------------------------------------------- vedo -----
    vol = Volume(data).spacing(voxel_size[::-1])
    vol.cmap("bone_r").alpha([0, 0.1, 0.3, 0.6, 0.8, 1.0]).mode(0)

    legend     = Text2D("Mode: SLICER — press R to switch",  pos="top-left",    c="white", s=1.2)
    desc_label = Text2D(description,                         pos="bottom-left", c="white", s=1.0)
    state = {"is_slicer": True}

    def launch_slicer():
        p = Slicer3DPlotter(vol, cmaps=["bone_r", "jet", "plasma"],
                            use_slider3d=False, bg="black", bg2="black")
        legend.text("Mode: SLICER — press R to switch")
        p += legend; p += desc_label
        p.add_callback("KeyPress", toggle_mode)
        p.show(viewup="z"); p.close()

    def launch_render():
        p = RayCastPlotter(vol, bg="black", bg2="black", axes=7)
        legend.text("Mode: RENDER — press S to switch")
        p += legend; p += desc_label
        p.add_callback("KeyPress", toggle_mode)
        p.show(viewup="z"); p.close()

    def toggle_mode(evt):
        k = evt.keypress.lower()
        if k == "r" and state["is_slicer"]:
            state["is_slicer"] = False
            launch_render()
        elif k == "s" and not state["is_slicer"]:
            state["is_slicer"] = True
            launch_slicer()

    launch_slicer()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: view_volume.py <path/to/volume.{h5|n5|zarr}>")
    else:
        view_volume(sys.argv[1])

