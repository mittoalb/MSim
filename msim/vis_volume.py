import os
import z5py
import json
import numpy as np
from vedo import Volume, Text2D
from vedo.applications import Slicer3DPlotter, RayCastPlotter

def view_volume(volume_path):
    # -- Determine base name and metadata path --
    basename = os.path.splitext(volume_path)[0] if volume_path.endswith('.n5') else volume_path
    json_path = basename + ".json"

    # -- Open z5py file (zarr or n5) --
    f = z5py.File(volume_path, use_zarr_format=False)

    # -- Recursively find the first dataset path --
    def find_first_dataset(group):
        for key, val in group.items():
            if isinstance(val, z5py.Group):
                sub = find_first_dataset(val)
                if sub:
                    return f"{key}/{sub}" if sub else key
            else:
                return key
        return None

    dataset_path = find_first_dataset(f)
    if not dataset_path:
        print("No dataset found.")
        return

    data = f[dataset_path][:]
    print(f"Loaded: {volume_path} → {dataset_path} | shape: {data.shape}, dtype: {data.dtype}")

    # -- Normalize if needed --
    if data.dtype != np.uint8:
        data = (data - data.min()) / (data.max() - data.min())
        data = (255 * data).astype(np.uint8)

    # -- Load metadata --
    meta = {}
    if os.path.exists(json_path):
        with open(json_path, "r") as jf:
            meta = json.load(jf)

    voxel_size = meta.get("voxel_size", [1.0, 1.0, 1.0])
    description = meta.get("description", "No description")

    # -- Create volume --
    vol = Volume(data)
    vol.spacing(voxel_size[::-1])
    vol.cmap("bone_r")
    vol.alpha([0, 0.1, 0.3, 0.6, 0.8, 1.0])
    vol.mode(0)

    # -- UI labels --
    legend = Text2D("Mode: SLICER — press R to switch", pos="top-left", c="white", s=1.2)
    desc_label = Text2D(description, pos="bottom-left", c="white", s=1.0)

    # -- Viewer toggles --
    state = {"is_slicer": True}  # mutable closure

    def launch_slicer():
        slicer = Slicer3DPlotter(
            vol,
            cmaps=["bone_r", "jet", "plasma"],
            use_slider3d=False,
            bg="black",
            bg2="black",
        )
        legend.text("Mode: SLICER — press R to switch")
        slicer += legend
        slicer += desc_label
        slicer.add_callback("KeyPress", toggle_mode)
        slicer.show(viewup="z")
        slicer.close()

    def launch_render():
        render = RayCastPlotter(vol, bg="black", bg2="black", axes=7)
        legend.text("Mode: RENDER — press S to switch")
        render += legend
        render += desc_label
        render.add_callback("KeyPress", toggle_mode)
        render.show(viewup="z")
        render.close()

    def toggle_mode(evt):
        if evt.keypress.lower() == 'r' and state["is_slicer"]:
            state["is_slicer"] = False
            launch_render()
        elif evt.keypress.lower() == 's' and not state["is_slicer"]:
            state["is_slicer"] = True
            launch_slicer()

    # -- Start viewer --
    launch_slicer()
