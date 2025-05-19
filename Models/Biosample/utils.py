import z5py
import os
import json
import shutil

def ball_footprint(r):
    z, y, x = np.ogrid[-r:r+1, -r:r+1, -r:r+1]
    return (x*x + y*y + z*z) <= (r*r)


def save_mzarr(data, codes, out_dir,
                         voxel_size, n_scales, base_chunk, logger):
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    f = z5py.File(out_dir, use_zarr_format=True)
    curr = data
    datasets = []
    for lvl in range(n_scales):
        path   = str(lvl)
        chunks = tuple(min(c, s) for c, s in zip(base_chunk, curr.shape))
        f.create_dataset(path, data=curr, chunks=chunks, compression="raw")
        scale = 2**lvl
        datasets.append({
            "path": path,
            "coordinateTransformations": [
                {"type": "scale",       "scale": [scale]*3},
                {"type": "translation", "translation": [scale/2-0.5]*3},
            ]
        })
        curr = curr[::2, ::2, ::2]

    f.attrs["lookup"]     = {int(k):v for k,v in codes.items()}
    f.attrs["voxel_size"] = voxel_size
    multiscale_meta = {
        "version": "0.4",
        "axes": [
            {"name":"z","type":"space"},
            {"name":"y","type":"space"},
            {"name":"x","type":"space"},
        ],
        "datasets": datasets,
        "type": "image",
        "metadata": {"voxel_size": voxel_size},
    }
    f.attrs["multiscales"] = [multiscale_meta]

    with open(os.path.join(out_dir, "multiscale.json"), "w") as fj:
        json.dump([multiscale_meta], fj, indent=2)

    if logger:
        logger.info(f"Saved multiscale Zarr → {out_dir}")
