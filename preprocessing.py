"""
Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

This work is licensed under a Creative Commons
Attribution-NonCommercial-ShareAlike 4.0 International License.
You should have received a copy of the license along with this
work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

Ref:
    https://github.com/NVlabs/edm2/blob/main/dataset_tool.py
"""

import argparse
import io
import json
import multiprocessing
import os
from pathlib import Path
from typing import Optional, Union

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

from utils import center_crop_arr

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def file_ext(name: Union[str, Path]) -> str:
    """Return the file extension without the leading ."""
    return os.path.splitext(str(name))[-1][1:]


def is_image_ext(fname: Union[str, Path]) -> bool:
    """Check if the extension is recognized by Pillow."""
    ext = file_ext(fname).lower()
    return f'.{ext}' in Image.EXTENSION


def maybe_min(a: int, b: Optional[int]) -> int:
    """Return min(a, b) if b is not None, else a."""
    return min(a, b) if b is not None else a


def open_image_folder_fnames(source_dir: str, *, max_images: Optional[int]):
    """Scan folder recursively for image files, load labels from dataset.json if present."""
    input_images = []

    def _recurse_dirs(root: str):
        with os.scandir(root) as it:
            for e in it:
                if e.is_file():
                    input_images.append(os.path.join(root, e.name))
                elif e.is_dir():
                    _recurse_dirs(os.path.join(root, e.name))

    _recurse_dirs(source_dir)
    input_images = sorted([f for f in input_images if is_image_ext(f)])
    max_idx = maybe_min(len(input_images), max_images)

    # Map absolute path -> rel path for label lookup
    arch_fnames = {fname: os.path.relpath(fname, source_dir).replace('\\', '/') for fname in input_images}

    # Load labels from dataset.json if present
    labels = {}
    meta_fname = os.path.join(source_dir, 'dataset.json')
    if os.path.isfile(meta_fname):
        with open(meta_fname, 'r') as f:
            data = json.load(f).get('labels', None)
            if data is not None:
                # data: list of [rel_path, class_idx]
                labels = {x[0]: x[1] for x in data}

    # If no labels from dataset.json, infer from top-level directory names
    if len(labels) == 0:
        toplevel_names = {}
        for fname in input_images:
            arch = arch_fnames[fname]
            top = arch.split('/')[0] if '/' in arch else ''
            toplevel_names[arch] = top
        unique_toplevels = sorted(set(toplevel_names.values()))
        if len(unique_toplevels) > 1:
            toplevel_indices = {n: i for i, n in enumerate(unique_toplevels)}
            labels = {arch: toplevel_indices[toplevel_names[arch]] for arch in toplevel_names}

    # Build list of (filename, label)
    out = []
    for fname in input_images[:max_idx]:
        rel = arch_fnames[fname]
        lbl = labels.get(rel, None)
        out.append((fname, lbl))
    return max_idx, out


def apply_transform(img: np.ndarray, resolution: int) -> Optional[np.ndarray]:
    """
    Apply a transform to `img` and return the resulting array (H, W, 3).
    Return None if we want to skip the image for any reason.
    """
    if img is None:
        return None
    return center_crop_arr(img, resolution)


def read_image(input_: str) -> np.ndarray:
    """
    Read and convert to (H,W,3) np.uint8 from a local path.
    Return None if something fails.
    """
    with Image.open(input_) as im:
        return np.array(im.convert('RGB'))


def _process_single_image(args):
    """
    Top-level worker function (picklable) that does:
        1) Read the image from file.
        2) Apply transform.
        3) Check dimensions.
        4) Encode as an uncompressed PNG (in-memory).
        5) Return a tuple: (arcname, image bytes, label).
    """
    global_idx, (input_, label), resolution, check_dims = args

    # 1) Read
    img = read_image(input_)
    if img is None:
        return (global_idx, None)  # skip if it fails to load

    # 2) Transform
    img = apply_transform(img, resolution)
    if img is None:
        return (global_idx, None)  # skip if transform returns None

    # 3) Check dimensions
    exp_w = check_dims["width"]
    exp_h = check_dims["height"]
    h, w, c = img.shape
    if (w != exp_w) or (h != exp_h):
        raise RuntimeError(f"[ERROR] Dimension mismatch: got {w}x{h}, expected {exp_w}x{exp_h}.")

    # 4) Encode as uncompressed PNG in memory
    idx_str = f"{global_idx:08d}"
    arcname = os.path.join(idx_str[:5], f"img{idx_str}.png")
    out_img = Image.fromarray(img)
    out_bytes = io.BytesIO()
    out_img.save(out_bytes, format="PNG", compress_level=0, optimize=False)

    # 5) Return arcname, image bytes, and label info
    return (global_idx, (arcname, out_bytes.getvalue(), label))


def create_dataset_in_group(h5file, full_path, data):
    """
    Create dataset at full_path in the h5file.
    If full_path contains group paths (e.g., "00000/img00000000.png"),
    this function ensures that the necessary groups are created.
    """
    parts = full_path.strip("/").split("/")
    if len(parts) == 1:
        h5file.create_dataset(full_path, data=np.array(data, dtype='S'))
    else:
        grp = h5file
        for part in parts[:-1]:
            if part not in grp:
                grp = grp.create_group(part)
            else:
                grp = grp[part]
        grp.create_dataset(parts[-1], data=np.array(data, dtype='S'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagenet-path", type=str, required=True,
                        help="Path to the ImageNet dataset")
    parser.add_argument("--output-path", type=str, default="data",
                        help="Path to the output HDF5 file")
    parser.add_argument("--resolution", type=int, default=256,
                        help="Preprocessing image resolution")
    parser.add_argument("--num-workers", type=int, default=16,
                        help="Number of workers to use")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Maximum number of images to process")
    args = parser.parse_args()

    # Initialize the PIL library.
    Image.init()

    # Use the output path as a directory to store our HDF5 file.
    os.makedirs(args.output_path, exist_ok=True)
    h5_path = os.path.join(args.output_path, "images.h5")
    h5_json_path = os.path.join(args.output_path, "images_h5.json")

    # Gather image filenames/labels from the source directory.
    print("Gathering image filenames/labels...")
    num_files, file_label_pairs = open_image_folder_fnames(args.imagenet_path, max_images=args.max_images)
    print(f"Found {num_files} images (possibly truncated by --max-images).")

    # Use one sample to check dimensions.
    sample_fname, _ = file_label_pairs[0]
    sample_arr = read_image(sample_fname)
    if sample_arr is None:
        raise ValueError("First sample image failed to load.")
    sample_arr = apply_transform(sample_arr, args.resolution)
    if sample_arr is None:
        raise ValueError("The first sample transform returned None.")
    h, w, c = sample_arr.shape
    if w != h:
        raise ValueError(f"Images must be square after transform; got {w}x{h}.")
    if w & (w - 1):
        raise ValueError("Width/height must be a power-of-two after transform.")

    check_dims = {"width": w, "height": h}

    # Prepare tasks for each image.
    tasks = []
    for i, (inp, lbl) in enumerate(file_label_pairs):
        tasks.append((i, (inp, lbl), args.resolution, check_dims))

    # Process images in parallel and write directly to the HDF5 file.
    metadata = []  # to store [arcname, label] pairs
    with h5py.File(h5_path, "w") as h5f:
        with multiprocessing.Pool(processes=args.num_workers) as pool:
            for global_idx, data in tqdm(
                pool.imap(_process_single_image, tasks, chunksize=32),
                total=len(tasks),
                desc="Processing"
            ):
                if data is None:
                    continue
                arcname, img_bytes, lbl = data
                create_dataset_in_group(h5f, arcname, img_bytes)
                if lbl is not None:
                    metadata.append([arcname, lbl])
        # Write dataset.json containing the metadata into the HDF5 file.
        dataset_json = json.dumps({"labels": metadata}).encode("utf-8")
        create_dataset_in_group(h5f, "dataset.json", dataset_json)

    # Optionally, save the list of dataset names (keys) into a JSON file.
    names = []
    with h5py.File(h5_path, "r") as h5f:
        h5f.visititems(lambda name, obj: names.append(name))
    with open(h5_json_path, "w") as f:
        json.dump(names, f, indent=2)

    print(f"Created HDF5 file: {h5_path}")
    print("Done.")
