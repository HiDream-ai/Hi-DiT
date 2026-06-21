import argparse
import gc
import json
import os

import accelerate
from accelerate.utils import gather_object
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CustomINH5Dataset, load_h5_file
from models.autoencoder import vae_models
from utils import preprocess_imgs_vae


def create_dataset_in_group(h5file, full_path, data):
    """
    Create a dataset at full_path in the h5file.
    If full_path contains group paths (e.g., "00000/img00000000.png"),
    this function ensures that the necessary groups are created.
    """
    parts = full_path.strip("/").split("/")
    if len(parts) == 1:
        h5file.create_dataset(full_path, data=data)
    else:
        grp = h5file
        for part in parts[:-1]:
            if part not in grp:
                grp = grp.create_group(part)
            else:
                grp = grp[part]
        grp.create_dataset(parts[-1], data=data)


class CustomINH5PathDataset(CustomINH5Dataset):
    """
    Inherits from CustomINH5Dataset and overrides the __getitem__ method
    """
    def __getitem__(self, index):
        image_fname = self.filelist[index]
        image = load_h5_file(self.h5f, image_fname)
        return torch.from_numpy(image), torch.tensor(self.labels[index]), image_fname


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae-arch", type=str, default="f16d32",
                        choices=["f8d4", "f16d32"], help="The architecture of the VAE.")
    parser.add_argument("--vae-ckpt-path", type=str, default="pretrained/repae-invae-400k/repae-invae-400k.pt",
                        help="Path to the VAE checkpoint.")
    parser.add_argument("--vae-latents-name", type=str, default="repae-invae-400k",
                        help="The name of the pre-extracted latents")
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Directory containing the ImageNet dataset to encode.")
    parser.add_argument("--output-dir", type=str, default="data",
                        help="Directory to save the latents.")
    parser.add_argument("--pproc-batch-size", type=int, default=128,
                        help="Batch size for encoding.")
    args = parser.parse_args()

    # Initialize Accelerate.
    accelerator = accelerate.Accelerator()
    device = accelerator.device

    # Load the VAE checkpoint.
    vae = vae_models[args.vae_arch]()
    state_dict = torch.load(args.vae_ckpt_path, map_location="cpu")
    vae.load_state_dict(state_dict)
    vae.to(device).eval()

    del state_dict
    gc.collect()
    torch.cuda.empty_cache()

    # Load the dataset.
    dataset = CustomINH5PathDataset(args.data_dir)
    dataloader = DataLoader(dataset, batch_size=args.pproc_batch_size, shuffle=False, num_workers=4)

    # Prepare the model and dataloader for multi-GPU inference.
    vae, dataloader = accelerator.prepare(vae, dataloader)

    # Create output paths.
    os.makedirs(args.output_dir, exist_ok=True)
    h5_path = os.path.join(args.output_dir, f"{args.vae_latents_name}.h5")
    h5_json_path = os.path.join(args.output_dir, f"{args.vae_latents_name}_h5.json")

    # These metadata lists are small relative to the latent vectors.
    metadata_labels = []
    metadata_filenames = []
    processed_filenames = set()

    # Open the HDF5 file in the main process for incremental writing.
    if accelerator.is_main_process:
        h5f = h5py.File(h5_path, "w")
    else:
        h5f = None

    # Process and write each batch immediately.
    for batch in tqdm(dataloader, desc="Encoding"):
        b_images, b_labels, b_image_fnames = batch
        b_images = preprocess_imgs_vae(b_images.to(device))
        with torch.no_grad():
            posterior, _, _ = vae(b_images, return_recon=False)
            # encoded.parameters contains the latent tensor mean and std
            b_dists = torch.cat([posterior.mean, posterior.std], dim=1)

        # Gather the batch across GPUs.
        b_dists = accelerator.gather(b_dists)
        b_labels = accelerator.gather(b_labels)
        b_image_fnames = gather_object(b_image_fnames)

        # On the main process, write the gathered batch.
        if accelerator.is_main_process:
            b_dists_np = b_dists.cpu().numpy()
            b_labels_list = b_labels.tolist()

            # b_image_fnames is already available on the main process.
            for latent, fname, label in zip(b_dists_np, b_image_fnames, b_labels_list):
                # Check if the filename has already been processed, as we are using DDP...
                if fname in processed_filenames:
                    continue

                # Convert the filename to a unique key.
                archive_fname = fname.replace("img", "img-mean-std-").replace(".png", ".npy")

                # Save the data
                create_dataset_in_group(h5f, archive_fname, latent)
                metadata_labels.append((archive_fname, label))
                metadata_filenames.append(archive_fname)
                processed_filenames.add(fname)
        accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        # Write metadata into the HDF5 file.
        dataset_json = np.array(json.dumps({"labels": metadata_labels}).encode('utf-8'), dtype='S')
        create_dataset_in_group(h5f, "dataset.json", dataset_json)
        metadata_filenames.append("dataset.json")
        h5f.close()

        # Dump a list of all dataset keys into a separate JSON file.
        with open(h5_json_path, "w") as f:
            json.dump(metadata_filenames, f, indent=2)
        print(f"Created HDF5 file: {h5_path}")
    accelerator.wait_for_everyone()
