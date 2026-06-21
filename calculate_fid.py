import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch_fidelity


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate generated image folders with torch-fidelity.")
    parser.add_argument("--input-dir", type=str, required=True,
                        help="Path to a folder containing generated image files.")
    parser.add_argument("--fid-stats", type=str, default="imagenet_in256_stats.npz",
                        help="Path to the precomputed FID statistics .npz file.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
                        help="Device used by torch-fidelity.")
    parser.add_argument("--no-isc", action="store_true",
                        help="Disable Inception Score computation.")
    parser.add_argument("--kid", action="store_true",
                        help="Also compute KID.")
    parser.add_argument("--prc", action="store_true",
                        help="Also compute precision and recall.")
    parser.add_argument("--output-json", type=str, default=None,
                        help="Optional path to save the raw metric dictionary as JSON.")
    parser.add_argument("--quiet", action="store_true",
                        help="Disable verbose torch-fidelity logging.")
    return parser.parse_args()


def main():
    args = parse_args()

    input_dir = Path(args.input_dir)
    fid_stats = Path(args.fid_stats)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Generated image folder does not exist: {input_dir}")
    if not fid_stats.is_file():
        raise FileNotFoundError(f"FID statistics file does not exist: {fid_stats}")

    use_cuda = args.device == "cuda"
    if use_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available. Use --device cpu or run on a CUDA machine.")

    metrics_dict = torch_fidelity.calculate_metrics(
        input1=str(input_dir),
        input2=None,
        fid_statistics_file=str(fid_stats),
        cuda=use_cuda,
        isc=not args.no_isc,
        fid=True,
        kid=args.kid,
        prc=args.prc,
        verbose=not args.quiet,
    )

    def to_jsonable(value):
        if hasattr(value, "item"):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        return value

    serializable = {key: to_jsonable(value) for key, value in metrics_dict.items()}

    fid = serializable.get("frechet_inception_distance")
    if fid is not None:
        print(f"FID: {fid:.4f}")
    inception_score = serializable.get("inception_score_mean")
    if inception_score is not None:
        print(f"Inception Score: {inception_score:.4f}")
    if args.output_json is not None:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with output_json.open("w") as f:
            json.dump(serializable, f, indent=2)
        print(f"Saved metrics to {output_json}")


if __name__ == "__main__":
    main()
