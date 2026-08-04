#  Copyright (c) 2025 Max Planck Society
#  License: https://bedlam2.is.tuebingen.mpg.de/license.html
#
# Shoe-on-body pipeline: generate an animated SMPL-X body mesh from an AMASS
# motion sequence and bake it to a Blender PC2 point cache.
#
# The body is posed on a toeless v_template, driven by an AMASS pose sequence,
# and lifted by a per-shoe vertical offset so the foot rests on top of the shoe
# sole. The per-frame vertices are written to a PC2 cache that the accompanying
# Blender scene plays back via a Mesh Cache modifier (bedlam_clothing/shoes/shoeTexture.blend).
#
# Run from the repository root:
#   python -m bedlam_clothing.shoes.offset \
#       --motion bedlam_clothing/shoes/data/<sequence>.npz --shoe-id shoe_gso_003
#
# Any sequence from the released AMASS set works, SMPL+H or SMPL-X. Only
# 'poses[:, :66]' (global orient + 21 body joints) and 'trans' are read, and both
# releases lay those out the same way.
#
# The released set is Z-up and the offset lifts along Z. Raw MoSh output (an
# 'amass/mosh_results/...' path) is still in the SMPL-X canonical Y-up frame and
# is NOT supported -- rotate it to Z-up first.
#
# License-gated inputs are NOT shipped with the repo. Download them first:
#   support_data/body_models/smplx_locked_head/download_smplx.sh   (SMPL-X)
#   support_data/v_templates/download_toeless_vtemplate.sh          (template)
# and get an AMASS sequence from https://amass.is.tue.mpg.de (see bedlam_clothing/shoes/README.md).
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import smplx

from bedlam_clothing.motion.io import read_mesh_obj, save_pc2

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parents[1]
DEFAULT_SMPLX = REPO_ROOT / "support_data" / "body_models" / "smplx_locked_head" / "SMPLX_NEUTRAL.npz"
DEFAULT_TEMPLATE = REPO_ROOT / "support_data" / "v_templates" / "smplx_neutral-lh_vtemplate_toeless.obj"
DEFAULT_OFFSETS_CSV = PACKAGE_DIR / "shoe_offsets.csv"
DEFAULT_OUT = REPO_ROOT / "output" / "shoes" / "output.pc2"


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--shoe-id", default="shoe_gso_003",
                   help="Shoe id to look up in the offsets CSV "
                        "(default: shoe_gso_003, which reproduces the reference cache).")
    p.add_argument("--smplx-model", type=Path, default=DEFAULT_SMPLX,
                   help="SMPL-X neutral locked-head model .npz (download separately).")
    p.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE,
                   help="Toeless SMPL-X v_template .obj (download separately).")
    p.add_argument("--motion", type=Path, required=True,
                   help="AMASS motion .npz with 'poses' and 'trans'. Any sequence from the "
                        "released SMPL+H or SMPL-X set works. Must be Z-up: raw MoSh output "
                        "is Y-up and is not supported (provide yourself).")
    p.add_argument("--offsets-csv", type=Path, default=DEFAULT_OFFSETS_CSV,
                   help="Path to shoe_offsets.csv.")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT,
                   help="Output PC2 path (default: output/shoes/output.pc2).")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                   help="Torch device (default: auto -> cuda if available else cpu).")
    return p.parse_args()


def load_offset(csv_path: Path, shoe_id: str) -> float:
    """Return the float offset for ``shoe_id`` from the offsets CSV (id,offset)."""
    import csv
    with open(csv_path, newline="") as f:
        table = {row[0]: row[1] for row in csv.reader(f) if row}
    table.pop("shoe_id", None)  # drop the header row
    if shoe_id not in table:
        raise KeyError(f"Shoe id {shoe_id!r} not found in {csv_path} "
                       f"({len(table)} entries, e.g. {list(table)[:3]}).")
    return float(table[shoe_id])


def main():
    args = parse_args()

    for path in (args.smplx_model, args.template, args.motion, args.offsets_csv):
        if not Path(path).exists():
            raise FileNotFoundError(
                f"Missing required input: {path}\n"
                f"See bedlam_clothing/shoes/README.md -- SMPL-X, the toeless template, and the AMASS "
                f"sequence must be downloaded/placed before running."
            )

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    vertices = read_mesh_obj(args.template, v_only=True)["v"]
    vertices = torch.tensor(vertices, dtype=torch.float, device=device)

    motion = np.load(str(args.motion), allow_pickle=True)
    missing = [k for k in ("poses", "trans") if k not in motion]
    if missing:
        raise KeyError(f"{args.motion} does not look like an AMASS sequence: missing "
                       f"{missing}, found {sorted(motion.files)}.")
    # 66 = 3 global orient + 63 body pose. Both the SMPL+H (156) and the SMPL-X
    # (165) AMASS layouts start with these, so either release works.
    n_pose_params = motion["poses"].shape[1]
    if n_pose_params < 66:
        raise ValueError(f"{args.motion} has {n_pose_params} pose parameters, "
                         f"but at least 66 are needed (3 global orient + 63 body).")

    pose = torch.tensor(motion["poses"], dtype=torch.float, device=device)
    trans = torch.tensor(motion["trans"], dtype=torch.float, device=device)
    batch = pose.shape[0]
    # AMASS spells this key differently across releases.
    fps = next((float(motion[k]) for k in ("mocap_framerate", "mocap_frame_rate")
                if k in motion), None)

    # num_betas=10 reproduces the reference cache; betas are zeroed below, so the
    # exact count does not affect the output (no shape contribution).
    smpl_model = smplx.create(str(args.smplx_model), "smplx", num_betas=10,
                              use_pca=False, batch_size=batch,
                              v_template=vertices).to(device)

    offset = load_offset(args.offsets_csv, args.shoe_id)

    # Lift the body along the world up axis so the foot rests on top of the shoe
    # sole. The released AMASS set is Z-up, so the offset applies to component 2
    # only. This is wrong for Y-up sources such as raw MoSh output.
    transl = trans.clone()
    transl[:, 2] += offset

    output = smpl_model(betas=torch.zeros([batch, 10], device=device),
                        global_orient=pose[:, :3],
                        body_pose=pose[:, 3:66],
                        left_hand_pose=torch.zeros([batch, 45], device=device),
                        right_hand_pose=torch.zeros([batch, 45], device=device),
                        jaw_pose=torch.zeros([batch, 3], device=device),
                        leye_pose=torch.zeros([batch, 3], device=device),
                        reye_pose=torch.zeros([batch, 3], device=device),
                        transl=transl)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    verts = output.vertices.cpu().detach().numpy()
    save_pc2(args.out, verts)
    print(f"Wrote {args.out}  "
          f"(frames={verts.shape[0]}, verts={verts.shape[1]}, "
          f"shoe={args.shoe_id}, offset={offset}, device={device})")
    print(f"In shoeTexture.blend, set the frame range to 1-{verts.shape[0]}"
          + (f" and the frame rate to {fps:g} fps." if fps is not None else "."))


if __name__ == "__main__":
    main()
