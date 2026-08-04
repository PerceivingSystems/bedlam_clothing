# Shoe-on-body pipeline

`offset.py` builds an animated SMPL-X body and writes a Blender PC2 cache.
`shoeTexture.blend` reads that PC2 cache and applies the shoe material.

The script poses the body on a toeless v_template. It drives the body with an
AMASS motion sequence. It then lifts the body by a per-shoe vertical offset. The
offset puts the foot on top of the shoe sole.

```
bedlam_clothing/shoes/offset.py        bedlam_clothing/shoes/shoeTexture.blend
SMPL-X + toeless v_template + AMASS pose + offset
            │                                  Mesh Cache modifier reads the
            ▼                                  PC2 cache ──► open in Blender
   output/shoes/output.pc2  ─────────────────────►
   (Blender PC2 cache)
```

The script uses two functions from `bedlam_clothing.motion.io`: `save_pc2` and
`read_mesh_obj`. It also needs `torch` and `smplx`, which `pyproject.toml` lists.
There is no separate dependency list.

## The shoe is a material, not a mesh

This pipeline contains no shoe geometry. The shoe is a material that applies a
texture and a displacement to the whole body. The v_template is toeless for this
reason. It has no toes, and the displacement map shapes the shoe onto the foot
stump.

| Map | Feeds | Effect |
|-----|-------|--------|
| `color_map.png` | Colour and Alpha | Shoe albedo. The alpha channel covers about 2.7% of the UV space, which is the shoe island. The RGB channels are black outside it. |
| `normal_map.png` | Normal Map (tangent space) | Surface detail. |
| `displacement_map.png` | Displacement (object space, midlevel 0.5, scale 0.05) | Sole relief. The value depends on the shoe and reaches about 25 mm. |

The material calculates `Base Color = Colour + (1 − Alpha) × SkinColour`. The
`Skin Colour` group input keeps its default value of black, which matches the
original research scene. The body therefore renders as a black silhouette, and
the light shows only the shoe. If you want a skin tone, connect this input.

The displacement map shapes the sole. The pipeline does not model the sole as
geometry. The per-shoe offset in `shoe_offsets.csv` therefore sets the height of
the body.

## Contents
```
bedlam_clothing/shoes/
├── offset.py          # builds the body and writes output.pc2
├── shoeTexture.blend  # scene: body mesh, shoe material, camera, lights, ground
└── shoe_offsets.csv   # per-shoe vertical offsets (182 shoes)
```

## Prerequisites (download these separately)

This repository does not include the files below. `offset.py` needs the body
model, the v_template and the motion sequence. `shoeTexture.blend` needs the
texture maps of one shoe.

| Asset | Where it goes | How |
|-------|---------------|-----|
| SMPL-X neutral locked-head model | `support_data/body_models/smplx_locked_head/SMPLX_NEUTRAL.npz` | `cd support_data/body_models/smplx_locked_head && ./download_smplx.sh` |
| Toeless v_template | `support_data/v_templates/smplx_neutral-lh_vtemplate_toeless.obj` | `cd support_data/v_templates && ./download_toeless_vtemplate.sh` |
| An AMASS motion sequence | any path, for example `bedlam_clothing/shoes/data/` | download from <https://amass.is.tue.mpg.de> |
| Shoe texture maps (182 shoes) | `support_data/shoes/` | `cd support_data/shoes && ./download_shoes.sh` |

The archive has one folder for each shoe:

```
support_data/shoes/
├── shoe_offsets.csv                     # same content as bedlam_clothing/shoes/shoe_offsets.csv
└── texture_maps/
    └── shoe_gso_003/
        ├── color_map.png                # albedo and alpha
        ├── displacement_map.png         # sole height
        └── normal_map.png               # surface detail
```

Git ignores `bedlam_clothing/shoes/data/`, the downloaded contents of `support_data/shoes/` and
all `*.pc2` files.

### Which AMASS sequences work

Use a sequence from the released AMASS set. The script reads two arrays from the
`.npz` file: `trans`, and the first 66 columns of `poses`. Those 66 columns are
the global orientation and the 21 body joints. The SMPL+H release and the SMPL-X
release both use this layout, so a sequence from either one works. The script
ignores the hand, face and DMPL parameters.

The released set uses a Z-up world, and the offset lifts the body along Z. Raw
MoSh output is different. A file under an `amass/mosh_results/` path is still in
the SMPL-X canonical Y-up frame. On such a file the offset moves the body
sideways instead of up. Rotate the file to Z-up before you use it.

## Run

Build the PC2 cache from the repository root:

```bash
pip install -e .                       # once, per the root README
python -m bedlam_clothing.shoes.offset --motion <sequence>.npz --shoe-id shoe_gso_003
```

Select a shoe id from `shoe_offsets.csv`, for example `shoe_gso_120`. The ids run
from `shoe_gso_000` to `shoe_gso_254`, but the range has gaps. Only 182 of the
255 numbers are present. Read the file before you select an id.

The script accepts `--device cpu`, `--device cuda` or `--device auto`. If the
installed torch build does not support the local GPU, use `--device cpu`. The
script runs in a few seconds on either device.

The script prints the frame count of the cache and the frame rate of the source
sequence. Note both values. Next, open `bedlam_clothing/shoes/shoeTexture.blend` in Blender.

## How the scripts work

`offset.py` runs one SMPL-X forward pass for the whole sequence:

1. Read the vertices of the toeless v_template with `read_mesh_obj`.
2. Read `poses` and `trans` from the AMASS `.npz` file.
3. Build the SMPL-X model with `smplx.create` and give it those vertices as
   `v_template`.
4. Read the offset of the shoe id from `shoe_offsets.csv`.
5. Copy `trans` and add the offset to component 2, the world up axis.
6. Run the model and write the vertices with `save_pc2`.

The `v_template` argument replaces the shape of the standard SMPL-X body. The
script also sets every `betas` value to zero. The template alone therefore
defines the body, which is how the feet stay toeless.

The script poses the body only. It sets the hand, jaw and eye parameters to zero.
It reads `poses[:, :3]` for the global orientation and `poses[:, 3:66]` for the
21 body joints. One forward pass covers every frame, because `batch_size` equals
the frame count. A sequence of 362 frames takes about 3 seconds on a CPU.

`save_pc2` writes absolute world-space positions, not offsets from a rest pose.
The Mesh Cache modifier in the scene therefore uses `OVERWRITE` mode.

`download_shoes.sh` asks for your BEDLAM2.0 email address and password. It posts
them to the download server, saves `b2_shoes.tar.gz`, and extracts the archive
into the current directory. Run the script again to continue a stopped transfer.

## The scene

`shoeTexture.blend` contains one body object, `smplx_body`. The object has 10,475
vertices, which matches the PC2 cache. A Mesh Cache modifier on the object uses
`OVERWRITE` mode. `offset.py` bakes absolute world-space vertex positions. The
modifier therefore replaces the positions and does not add to them.

The original research file had no camera, no lights and no ground plane. This
version adds a camera, a key light, a fill light and a ground plane. The ground
plane shows the contact between the sole and the floor.

The scene does not encode the length of any one sequence. The frame range in the
file is 1 to 250, which is a placeholder. Set the range and the scene frame rate
to the two values that `offset.py` prints. AMASS sequences use different frame
rates, so check this for every sequence.

The Mesh Cache modifier reads `//../output/shoes/output.pc2`. To use a cache at
another path, change the file path in the modifier.

### Change the shoe

The scene points at one shoe: `shoe_gso_003`. `--shoe-id` changes only the offset
that `offset.py` bakes. It does not change the scene. Use the same shoe id in
both places. If the two differ, the sole thickness does not match the lift.

To show another shoe:

1. Open the Shading workspace.
2. Select the `smplx_body` object.
3. Set the file of each Image Texture node from the table below.

| Image Texture node | File under `support_data/shoes/texture_maps/<shoe id>/` |
|--------------------|---------------------------------------------------------|
| `Image Texture` | `color_map.png` |
| `Image Texture.001` | `normal_map.png` |
| `Image Texture.002` | `displacement_map.png` |

Keep the color space of each node. `color_map.png` uses sRGB. The other two use
Non-Color.

## Licenses

- Code: the license of this repository (`../../LICENSE.md`).
- You must get the SMPL-X model and the AMASS motion under their own licenses
  (<https://smpl-x.is.tue.mpg.de>, <https://amass.is.tue.mpg.de>). This
  repository does not include them.
- The toeless v_template is a BEDLAM2.0 asset
  (<https://bedlam2.is.tuebingen.mpg.de/license.html>).
- The shoe textures come from **Google Scanned Objects** under CC-BY-4.0. This
  repository does not include them. `shoeTexture.blend` refers to them by file
  name only. Credit Google if you redistribute them.
