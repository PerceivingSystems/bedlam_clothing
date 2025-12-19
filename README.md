# BEDLAM2.0 assets processing tools (WORK IN PROGRESS)

This repository contains processing code for the BEDLAM2.0 NeurIPS 2025 paper.

If you are looking for code to train and evaluate the ML models from the paper then please visit this
repository: https://github.com/pixelite1201/BEDLAM

If you are looking for the BEDLAM2.0 render pipeline tools then please visit this
repository: https://github.com/PerceivingSystems/bedlam2_render

### Installing dependencies

Create a virtual environment and install the dependencies by running

```commandline
pip install -e .
```

### Prerequisites

Download the SMPL-X models by running
```
cd support_data/body_models/smplx_locked_head
./download_smplx.sh
```
As an alternative, download them manually and place them in the `support_data/body_models/smplx_locked_head` folder so that the following files exist:

- `support_data/body_models/smplx_locked_head/SMPLX_NEUTRAL.npz`
- `support_data/body_models/smplx_locked_head/SMPLX_FEMALE.npz`
- `support_data/body_models/smplx_locked_head/SMPLX_MALE.npz`

### Exporting SMPL-X sequences as Alembic `abc` for use in Unreal Engine

To export a `npz` SMPL-X motion as an Unreal-compatible `abc`, use 
```
python -m scripts.export_motions_as_abc <ARGUMENTS> 
```
For help on the arguments, run
```
python -m scripts.export_motions_as_abc -h
```

#### Exporting "toeless" SMPL-X motions (for adding shoes)
To represent shoes, we use a special SMPL-X v_template with a stocking-like foot (a.k.a. **toeless**). We can apply
displacement maps to this v_template, while still mainitaining the deformation provided by the betas, in order to deform the feet in the shape of a shoe.

You can download the SMPL-X toeless v_template from the BEDLAM2.0 website or by running the following commands:

```commandline
cd support_data/v_templates
./download_toeless_vtemplate.sh
```

To export a motion as `abc` using the toeless v_template, run
```
python -m scripts.export_motions_as_abc <OTHER_ARGUMENTS> --v_template_fname support_data/v_templates/smplx_neutral-lh_vtemplate_toeless.obj
```

# Citation

You acknowledge that the Data & Software is a valuable scientific resource and agree to appropriately reference the
following paper in any publication making use of the Data & Software.

Citation:

```
@inproceedings{tesch2025bedlam2,
  title={{BEDLAM}2.0: Synthetic humans and cameras in motion},
  author={Joachim Tesch and Giorgio Becherini and Prerana Achar and Anastasios Yiannakidis and Muhammed Kocabas and Priyanka Patel and Michael J. Black},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year={2025}
}
```
