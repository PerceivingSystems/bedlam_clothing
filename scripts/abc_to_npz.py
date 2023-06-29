#  Copyright (c) 2023 Max Planck Society
#  License: https://bedlam.is.tuebingen.mpg.de/license.html
#
# Extracts the information from a clothing simulation stored in an Alembic file, performs subsampling of the frames
# (if required) and stores the result in a npz file. Optionally, the frames can be exported as Wavefront obj files.
#
# Usage:
# abc_to_npz.py [-h] [--input-abc-fname INPUT_ABC_FNAME]
#                    [--output-npz-fname OUTPUT_NPZ_FNAME]
#                    [--target-fps TARGET_FPS]                          (desired framerate of the output npz file)
#                    [--source-fps SOURCE_FPS]                          (framerate of the input abc file)
#                    [--export-first-frame-as-obj]                      (will export a single frame as an obj file)
#                    [--export-all-frames-as-obj]                       (will export all frames as obj files)
#                    [--start-frame-override START_FRAME_OVERRIDE]
#                    [--end-frame-override END_FRAME_OVERRIDE]
#                    [--offset OFFSET OFFSET OFFSET]                    (offset in x, y, z)
#
# Example:
# python -m scripts.abc_to_npz --input-abc-fname clothing_abc/rp_aaron_posed_018/clothing_simulations/1002/1002.abc
#                              --output-npz-fname clothing_npz_6fps/clothing_simulations/1002/1002.npz
#                              --target-fps 6
#                              --export-first-frame-as-obj

import argparse
import os
import os.path as osp
import tempfile
from dataclasses import dataclass
from typing import Optional

import bpy
import numpy as np
from loguru import logger

from bedlam_clothing.file_io.obj import read_mesh_obj, save_mesh_obj
from bedlam_clothing.utils.rotation import rotate_points_around_axis


@dataclass
class MeshSequence:
    vertices_seq: np.ndarray
    vertices_uv: np.ndarray
    faces: np.ndarray
    faces_uv: np.ndarray
    start_frame: int
    end_frame: int
    fps: int
    source_fps: int

    @property
    def frame_increment(self):
        return self.source_fps // self.fps

    @property
    def selected_frames(self):
        return np.arange(self.start_frame, self.end_frame + 1, self.frame_increment)

    @staticmethod
    def _extract_mesh_info_from_current_blender_scene():
        """
        Export the current frame to a temporary obj file to easily extract information about uvs and triangles.

        Returns:
            a dictionary containing the mesh information (vertices, faces, uv, uv faces)

        """
        tmp_obj_file = tempfile.NamedTemporaryFile(suffix=".obj", prefix="temp_", delete=True)

        bpy.ops.export_scene.obj(filepath=tmp_obj_file.name,
                                 use_selection=True,
                                 use_materials=False,
                                 use_triangles=True,
                                 use_blen_objects=False,
                                 use_normals=False,
                                 keep_vertex_order=True)

        mesh_info = read_mesh_obj(tmp_obj_file.name)
        tmp_obj_file.close()
        return mesh_info

    @staticmethod
    def init_blender_scene(abc_fname: str, source_fps: int):
        bpy.ops.scene.new(type='EMPTY')
        bpy.context.scene.render.fps = source_fps
        bpy.context.scene.render.fps_base = 1.0

        while bpy.data.objects:
            bpy.data.objects.remove(bpy.data.objects[0], do_unlink=True)

        bpy.ops.wm.alembic_import(filepath=abc_fname)

    @staticmethod
    def read_abc_mesh_sequence(abc_fname: str, target_fps: int = 30, source_fps: int = 30,
                               start_frame_override: int = None,
                               end_frame_override: int = None, offset: Optional[np.ndarray] = None):
        """
        Reads a mesh sequence from an Alembic file and stores it in a MeshSequence object. Optionally, the sequence can
        be subsampled by selecting only one frame every `frame_increment` frames.

        Args:
            abc_fname: the path to the Alembic file
            target_fps: the output fps of the sequence; if smaller than the source fps, subsampling will be performed
            source_fps: the fps of the Alembic file
            start_frame_override: if not None, overrides the start frame of the sequence
            end_frame_override: if not None, overrides the end frame of the sequence
            offset: if not None, adds this offset to the vertices of the sequence

        Returns:
            a MeshSequence object

        """

        if source_fps % target_fps != 0:
            raise ValueError(f'Target fps {target_fps} must be a divisor of source fps {source_fps}')

        frame_increment = source_fps // target_fps

        MeshSequence.init_blender_scene(abc_fname, source_fps)
        scene = bpy.context.scene
        obj = bpy.data.objects[0]
        start_frame = scene.frame_start if start_frame_override is None else start_frame_override
        end_frame = scene.frame_end if end_frame_override is None else end_frame_override

        bpy.context.scene.frame_set(start_frame)

        mesh_info = MeshSequence._extract_mesh_info_from_current_blender_scene()

        dg = bpy.context.evaluated_depsgraph_get()

        def extract_co(frame):
            bpy.context.scene.frame_set(frame)
            m = obj.evaluated_get(dg).to_mesh(preserve_all_data_layers=True, depsgraph=dg)
            return np.array([(obj.matrix_world @ v.co) for v in m.vertices], dtype=np.float32)

        vertices_seq = [extract_co(frame) for frame in range(start_frame, end_frame + 1, frame_increment)]
        vertices_seq = np.stack(vertices_seq)  # (n_frames, n_vertices, 3)
        vertices_seq = rotate_points_around_axis(vertices_seq, deg=-90, axis='x')

        if offset is not None:
            vertices_seq += offset.flatten().reshape((1, 1, 3))

        return MeshSequence(vertices_seq=vertices_seq,
                            vertices_uv=mesh_info['uv'],
                            faces=mesh_info['f'],
                            faces_uv=mesh_info['uvf'],
                            start_frame=start_frame,
                            end_frame=end_frame,
                            fps=target_fps,
                            source_fps=source_fps)

    def export(self, npz_fname, export_first_frame_as_obj=False, export_all_as_obj=False, extra_data: dict = None):
        os.makedirs(osp.dirname(npz_fname), exist_ok=True)
        metadata = {'fps': self.fps, 'source_fps': self.source_fps, 'start_frame': self.start_frame,
                    'end_frame': self.end_frame}
        extra_data = {} if extra_data is None else extra_data

        np.savez_compressed(npz_fname, vertices_seq=self.vertices_seq, vertices_uv=self.vertices_uv, faces=self.faces,
                            faces_uv=self.faces_uv, metadata=metadata, **extra_data)

        if export_all_as_obj:
            for i, f in enumerate(self.selected_frames):
                obj_fname = f'{osp.splitext(npz_fname)[0]}_{str(f).zfill(4)}.obj'
                save_mesh_obj(obj_fname, self.vertices_seq[i], self.faces, self.vertices_uv, self.faces_uv)
        elif export_first_frame_as_obj:
            obj_fname = f'{osp.splitext(npz_fname)[0]}_{str(self.start_frame).zfill(4)}.obj'
            save_mesh_obj(obj_fname, self.vertices_seq[0], self.faces, self.vertices_uv, self.faces_uv)


def process(args, extra_data: Optional[dict] = None):
    mesh_seq = MeshSequence.read_abc_mesh_sequence(args.input_abc_fname,
                                                   target_fps=args.target_fps,
                                                   start_frame_override=args.start_frame_override,
                                                   end_frame_override=args.end_frame_override,
                                                   offset=np.array(args.offset) if args.offset is not None else None)

    mesh_seq.export(args.output_npz_fname,
                    args.export_first_frame_as_obj,
                    args.export_all_frames_as_obj,
                    extra_data=extra_data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-abc-fname', type=str)
    parser.add_argument('--output-npz-fname', type=str)
    parser.add_argument('--target-fps', type=int, default=30)
    parser.add_argument('--source-fps', type=int, default=30)
    parser.add_argument('--export-first-frame-as-obj', action='store_true')
    parser.add_argument('--export-all-frames-as-obj', action='store_true')
    parser.add_argument('--start-frame-override', type=int, default=None)
    parser.add_argument('--end-frame-override', type=int, default=None)
    parser.add_argument('--offset', nargs=3, type=float, default=None)
    args = parser.parse_args()

    logger.info(f'{args}')

    process(args, extra_data=None)


if __name__ == '__main__':
    main()
