from __future__ import annotations

from loguru import logger


def convert_framerate(poses, trans, fps, target_fps):
    fps = int(round(float(fps)))
    if fps % int(target_fps) == 0:
        every_n_frames = int(fps) // int(target_fps)
        poses = poses[::every_n_frames]
        trans = trans[::every_n_frames]
    elif target_fps == 30:
        if fps == 100:
            poses = poses[::3]
            trans = trans[::3]
        elif fps == 59:
            poses = poses[::2]
            trans = trans[::2]
        elif fps == 200:
            poses = poses[::7]
            trans = trans[::7]
        elif fps == 250:
            poses = poses[::8]
            trans = trans[::8]
        else:
            logger.error('Unknown source fps:%d' % fps)
            exit(-1)
    else:
        logger.error('Unknown source fps:%d' % fps)
        exit(-1)
    return poses, trans
