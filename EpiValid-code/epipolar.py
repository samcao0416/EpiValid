import numpy as np
from scene import Pose, Sensor, Camera
from se3_vec import SE3VecToMat


def _GetExtr(pose: Pose) -> Pose:
    return pose if pose.is_to_local else pose.inverse()


def _RT_Concate(rot, trans, shape=(3, 4)):
    ret = np.eye(*shape)
    ret[:3, :3] = rot
    ret[:3, 3] = trans
    return ret


def _CrossMat(vec):
    x, y, z = vec
    return np.array([
        [0, -z, y],
        [z, 0, -x],
        [-y, x, 0]
    ])


def GetPinholeMatrix(sensor: Sensor):
    w, h = sensor.resolution
    p = sensor.pinhole
    return np.array([
        [p.fx, 0, p.cx + (w-1)/2],
        [0, p.fy, p.cy + (h-1)/2],
        [0, 0, 1]
    ])


def ComputeFundamental(cam_left: Camera, cam_right: Camera):
    extr_right = _GetExtr(cam_right.pose)
    extr_left = _GetExtr(cam_left.pose)
    w2left = _RT_Concate(
        *SE3VecToMat(extr_left.transform.vector),
        shape=(4, 4)
    )
    right2w = _RT_Concate(
        *SE3VecToMat(extr_right.transform.inverse().vector),
        shape=(4, 4)
    )
    right2left = w2left @ right2w

    rot = right2left[:3, :3]
    trans = right2left[:3, 3]
    trans_cross = _CrossMat(trans)

    intr_right = GetPinholeMatrix(cam_right.sensor)
    intr_left = GetPinholeMatrix(cam_left.sensor)

    return np.linalg.inv(intr_left).T @ trans_cross @ rot @ np.linalg.inv(intr_right)
