from typing import Tuple

import numpy as np
from scipy.spatial.transform.rotation import Rotation


def AngleAxis2Mat(rotvec):
    vec = rotvec.reshape(3)
    angle = np.linalg.norm(vec)
    axis = rotvec / angle if angle != 0 else np.zeros(3)
    cos = np.cos(angle)
    ret = np.eye(3, 3, 0, rotvec.dtype) * cos
    ret += axis.reshape(3, 1) @ axis.reshape(1, 3) * (1 - cos)
    rx, ry, rz = axis
    ret += np.array([
        [0, -rz, ry],
        [rz, 0, -rx],
        [-ry, rx, 0]
    ]) * np.sin(angle)
    return ret


def SE3VecInv(vec: np.ndarray):
    assert vec.shape == (6,)
    # rotation vector's inverse is its negative
    ret = np.zeros_like(vec)
    ret[:3] = -vec[:3]
    t_inv = -Rotation.from_rotvec(ret[:3]).apply(vec[3:])
    ret[3:] = t_inv
    return ret


def SE3VecToMat(vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    assert vec.shape == (6,)
    # rot = Rotation.from_rotvec(vec[:3]).as_matrix()
    # use numba version to speed up bundle adjustment
    rot = AngleAxis2Mat(vec[:3])
    trans = vec[3:]
    return rot, trans
