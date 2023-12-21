from typing import Optional, List, Hashable, Tuple, NamedTuple

import numpy as np
from se3_vec import SE3VecInv


class Transform:
    def __init__(self, rt_vec: np.ndarray = np.zeros(6)):
        self._ary = np.empty(6)
        self.vector = rt_vec
        self._rot = self._ary[:3]
        self._trans = self._ary[3:]

    @property
    def rotation(self):
        return self._rot

    @property
    def translation(self):
        return self._trans

    @property
    def vector(self):
        return self._ary

    @vector.setter
    def vector(self, x: np.ndarray):
        assert x.shape == (6, )
        self._ary[:] = x[:]

    def inverse(self):
        return Transform(SE3VecInv(self._ary))


class Pose:
    def __init__(self, transform: Optional[Transform] = None,
                 is_to_local: bool = True):
        self.transform: Transform = Transform() if transform is None else transform
        self.is_to_local = is_to_local

    def inverse(self):
        return self.__class__(self.transform.inverse(), (not self.is_to_local))


class Pinhole:
    def __init__(self, fx: float, fy: float, cx: float, cy: float):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

    def matrix(self):
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ])

    def __repr__(self) -> str:
        return f'Pinhole(fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy})'


class Distortion:
    def __init__(self, k1: float = 0.0, k2: float = 0.0,
                 p1: float = 0.0, p2: float = 0.0, k3: float = 0.0):
        self.k1 = k1
        self.k2 = k2
        self.p1 = p1
        self.p2 = p2
        self.k3 = k3


class Resolution(NamedTuple):
    width: int
    height: int


class Sensor:
    def __init__(self, resolution: Optional[Resolution] = None,
                 pinhole: Optional[Pinhole] = None, distortion: Optional[Distortion] = None):
        self._resolution: Optional[Resolution] = resolution
        self.pinhole: Optional[Pinhole] = pinhole
        self.distortion: Optional[Distortion] = distortion

    @property
    def resolution(self) -> Optional[Resolution]:
        return self._resolution

    @resolution.setter
    def resolution(self, resolution: Resolution):
        if self._resolution is None:
            self._resolution = resolution
        else:
            if resolution != self._resolution:
                raise ValueError('Resolution mismatch')


class Camera:
    def __init__(self, camera_id: Hashable, sensor: Optional[Sensor] = None,
                 pose: Optional[Pose] = None, image_name = None):
        self.camera_id = camera_id
        self.sensor: Sensor = Sensor() if sensor is None else sensor
        self.pose: Pose = Pose() if pose is None else pose
        self.image_name = image_name

    def __hash__(self):
        return hash(self.camera_id)
