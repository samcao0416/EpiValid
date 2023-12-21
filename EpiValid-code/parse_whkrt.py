import json
import os

import numpy as np
from scipy.spatial.transform.rotation import Rotation

from scene import *


def parse(path_js):
    with open(path_js, 'r') as f:
        js = json.load(f)

    Ks = np.loadtxt(js['intrs'])
    num_k = len(Ks)
    assert Ks.shape == (num_k, 9)
    c2ws = np.loadtxt(js['c2ws'])
    assert c2ws.shape == (num_k, 12)
    ress = np.loadtxt(js['res'], dtype=int)
    assert ress.shape == (num_k, 2)

    if 'dists' in js.keys():
        dists = np.loadtxt(js['dists'])
        num_dist, order_dist = dists.shape
        assert (num_dist == num_k) and (order_dist <= 5)
    else:
        dists = np.zeros((num_k, 0))

    cams = dict()
    for i, (K, c2w, wh, dd) in enumerate(zip(Ks, c2ws, ress, dists)):
        width, height = wh
        fx, fy, cx, cy = K[[0, 4, 2, 5]]
        cx -= (width - 1) / 2
        cy -= (height - 1) / 2
        pin = Pinhole(fx, fy, cx, cy)
        res = Resolution(*wh)
        dist = Distortion(*dd)
        ssr = Sensor(res, pin, dist)

        c2w_mat = c2w.reshape(3, 4)
        rvec = Rotation.from_matrix(c2w_mat[:3, :3]).as_rotvec()
        rt = np.concatenate([rvec, c2w_mat[:3, 3]])
        tsfm = Transform(rt)
        pose = Pose(tsfm, is_to_local=False)

        cam = Camera(i, ssr, pose)
        cams[i] = cam
    return cams

def parse_nerf(path_js):
    with open(path_js, 'r') as f:
        js = json.load(f)

    width = js["w"]
    height = js["h"]
    fx = js["fl_x"]
    fy = js["fl_y"]
    cx = js["cx"]
    cy = js["cy"]

    try:
        K1 = js["K1"]
        K2 = js["K2"]
        K3 = js["K3"]
        P1 = js["P1"]
        P2 = js["P2"]
    except:
        K1 = 0
        K2 = 0
        K3 = 0
        P1 = 0
        P2 = 0

    cx -= (width - 1) / 2
    cy -= (height - 1) / 2

    pin = Pinhole(fx, fy, cx, cy)

    
    res = Resolution(width, height)
    
    dist = Distortion(K1, K2, P1, P2, K3)

    cams = dict()

    for i, frame in enumerate(js["frames"]):
        c2w = np.array(frame["transform_matrix"])
        image_name = os.path.basename(frame["file_path"])[:-4]
        c2w[:, 1:3] = -c2w[:, 1:3] # reverse Y and Z axis
        #print(c2w)
        rvec = Rotation.from_matrix(c2w[:3, :3]).as_rotvec()
        rt = np.concatenate([rvec, c2w[:3, 3]])
        tsfm = Transform(rt)
        pose = Pose(tsfm, is_to_local=False)

        cam = Camera(i, Sensor(res, pin, dist), pose, image_name)
        cams[i] = cam

    return cams
