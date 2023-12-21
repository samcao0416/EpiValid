from functools import partial
import argparse

import numpy as np
import cv2
import matplotlib.pyplot as plt

from scene import Sensor
from epipolar import ComputeFundamental, GetPinholeMatrix


def undistort(image: np.ndarray, sensor: Sensor):
    if sensor.distortion is None:
        # no distortion, no need to undistort
        return image
    intr_mat = GetPinholeMatrix(sensor)
    dist = np.array([
        sensor.distortion.k1,
        sensor.distortion.k2,
        sensor.distortion.p1,
        sensor.distortion.p2,
        sensor.distortion.k3
    ])
    ret = np.ones_like(image) * 255
    return cv2.undistort(image, intr_mat, dist, dst=ret)


class Epipolar:
    def __init__(self, camera_left, camera_right, image_left, image_right):
        # construct two ClickFigures,
        # that passes click event back to here
        self._cam_left = camera_left
        self._cam_right = camera_right
        self._fmat = ComputeFundamental(self._cam_left, self._cam_right)
        self._img_left = undistort(image_left, self._cam_left.sensor)
        self._img_right = undistort(image_right, self._cam_right.sensor)
        # CAVEAT
        # image index???
        draw_left = partial(self.draw_line, 0)
        self._fig_left = ClickFigure(self._img_left, draw_left, 500)
        self._fig_left.figure.canvas.manager.set_window_title(camera_left.image_name)
        draw_right = partial(self.draw_line, 1)
        self._fig_right = ClickFigure(self._img_right, draw_right, 1700)
        self._fig_right.figure.canvas.manager.set_window_title(camera_right.image_name)
        self._figs = [self._fig_left, self._fig_right]

    def draw_line(self, index, x, y):
        print('PICK', index, x, y)
        # compute epipolar line from index and XY
        # CAVEAT
        # index???
        cvindex = 1 + int(not index)
        pts = np.array([(x, y)])
        lines = cv2.computeCorrespondEpilines(pts, cvindex, self._fmat)
        line = lines[0][0]
        print('line', line)
        # scatter on current image
        curr = self._figs[int(index)]
        curr.axis.scatter(x, y, marker='x')
        # draw line on THE OTHER image
        other = self._figs[int(not index)]
        other.drawline(line)

        curr.update()
        other.update()


class ClickFigure:
    def __init__(self, image: np.ndarray, pick_handler, *args):
        self._img = image
        self._width = image.shape[1]
        self._height = image.shape[0]
        self._fig = plt.figure()
        self._ax = self._fig.add_subplot(1, 1, 1)

        manager = plt.get_current_fig_manager()
        manager.window.setGeometry(args[0], 300, 1200, 800)

        self._ax.imshow(self._img, cmap='gray', vmin=0, vmax=255)
        self._pick = pick_handler
        self._fig.canvas.mpl_connect('button_press_event', self._onclick)

    def clear(self):
        self._fig.clear()
        self.update()

    def update(self):
        self._fig.canvas.draw()

    def drawline(self, coeff):
        a, b, c = coeff

        def y(x):
            return -(a * x + c) / b
        self._ax.plot([0, self._width], [y(0), y(self._width)])
        self._ax.set_xlim(0, self._width)
        self._ax.set_ylim(self._height, 0)

    @property
    def axis(self):
        return self._ax

    @property
    def figure(self):
        return self._fig

    def _onclick(self, event):
        # callback on double click only
        if event.dblclick != 0:
            x = event.xdata
            y = event.ydata
            # print(id(self), x, y)
            self._pick(x, y)


def _parse_args():
    parser = argparse.ArgumentParser(
        description='Interactive epipolar line drawer')
    parser.add_argument('mode', choices=['original', 'nerf', 'titan'], help = 'orginal: Yanshun data structure; nerf: nerf json data structure')
    parser.add_argument('cameras_json', help='path to cameras json')
    parser.add_argument('left_index', type=int, help='left camera index')
    parser.add_argument('right_index', type=int, help='right camera index')
    parser.add_argument('left_img', help='path to left image')
    parser.add_argument('right_img', help='path to right image')

    return parser.parse_args()


def _load_cameras(path_json: str):
    from parse_whkrt import parse
    return parse(path_json)

def _load_nerf_cameras(path_json: str):
    from parse_whkrt import parse_nerf
    return parse_nerf(path_json)


if __name__ == "__main__":
    args = _parse_args()
    if args.mode == 'original':
        cameras = _load_cameras(args.cameras_json)
        
    elif args.mode == 'nerf':
        cameras = _load_nerf_cameras(args.cameras_json)
        
    img_left = cv2.imread(args.left_img, cv2.IMREAD_GRAYSCALE)
    img_right = cv2.imread(args.right_img, cv2.IMREAD_GRAYSCALE)
    epi = Epipolar(cameras[args.left_index],
                cameras[args.right_index], img_left, img_right)
    plt.show()