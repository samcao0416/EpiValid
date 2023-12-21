from functools import partial
import argparse

import numpy as np
import cv2
import matplotlib.pyplot as plt

from scene import Sensor
from epipolar import ComputeFundamental, GetPinholeMatrix

import os



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


class Epipolar_multi:
    def __init__(self, cameras, photo_index, img_step, main_folder):
        # In json: 3, 1, 2, 4, 5, 6, 7, 8
        # 3, 4, 5, 6, 7, 8, 1, 2
        # 0, 1, 2, 3, 4, 5, 6, 7
        multi_co_list = [0, 4, 5, 6, 7, 8, 1, 2]
        # offsets = [0,0,0,0,0,0,0,0] # Debug
        # offsets = [0, 15, 25, 32, 50, 35, 30, 20]  # Group 0
        # offsets = [0, 2, 3, 5, 23, 16, 10, -1]  # Group 1
        offsets = [0, 7, 13, 17, 34, 25, 2, 8]  # Group 2
        # offsets = [0, 15, 25, 32, 50, 35, 30, 20]  # Group 3
        # offsets = [0, 15, 25, 32, 50, 35, 30, 20]  # Group 4
        # offsets = [0, 15, 25, 32, 50, 35, 30, 20]  # Group 5
        # offsets = [0, 15, 25, 32, 50, 35, 30, 20]  # Group 6
        # offsets = [0, 15, 25, 32, 50, 35, 30, 20]  # Group 7
        # offsets = [0, 15, 25, 32, 50, 35, 30, 20]  # Group 8
        # offsets = [0, 15, 25, 32, 50, 35, 30, 20]  # Group 9
        # offsets = [0, 15, 25, 32, 50, 35, 30, 20]  # Group 10
        cameras_list = []
        image_list = []
        for i in range(len(multi_co_list)):
            x = multi_co_list[i] if multi_co_list[i] < 3 else multi_co_list[i] - 1
            camera_index = x * img_step + photo_index + offsets[i]
            cameras_list.append(cameras[camera_index])
            image_path = os.path.join(main_folder, "images", cameras[camera_index].image_path)
            image_list.append(undistort(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE), cameras[camera_index].sensor))
        
        fmat_list = []
        for i in range(len(multi_co_list)):

            camera_left = cameras_list[i]
            camera_right = cameras_list[(i + 1) % len(multi_co_list)]
            _fmat = ComputeFundamental(camera_left, camera_right)
            fmat_list.append(_fmat)
        
        
        self.multi_co_list = multi_co_list
        self.cameras_list = cameras_list
        self.fmat_list = fmat_list
        self.image_list = image_list

        self._figs_list = []

        window_position = [[1495 , 30],
                           [1045 , 30],
                           [595 , 300],
                           [1045 , 710],
                           [1495 , 710],
                           [1945 , 710],
                           [2395 , 300],
                           [1945 , 30],]
        
        

        for i in range(len(multi_co_list)):

            # draw_next = partial(self.draw_line, i, (i + 1) % len(multi_co_list), 2)
            # draw_last = partial(self.draw_line, i, (i - 1) % len(multi_co_list), 1)
            draw_lines = partial(self.draw_line, i, (i + 1) % len(multi_co_list), (i - 1) % len(multi_co_list))
            _fig = ClickFigure(image_list[i], draw_lines,  window_position[i][0], window_position[i][1])
            _fig.figure.canvas.manager.set_window_title("camera_%02d: %s" %(multi_co_list[i], cameras_list[i].image_path))
            self._figs_list.append(_fig)


        # i = 0
        # draw_next = partial(self.draw_line, i, (i + 1) % len(multi_co_list), 2)
        # draw_last = partial(self.draw_line, i, (i - 1) % len(multi_co_list), 1)
        # _fig = ClickFigure(image_list[i], draw_next, draw_last,  window_position[i][0], window_position[i][1])
        # _fig.figure.canvas.manager.set_window_title("camera_%02d" %(multi_co_list[i]))
        # self._figs_list.append(_fig)

        # i = 1
        # draw_next = partial(self.draw_line, i, (i + 1) % len(multi_co_list), 2)
        # draw_last = partial(self.draw_line, i, (i - 1) % len(multi_co_list), 1)
        # _fig_2 = ClickFigure(image_list[i], draw_next, draw_last,  window_position[i][0], window_position[i][1])
        # _fig_2.figure.canvas.manager.set_window_title("camera_%02d" %(multi_co_list[i]))
        # self._figs_list.append(_fig_2)

    def draw_line(self, curr_index, next_index, last_index, x, y):
        print('PICK', curr_index, next_index, last_index, x, y)
        
        pts = np.array([(x, y)])
        # lines = cv2.computeCorrespondEpilines(pts, cv_index, self.fmat_list[int(curr_index)])
        # line = lines[0][0]

        lines_next = cv2.computeCorrespondEpilines(pts, 2, self.fmat_list[int(curr_index)])
        line_next = lines_next[0][0]
        lines_last = cv2.computeCorrespondEpilines(pts, 1, self.fmat_list[int(last_index)])
        line_last = lines_last[0][0]

        print('line_next', line_next)
        print('line_last', line_last)
        curr = self._figs_list[int(curr_index)]
        curr.axis.scatter(x, y, marker='x')

        nextt = self._figs_list[int(next_index)]
        nextt.drawline(line_next)
        lastt = self._figs_list[int(last_index)]
        lastt.drawline(line_last)
        # target = self._figs_list[int(target_index)]
        # target.drawline(line)

        curr.update()
        nextt.update()
        lastt.update()
        # target.update()
        # compute epipolar line from index and XY
        # CAVEAT
        # index???
        # cvindex = 1 + int(not index)
        

class ClickFigure:
    def __init__(self, image: np.ndarray, pick_handler, *args):
        self._img = image
        self._width = image.shape[1]
        self._height = image.shape[0]
        self._fig = plt.figure()
        self._ax = self._fig.add_subplot(1, 1, 1)

        manager = plt.get_current_fig_manager()
        manager.window.setGeometry(args[0], args[1], 450, 650)

        self._ax.imshow(self._img, cmap='gray', vmin=0, vmax=255)
        self._pick= pick_handler

        #self._fig.canvas.mpl_connect('motion_notify_event', self._onmove) #debug
       # self._fig.canvas.mpl_connect('key_press_event', self.on_key_press)
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

    # def _onmove(self):
    #     print("moving")

    # def on_key_press(self,event):
    #     print(f"Key pressed: {event.key}")



def _parse_args():
    parser = argparse.ArgumentParser(
        description='Interactive epipolar line drawer')
    parser.add_argument('cameras_json', help='path to cameras json')
    parser.add_argument('camera_index', type=int, help='camera index in the group')
    parser.add_argument('img_step', type=int, help='right camera index')

    return parser.parse_args()


def _load_nerf_cameras(path_json: str):
    from parse_whkrt import parse_nerf
    return parse_nerf(path_json)



if __name__ == "__main__":
    args = _parse_args()

    cameras = _load_nerf_cameras(args.cameras_json)

    epi = Epipolar_multi(cameras, args.camera_index, args.img_step, os.path.dirname(args.cameras_json))
    
    plt.show()



    """
    每一个image都有一个click figure
    click figure里之前是draw left/ right, 现在应该是一个click figure里有一个draw last, 一个draw next
    _fmat中left是0, right是1
    之前的draw left, 意思是在左图上画点, 右图上画线: 输入index: 0, cvindex: 2, curr -> scatter, other -> line, 对应draw_next
    现在的draw next: 应该是输入current index, cvindex, next_index. Current_index: list中的index, 用来选择image_list和fmat_list, cv index: hardcode 2, next_index: list中的index
    现在的drwa last: current index  cv index: hardcode 1

    """