import numpy as np
import cv2
import copy
from collections import deque
import argparse


#blue
# _LOWER_COLOR = np.array([80, 50, 50])
# _UPPER_COLOR = np.array([110, 255, 255])

#red
_LOWER_COLOR = np.array([0, 50, 50])
_UPPER_COLOR = np.array([10, 255, 255])


class ParticleFilter:

    def __init__(self, frame_size, particle_N):
        self.SAMPLEMAX = particle_N
        self.height = frame_size[0]
        self.width = frame_size[1]

    def initialize(self):
        self.Y = np.random.random(self.SAMPLEMAX) * self.height
        self.X = np.random.random(self.SAMPLEMAX) * self.width

    # Need adjustment for tracking object velocity
    def modeling(self):
        self.Y += np.random.random(self.SAMPLEMAX) * 200 - 100 # 2:1
        self.X += np.random.random(self.SAMPLEMAX) * 200 - 100

    def normalize(self, weight):
        return weight / np.sum(weight)

    def resampling(self, weight):
        index = np.arange(self.SAMPLEMAX)
        sample = []

        # choice by weight
        for i in range(self.SAMPLEMAX):
            idx = np.random.choice(index, p=weight)
            sample.append(idx)
        return sample

    def calcLikelihood(self, image):
        # white space tracking
        mean, std = 250.0, 10.0
        intensity = []

        for i in range(self.SAMPLEMAX):
            y, x = self.Y[i], self.X[i]
            if y >= 0 and y < self.height and x >= 0 and x < self.width:
                intensity.append(image[int(y),int(x)])
            else:
                intensity.append(-1)

        # normal distribution
        weights = 1.0 / np.sqrt(2 * np.pi * std) * np.exp(-(np.array(intensity) - mean)**2 /(2 * std**2))
        weights[intensity == -1] = 0
        weights = self.normalize(weights)
        return weights

    def filtering(self, image):
        self.modeling()
        weights = self.calcLikelihood(image)
        index = self.resampling(weights)
        self.Y = self.Y[index]
        self.X = self.X[index]

        # return COG
        return np.sum(self.Y) / float(len(self.Y)), np.sum(self.X) / float(len(self.X))


def tracking():

    # set some arguments
    parser = argparse.ArgumentParser(description='yolov2_darknet_predict for video')
    parser.add_argument('--video_file', '-v', type=str, default=False,help='path to video')
    parser.add_argument('--camera_ID', '-c', type=int, default=0,help='camera ID')
    # parser.add_argument('--save_name', '-s', type=str, default=False,help='camera ID')
    args = parser.parse_args()

    # load a video file or connect to the web camera
    if not args.video_file == False:
        cap = cv2.VideoCapture(args.video_file)
    else:
        cap = cv2.VideoCapture(args.camera_ID)

    ret, frame = cap.read()

    particle_N = 1000
    pf = ParticleFilter(frame.shape, particle_N)
    pf.initialize()

    max_points_N = 30
    trajectory_points = deque(maxlen=max_points_N)

    cv2.namedWindow("result")

    while True:

        ret, frame = cap.read()
        # cv2.imshow("frame", frame)
        result_frame = copy.deepcopy(frame)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Threshold the HSV image to get only a color
        mask = cv2.inRange(hsv, _LOWER_COLOR, _UPPER_COLOR)

        # Start Tracking
        y, x = pf.filtering(mask)

        # origin is upper left
        frame_size = frame.shape
        p_range_x = np.max(pf.X)-np.min(pf.X)
        p_range_y = np.max(pf.Y)-np.min(pf.Y)
        # print "position_x_rate"
        # print x/frame_size[1]
        # print "position_y_rate"
        # print y/frame_size[0]

        for i in range(pf.SAMPLEMAX):
            cv2.circle(result_frame, (int(pf.X[i]), int(pf.Y[i])), 2, (0, 255, 0), -1)

        if p_range_x < 300 and p_range_y < 300:

            center = (int(x), int(y))
            cv2.circle(result_frame, center, 10, (255, 0, 0), -1)
            trajectory_points.appendleft(center)

            for i in range(1, len(trajectory_points)):
                if trajectory_points[i - 1] is None or trajectory_points[i] is None:
                    continue
                cv2.line(result_frame, trajectory_points[i-1], trajectory_points[i], (0, 255, 255), thickness=3)
        else:
            trajectory_points = deque(maxlen=max_points_N)

        cv2.imshow("result", result_frame)

        if cv2.waitKey(20) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    tracking()
