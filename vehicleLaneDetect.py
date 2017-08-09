import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.ndimage.measurements import label
#from VehicleDetection.detect_util import *
from VehicleDetection.hog_subsample import find_cars, add_heat, apply_threshold, draw_labeled_bboxes
from moviepy.editor import VideoFileClip
from collections import deque
import pickle


class do_process(object):
    def __init__(self):
        self.clf = None
        self.frame = 0                  # count current frame
        self.box_que = deque(maxlen=6)  # record N recent frame for detected box
        self.n_box = []                 # record number of box in each frame

    def process_image(self, img):
        """
        main pipeline to process each frame of video
        :param img:
        :param clf:
        :return:
        """
        self.frame += 1 # counting number of frame
        #if self.frame <= 487:
        #    return img
        # read parameter from clf
        svc            = self.clf["svc"]
        X_scaler       = self.clf["scaler"]
        orient         = self.clf["orient"]
        pix_per_cell   = self.clf["pix_per_cell"]
        cell_per_block = self.clf["cell_per_block"]
        spatial_size   = self.clf["spatial_size"]
        hist_bins      = self.clf["hist_bins"]
        color_space    = self.clf["color_space"]

        # Other parameter not in pickle
        hog_channel    = "ALL"      # Can be 0, 1, 2, or "ALL"
        spatial_feat   = True       # Spatial features on or off
        hist_feat      = True       # Histogram features on or off
        hog_feat       = True       # HOG features on or off

        raw_img = np.copy(img)
        ystart = 384 #480
        ystop = 650  #672
        scale_s = 1.30
        scale_e = 1.50
        steps = 3

        # May Convert to the wrong channel
        box_lists = []  # a list to record different subsample scale
        for scale in np.linspace(scale_s, scale_e, steps):
            _img, box_list = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient,
                                       pix_per_cell, cell_per_block, spatial_size, hist_bins,
                                       color_space=color_space)
            box_lists.extend(box_list)
        if len(box_lists) == 0:
            if len(self.box_que) > 0:
                self.box_que.popleft()
        else:
            self.box_que.append(box_lists)
        self.n_box.append(len(box_lists))

        #if len(box_lists) < 1:
        #    return img

        #out_img = np.copy(img)
        #for b in box_lists:
        #    cv2.rectangle(out_img, b[0], b[1], (0, 0, 255), 6)

        # build heat map and remove false positive
        heat = np.zeros_like(raw_img[:,:,0]).astype(np.float)

        # Add heat to each box in box list
        for bl in self.box_que:
            heat = add_heat(heat, bl)
        #heat = heat / steps

        # Apply threshold to help remove false positives
        heat = apply_threshold(heat, 0)
        #if len(self.box_que) <=3:
        #    heat = apply_threshold(heat, 0)
        #else:
        #    heat = apply_threshold(heat, 0)

        # Visualize the heatmap when displaying
        heatmap = np.clip(heat, 0, 255)

        # Find final boxes from heatmap using label function
        struct = np.ones((3, 3))
        labels = label(heatmap,structure=struct)
        draw_img = draw_labeled_bboxes(np.copy(raw_img), labels, heat, 3.3)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(draw_img, "number of box:{}".format(len(box_lists)),
                    (50,50), font, 1, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(draw_img, "number of frame:{}".format(self.frame),
                    (50,100), font, 1, (255,255,255), 2, cv2.LINE_AA)

        #if len(box_lists) >= 1:
        #    fig = plt.figure()
        #    plt.subplot(121)
        #    plt.imshow(draw_img)
        #    plt.title('Car Positions')
        #    plt.subplot(122)
        #    plt.imshow(heatmap, cmap='hot')
        #    plt.title('Heat Map')
        #    fig.tight_layout()
        #    plt.show()

        return draw_img



if __name__ == '__main__':
        """
        Using pipeline to detect vehicles on video
        """
        # load the parameter of vehicles classifier
        param = None
        with open("./VehicleDetection/svc_pickle.p", "rb") as f:
            param = pickle.load(f)

        run = do_process()
        run.clf = param

        # assign file name
        fn = "./VehicleDetection/test_video.mp4"

        project_output = "./test_out.mp4"
        clip1 = VideoFileClip(fn)
        proj_clip = clip1.fl_image(run.process_image)
        proj_clip.write_videofile(project_output, audio=False)
        plt.plot(run.n_box)
        plt.show()



