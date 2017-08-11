import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.ndimage.measurements import label
#from VehicleDetection.detect_util import *
from VehicleDetection.hog_subsample import find_cars, add_heat, apply_threshold, draw_labeled_bboxes
from AdvancedLaneLines.detect_lanes import *
from moviepy.editor import VideoFileClip
from collections import deque
import pickle


# Declare a global Line class object to store useful parameter to check the
# sanity between each frame of images
global line_l, line_r
line_l = Line()
line_r = Line()


class do_process(object):
    '''
    upper_layer function to input relative parameter and set
    other global parameter to precessing image
    '''
    #left_fit, right_fit = None, None
    line_l.current_fit = deque(maxlen=10)
    line_r.current_fit = deque(maxlen=10)

    def __init__(self):
        self.clf = None                 # record parameters of vehicle classifier
        self.frame = 0                  # count current frame
        self.box_que = deque(maxlen=6)  # record N recent frame for detected box
        self.n_box = []                 # record number of box in each frame
        self.lane_param = None          # record parameters of lane-detecting

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
        #draw_img = draw_labeled_bboxes(np.copy(raw_img), labels, heat, 3.3)
        #font = cv2.FONT_HERSHEY_SIMPLEX
        #cv2.putText(draw_img, "number of box:{}".format(len(box_lists)),
        #            (50,50), font, 1, (255,255,255), 2, cv2.LINE_AA)
        #cv2.putText(draw_img, "number of frame:{}".format(self.frame),
        #            (50,100), font, 1, (255,255,255), 2, cv2.LINE_AA)

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

        # =====  Lane detection =====
        # load parameter
        mtx, dist, M, Minv = self.lane_param

        # read image
        imgcp = np.copy(img)
        img_size = (imgcp.shape[1], imgcp.shape[0])
        ploty = np.linspace(0, img_size[1]-1, num=img_size[1])# to cover same y-range as image

        # undistort image before any preprocessing
        undist = cv2.undistort(imgcp, mtx, dist, None, mtx)
        undist = cv2.GaussianBlur(undist, (3,3), 0)

        # color/grdient threshold
        thresh_bin = thresh_pipeline(undist, s_thresh=(160, 250), sx_thresh=(30, 60))

        # Perspective transform to get bird-eyes view
        warped = cv2.warpPerspective(thresh_bin, M, img_size, flags=cv2.INTER_LINEAR)

        # find lane by sliding window
        lane_img, leftx, rightx = sliding_window_search(warped*255)

        leftx = np.transpose(np.nonzero(leftx))
        rightx = np.transpose(np.nonzero(rightx))

        # Measure curvature (wrap into a funtion)
        out_group = measure_curvature(leftx, rightx, y_len=img_size[1])
        left_fit, right_fit, left_fitx, right_fitx, left_curverad, right_curverad = out_group

        # check sanity
        is_sanity = 0
        do_filter = check_sanity(left_fitx, right_fitx, left_curverad, right_curverad)

        if do_filter >= 1:
            lane_img, leftx, rightx = filter_search(warped*255, line_l.current_fit[-1], line_r.current_fit[-1])
            leftx = np.transpose(np.nonzero(leftx))
            rightx = np.transpose(np.nonzero(rightx))

            # Measure curvature (wrap into a funtion)
            out_group = measure_curvature(leftx, rightx, y_len=img_size[1])
            left_fit, right_fit, left_fitx, right_fitx, left_curverad, right_curverad = out_group

            # check sanity again if not pass use previous
            is_sanity = check_sanity(left_fitx, right_fitx, left_curverad, right_curverad)

        if is_sanity < 1:
            line_l.current_fit.append(left_fit)   # use append  and measured_curvature for ave curv
            line_r.current_fit.append(right_fit)
            line_l.best_fit = np.mean(line_l.current_fit, axis=0)
            line_r.best_fit = np.mean(line_r.current_fit, axis=0)
            out_group =  get_curvature(line_l.best_fit, line_r.best_fit, y_len=img_size[1])
            left_fitx, right_fitx, left_curverad, right_curverad = out_group

            line_l.allx = left_fitx
            line_l.radius_of_curvature = left_curverad
            line_r.allx = right_fitx
            line_r.radius_of_curvature = right_curverad
        else:
            left_fit = line_l.current_fit[-1]
            left_curverad = line_l.radius_of_curvature
            left_fitx = line_l.allx
            right_fit = line_r.current_fit[-1]
            right_curverad = line_r.radius_of_curvature
            right_fitx = line_r.allx

        # Create an image to draw the lines on
        result = visualize_output(undist, warped, Minv, left_fitx, right_fitx, left_curverad, right_curverad, is_sanity, do_filter)

        # Next draw detected box for vehicles
        draw_img = draw_labeled_bboxes(np.copy(result), labels, heat, 3.3)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(draw_img, "number of box:{}".format(len(box_lists)),
                    (650,50), font, 1, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(draw_img, "number of frame:{}".format(self.frame),
                    (650,100), font, 1, (255,255,255), 2, cv2.LINE_AA)

        return draw_img



if __name__ == '__main__':
        """
        Using pipeline to detect lane lines and vehicles on video
        """
        # load camera parameters (mtx, dist)
        mtx, dist = [] ,[]
        with open('./AdvancedLaneLines/camera_cal/wide_dist_pickle.p', 'rb') as f:
            data = pickle.load(f)
            mtx, dist = data['mtx'], data['dist']
        # add matrix for perspective transform
        M, Minv = get_warped()

        # load the parameter of vehicles classifier
        param = None
        with open("./VehicleDetection/svc_pickle.p", "rb") as f:
            param = pickle.load(f)

        run = do_process()
        run.clf = param
        run.lane_param = (mtx, dist, M, Minv)

        # assign file name
        fn = "./VehicleDetection/test_video.mp4"

        project_output = "./test_out.mp4"
        clip1 = VideoFileClip(fn)
        proj_clip = clip1.fl_image(run.process_image)
        proj_clip.write_videofile(project_output, audio=False)
        plt.plot(run.n_box)
        plt.show()



