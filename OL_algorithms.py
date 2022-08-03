import cv2
import numpy as np
import threading
import time
import math


class WebcamVideoAlgorithms:

    # in general, morphological operations are not performed, and noise removal in these methods does not have such an effect as they slow down the program execution

    def __init__(self, video_const):
        self.VC = video_const
        self.reset_hough_radii()
        self.reset_refined_objects_list()  # only used with refined_objects_list
        self.scene_change_significancy = False
        self.detection_mode_change = False
        self.new_time = time.time()
        self.show_new_detection_results = True


    def denoise(self, unprocessed_frame):
        # removes noise very well, but does not find application because it slows down the program execution too much
        frame = cv2.fastNlMeansDenoising(unprocessed_frame)

        return frame


    def check_scene_change_significancy(self, last_unprocessed_frame, unprocessed_frame):
        last_gray_frame = cv2.cvtColor(last_unprocessed_frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.cvtColor(unprocessed_frame, cv2.COLOR_BGR2GRAY)
        absdiff_frame = cv2.absdiff(last_gray_frame, gray_frame)
        (ret, thresh_frame) = cv2.threshold(absdiff_frame, self.VC.MIN_BINARY_THRESH_VALUE_FOR_ABSDIFF, self.VC.MAX_BINARY_THRESH_VALUE_FOR_ABSDIFF, cv2.THRESH_BINARY)

        num_of_non_zero_pixels = cv2.countNonZero(thresh_frame)
        num_of_all_pixels = thresh_frame.shape[1] * thresh_frame.shape[0]
        consecutive_frames_difference = num_of_non_zero_pixels / num_of_all_pixels

        if consecutive_frames_difference >= self.VC.MAX_CONSECUTIVE_FRAMES_DIFFERENCE:
            self.scene_change_significancy = True
        else:
            self.scene_change_significancy = False


    def is_scene_change_significant(self): return self.scene_change_significancy


    def set_detection_mode_change(self, detection_mode_change):
        self.detection_mode_change = detection_mode_change


    def is_detection_mode_changed(self): return self.detection_mode_change


    def update_detection_results_showing(self):
        self.new_time = time.time()
        self.show_new_detection_results = True


    def check_new_detection_results_showing(self):
        self.old_time = self.new_time
        if round(float(time.time() - self.old_time), 1) > self.VC.MIN_SHOW_RESULTS_INTERVAL:
            self.show_new_detection_results = True
        else:
            self.show_new_detection_results = False


    def are_new_detection_results_showing(self): return self.show_new_detection_results


    def match_gray_templates(self, unprocessed_frame, gray_temp_images_list):
        gray_frame = cv2.cvtColor(unprocessed_frame, cv2.COLOR_BGR2GRAY)
        (frame, gray_frame) = self.do_common_template_matching_functions(unprocessed_frame, gray_frame, gray_temp_images_list, self.VC.TEMPLATE_MATCHING_GRAY_THRESHS_LIST)

        return (frame, gray_frame)


    def match_binary_templates(self, unprocessed_frame, binary_temp_images_list):
        gray_frame = cv2.cvtColor(unprocessed_frame, cv2.COLOR_BGR2GRAY)
        binary_frame = cv2.adaptiveThreshold(gray_frame, self.VC.MAX_ADAPTIVE_THRESH_VALUE, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, self.VC.ADAPTIVE_THRESH_WINDOW_SIZE, \
            self.VC.ADAPTIVE_THRESH_MEAN_OFFSET)
        (frame, binary_frame) = self.do_common_template_matching_functions(unprocessed_frame, binary_frame, binary_temp_images_list, self.VC.TEMPLATE_MATCHING_BINARY_THRESHS_LIST)

        return (frame, binary_frame)


    def do_common_template_matching_functions(self, unprocessed_frame, gray_or_binary_frame, gray_or_binary_temp_images_list, template_matching_threshs_list):
        # self.temp_center_points_and_sizes_list contains elements of the form (x_center, y_center, template_width, template_height)
        # self.temp_center_points_and_sizes_list contains elements of the form max_matching_value
        frame = unprocessed_frame
        self.temp_center_points_and_sizes_list = []
        self.temp_matching_thresholds_list = []

        if self.VC.THREADED_TEMPLATE_MATCHING:
            threads_list = []
            for gray_or_binary_temp_image in gray_or_binary_temp_images_list:
                thread = threading.Thread(target=self.process_template_matching, args=(gray_or_binary_frame, gray_or_binary_temp_image, template_matching_threshs_list))
                thread.daemon = True
                thread.start()
                threads_list.append(thread)
            for thread in threads_list:
                thread.join()
        else:
            for gray_or_binary_temp_image in gray_or_binary_temp_images_list:
                self.process_template_matching(gray_or_binary_frame, gray_or_binary_temp_image, template_matching_threshs_list)

        if self.VC.USE_REFINED_OBJECTS_LIST:
            # this piece of code is only used with refined_objects_list
            self.process_refined_objects_list(unprocessed_frame, self.temp_center_points_and_sizes_list)
            for ready_refined_object in self.ready_refined_objects_list:
                if not self.is_scene_change_significant():
                    cv2.rectangle(frame, (ready_refined_object.get_upper_left_x(), ready_refined_object.get_upper_left_y()), (ready_refined_object.get_lower_right_x(), \
                        ready_refined_object.get_lower_right_y()), self.VC.COLOR_BLUE, self.VC.LINE_THICKNESS)
                    cv2.drawMarker(frame, (ready_refined_object.get_x(), ready_refined_object.get_y()), self.VC.COLOR_BLUE, cv2.MARKER_CROSS, self.VC.MARKER_SIZE, self.VC.LINE_THICKNESS)
                    cv2.putText(frame, "{}p".format(ready_refined_object.get_height()), (ready_refined_object.get_x(), ready_refined_object.get_y()), cv2.FONT_HERSHEY_SIMPLEX, \
                        self.VC.TEXT_SIZE_SMALL, self.VC.COLOR_BLUE, self.VC.LINE_THICKNESS)
            if not self.is_scene_change_significant() and self.VC.PROCESS_FROM_CAM:
                cv2.putText(frame, "Detected: {}".format(len(self.ready_refined_objects_list)), (self.VC.DETECTED_OBJECTS_POS_X, self.VC.DETECTED_OBJECTS_POS_Y), cv2.FONT_HERSHEY_SIMPLEX, \
                    self.VC.TEXT_SIZE, self.VC.COLOR_BLUE, self.VC.LINE_THICKNESS)
        else:
            if self.are_new_detection_results_showing():
                self.old_temp_center_points_and_sizes_list = self.temp_center_points_and_sizes_list
            else:
                try: self.temp_center_points_and_sizes_list = self.old_temp_center_points_and_sizes_list
                except: self.old_temp_center_points_and_sizes_list = self.temp_center_points_and_sizes_list
            for temp_center_point_and_size in self.temp_center_points_and_sizes_list:
                temp_center_point_and_size_index = self.temp_center_points_and_sizes_list.index(temp_center_point_and_size)
                temp_matching_threshold_index = temp_center_point_and_size_index
                matched_top_left_point = (temp_center_point_and_size[0] - int(temp_center_point_and_size[2]/2), temp_center_point_and_size[1] - int(temp_center_point_and_size[2]/2))
                matched_bottom_right_point = (temp_center_point_and_size[0] + int(temp_center_point_and_size[3]/2), temp_center_point_and_size[1] + int(temp_center_point_and_size[3]/2))
                if not self.is_scene_change_significant():
                    cv2.rectangle(frame, matched_top_left_point, matched_bottom_right_point, self.VC.COLOR_BLUE, self.VC.LINE_THICKNESS)
                    cv2.drawMarker(frame, (temp_center_point_and_size[0], temp_center_point_and_size[1]), self.VC.COLOR_BLUE, cv2.MARKER_CROSS, self.VC.MARKER_SIZE, self.VC.LINE_THICKNESS)
                    cv2.putText(frame, "{}p".format(self.temp_center_points_and_sizes_list[temp_center_point_and_size_index][3]), (temp_center_point_and_size[0], \
                        temp_center_point_and_size[1]), cv2.FONT_HERSHEY_SIMPLEX, self.VC.TEXT_SIZE_SMALL, self.VC.COLOR_BLUE, self.VC.LINE_THICKNESS)
            if not self.is_scene_change_significant() and self.VC.PROCESS_FROM_CAM:
                cv2.putText(frame, "Detected: {}".format(len(self.temp_center_points_and_sizes_list)), (self.VC.DETECTED_OBJECTS_POS_X, self.VC.DETECTED_OBJECTS_POS_Y), \
                    cv2.FONT_HERSHEY_SIMPLEX, self.VC.TEXT_SIZE, self.VC.COLOR_BLUE, self.VC.LINE_THICKNESS)

        return (frame, gray_or_binary_frame)


    def process_template_matching(self, gray_or_binary_frame, gray_or_binary_temp_image, template_matching_threshs_list):
        # with threading, the problem of using the same data occurs when more than 10 or 15 objects are detected, so exception handling should be used
        # for close objects to be detected, their slight overlap must be allowed, which is why the allowed distances from the centers are set to 1/3
        processed_gray_or_binary_frame = cv2.matchTemplate(gray_or_binary_frame, gray_or_binary_temp_image, cv2.TM_CCOEFF_NORMED)
        temp_width = gray_or_binary_temp_image.shape[1]
        temp_height = gray_or_binary_temp_image.shape[0]

        for temp_matching_threshold in template_matching_threshs_list:
            matched_upper_left_points_list_unformatted = np.where(processed_gray_or_binary_frame >= temp_matching_threshold)
            for matched_upper_left_point in zip(*matched_upper_left_points_list_unformatted[::-1]):
                matched_center_point = (matched_upper_left_point[0] + int(temp_width/2), matched_upper_left_point[1] + int(temp_height/2))
                if self.temp_center_points_and_sizes_list:
                    matched_center_point_greater_overlapping = False
                    try:
                        for temp_center_point_and_size in self.temp_center_points_and_sizes_list:
                            temp_center_point_and_size_index = self.temp_center_points_and_sizes_list.index(temp_center_point_and_size)
                            temp_matching_threshold_index = temp_center_point_and_size_index
                            if not (matched_center_point[0] + int(temp_width/3) <= temp_center_point_and_size[0] - int(temp_center_point_and_size[2]/3) or matched_center_point[0] - \
                                int(temp_width/3) >= temp_center_point_and_size[0] + int(temp_center_point_and_size[2]/3) or \
                                matched_center_point[1] + int(temp_height/3) <= temp_center_point_and_size[1] - int(temp_center_point_and_size[3]/3) or matched_center_point[1] - \
                                    int(temp_height/3) >= temp_center_point_and_size[1] + int(temp_center_point_and_size[3]/3)):
                                matched_center_point_greater_overlapping = True
                                if temp_matching_threshold >= self.temp_matching_thresholds_list[temp_matching_threshold_index]:
                                    self.temp_center_points_and_sizes_list.pop(temp_center_point_and_size_index)
                                    self.temp_matching_thresholds_list.pop(temp_matching_threshold_index)
                                    self.temp_center_points_and_sizes_list.append((matched_center_point[0], matched_center_point[1], temp_width, temp_height))
                                    self.temp_matching_thresholds_list.append(temp_matching_threshold)
                    except:
                        pass
                    if matched_center_point_greater_overlapping == False:
                        self.temp_center_points_and_sizes_list.append((matched_center_point[0], matched_center_point[1], temp_width, temp_height))
                        self.temp_matching_thresholds_list.append(temp_matching_threshold)
                else:
                    self.temp_center_points_and_sizes_list.append((matched_center_point[0], matched_center_point[1], temp_width, temp_height))
                    self.temp_matching_thresholds_list.append(temp_matching_threshold)

        if len(self.temp_center_points_and_sizes_list) >= self.VC.MAX_DETECTED_OBJECTS:
            self.temp_center_points_and_sizes_list = self.temp_center_points_and_sizes_list[0:self.VC.MAX_DETECTED_OBJECTS]
            self.temp_matching_thresholds_list = self.temp_matching_thresholds_list[0:self.VC.MAX_DETECTED_OBJECTS]


    def auto_detect_canny_edges(self, unprocessed_frame):
        median_frame = np.median(unprocessed_frame)
        lower_canny_thresh_value = int(max(self.VC.MIN_CANNY_THRESH_VALUE, (1 - self.VC.CANNY_SIGMA_VALUE) * median_frame))
        upper_canny_thresh_value = int(min(self.VC.MAX_CANNY_THRESH_VALUE, (1 + self.VC.CANNY_SIGMA_VALUE) * median_frame))
        frame = cv2.Canny(unprocessed_frame, lower_canny_thresh_value, upper_canny_thresh_value)

        return frame


    def detect_hough_circles_with_canny_edges(self, unprocessed_frame):
        # funkcija cv2.pyrMeanShiftFiltering(unprocessed_frame, self.VC.PYR_MEAN_SHIFT_SP, self.VC.PYR_MEAN_SHIFT_SR) uprosječuje intenzitete sličnih područja i usporava izvođenje programa do tri puta, bez vidljivih poboljšanja pronalaženja objekata
        gray_frame = cv2.cvtColor(unprocessed_frame, cv2.COLOR_BGR2GRAY)
        canny_frame = self.auto_detect_canny_edges(gray_frame)
        frame = self.do_common_hough_circles_detection_functions(unprocessed_frame, canny_frame)

        return (frame, gray_frame, canny_frame)


    def detect_hough_circles_with_adaptive_thresh(self, unprocessed_frame):
        # funkcija cv2.pyrMeanShiftFiltering(unprocessed_frame, self.VC.PYR_MEAN_SHIFT_SP, self.VC.PYR_MEAN_SHIFT_SR) uprosječuje intenzitete sličnih područja i usporava izvođenje programa do tri puta, bez vidljivih poboljšanja pronalaženja objekata        gray_frame = cv2.cvtColor(unprocessed_frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.cvtColor(unprocessed_frame, cv2.COLOR_BGR2GRAY)
        thresh_frame = cv2.adaptiveThreshold(gray_frame, self.VC.MAX_ADAPTIVE_THRESH_VALUE, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, self.VC.ADAPTIVE_THRESH_WINDOW_SIZE, \
            self.VC.ADAPTIVE_THRESH_MEAN_OFFSET)
        frame = self.do_common_hough_circles_detection_functions(unprocessed_frame, thresh_frame)

        return (frame, gray_frame, thresh_frame)


    def do_common_hough_circles_detection_functions(self, unprocessed_frame, gray_or_thresh_frame):
        # hough_center_points_and_radii_list contains elements of the form (x_center, y_center, radius)
        frame = unprocessed_frame
        hough_radii_list = []
        if len(self.lasting_hough_radii_list) <= self.VC.MIN_NUM_OF_LASTING_HOUGH_RADII:
            hough_center_points_and_radii_list = cv2.HoughCircles(gray_or_thresh_frame, cv2.HOUGH_GRADIENT, self.VC.HOUGH_RATIO_OF_RESOLUTION, self.VC.HOUGH_MIN_DISTANCE_BETWEEN_CENTERS, \
                param1=self.VC.HOUGH_UPPER_CANNY_THRESH, param2=self.VC.HOUGH_UPPER_CENTER_THRESH, \
                minRadius=self.VC.HOUGH_MIN_RADIUS, maxRadius=self.VC.HOUGH_MAX_RADIUS)
        else:
            hough_center_points_and_radii_list = cv2.HoughCircles(gray_or_thresh_frame, cv2.HOUGH_GRADIENT, self.VC.HOUGH_RATIO_OF_RESOLUTION, self.VC.HOUGH_MIN_DISTANCE_BETWEEN_CENTERS, \
                param1=self.VC.HOUGH_UPPER_CANNY_THRESH, param2=self.VC.HOUGH_UPPER_CENTER_THRESH, \
                minRadius=int(self.average_hough_radius * self.VC.HOUGH_MIN_AVERAGE_RADIUS_FACTOR), maxRadius=int(self.average_hough_radius * self.VC.HOUGH_MAX_AVERAGE_RADIUS_FACTOR))

        if self.VC.USE_REFINED_OBJECTS_LIST:
            # this piece of code is only used with refined_objects_list
            if hough_center_points_and_radii_list is not None:
                hough_center_points_and_radii_list = hough_center_points_and_radii_list[0, :].astype(int)
                if len(hough_center_points_and_radii_list) >= self.VC.MAX_DETECTED_OBJECTS:
                    hough_center_points_and_radii_list = hough_center_points_and_radii_list[0:self.VC.MAX_DETECTED_OBJECTS]
                self.process_refined_objects_list(unprocessed_frame, hough_center_points_and_radii_list)
                for ready_refined_object in self.ready_refined_objects_list:
                    if not self.is_scene_change_significant():
                        cv2.circle(frame, (ready_refined_object.get_x(), ready_refined_object.get_y()), ready_refined_object.get_radius(), self.VC.COLOR_BLUE, self.VC.LINE_THICKNESS)
                        cv2.drawMarker(frame, (ready_refined_object.get_x(), ready_refined_object.get_y()), self.VC.COLOR_BLUE, cv2.MARKER_CROSS, self.VC.MARKER_SIZE, self.VC.LINE_THICKNESS)
                        cv2.putText(frame, "{}p".format(ready_refined_object.get_radius() * 2), (ready_refined_object.get_x(), ready_refined_object.get_y()), cv2.FONT_HERSHEY_SIMPLEX, \
                            self.VC.TEXT_SIZE_SMALL, self.VC.COLOR_BLUE, self.VC.LINE_THICKNESS)
            else:
                self.process_refined_objects_list(unprocessed_frame, [])
            if not self.is_scene_change_significant() and self.VC.PROCESS_FROM_CAM:
                cv2.putText(frame, "Detected: {}".format(len(self.ready_refined_objects_list)), (self.VC.DETECTED_OBJECTS_POS_X, self.VC.DETECTED_OBJECTS_POS_Y), cv2.FONT_HERSHEY_SIMPLEX, \
                    self.VC.TEXT_SIZE, self.VC.COLOR_BLUE, self.VC.LINE_THICKNESS)
        else:
            if self.are_new_detection_results_showing():
                self.old_hough_center_points_and_radii_list = hough_center_points_and_radii_list
            else:
                try: hough_center_points_and_radii_list = self.old_hough_center_points_and_radii_list
                except: self.old_hough_center_points_and_radii_list = hough_center_points_and_radii_list
            if hough_center_points_and_radii_list is not None:
                hough_center_points_and_radii_list = hough_center_points_and_radii_list[0, :].astype(int)
                if len(hough_center_points_and_radii_list) >= self.VC.MAX_DETECTED_OBJECTS:
                    hough_center_points_and_radii_list = hough_center_points_and_radii_list[0:self.VC.MAX_DETECTED_OBJECTS]
                num_of_hough_center_points_and_radii = len(hough_center_points_and_radii_list)
                for (center_point_x, center_point_y, radius) in hough_center_points_and_radii_list:
                    hough_radii_list.append(radius)
                    if not self.is_scene_change_significant():
                        cv2.circle(frame, (center_point_x, center_point_y), radius, self.VC.COLOR_BLUE, self.VC.LINE_THICKNESS)
                        cv2.drawMarker(frame, (center_point_x, center_point_y), self.VC.COLOR_BLUE, cv2.MARKER_CROSS, self.VC.MARKER_SIZE, self.VC.LINE_THICKNESS)
                        cv2.putText(frame, "{}p".format(radius * 2), (center_point_x, center_point_y), cv2.FONT_HERSHEY_SIMPLEX, self.VC.TEXT_SIZE_SMALL, self.VC.COLOR_BLUE, \
                            self.VC.LINE_THICKNESS)
            else:
                num_of_hough_center_points_and_radii = 0
            if not self.is_scene_change_significant() and self.VC.PROCESS_FROM_CAM:
                cv2.putText(frame, "Detected: {}".format(num_of_hough_center_points_and_radii), (self.VC.DETECTED_OBJECTS_POS_X, self.VC.DETECTED_OBJECTS_POS_Y), cv2.FONT_HERSHEY_SIMPLEX, \
                    self.VC.TEXT_SIZE, self.VC.COLOR_BLUE, self.VC.LINE_THICKNESS)

        if self.is_detection_mode_changed() or self.is_scene_change_significant():
            self.reset_hough_radii()
        else:
            self.average_hough_radii(hough_radii_list)

        return frame


    def reset_hough_radii(self):
        self.average_hough_radius = 0
        self.lasting_hough_radii_list = []


    def average_hough_radii(self, hough_radii_list):
        if hough_radii_list:
            for radius in hough_radii_list:
                self.lasting_hough_radii_list.append(radius)

        if len(self.lasting_hough_radii_list) >= self.VC.MAX_NUM_OF_LASTING_HOUGH_RADII:
            self.lasting_hough_radii_list = self.lasting_hough_radii_list[0:self.VC.MAX_NUM_OF_LASTING_HOUGH_RADII]

        if self.lasting_hough_radii_list:
            self.average_hough_radius = round(sum(self.lasting_hough_radii_list) / len(self.lasting_hough_radii_list), 1)
        else:
            self.average_hough_radius = 0


    def classify_with_haar_cascade(self, unprocessed_frame):
        frame = unprocessed_frame
        gray_frame = cv2.cvtColor(unprocessed_frame, cv2.COLOR_BGR2GRAY)
        equalized_gray_frame = cv2.equalizeHist(gray_frame)
        cascade_classifier = cv2.CascadeClassifier(self.VC.CASCADE_CLASSIFIER_FILE_PATH)
        cascade_sizes_list = []
        casc_center_points_and_sizes_list = []

        if self.VC.USE_EQUALIZED_GRAY_FRAME:
            casc_upper_left_points_and_sizes_list = cascade_classifier.detectMultiScale(equalized_gray_frame, self.VC.CASCADE_CLASSIFIER_SCALE_FACTOR, \
                self.VC.CASCADE_CLASSIFIER_MIN_NEIGHBOURS, minSize=(self.VC.CASCADE_CLASSIFIER_MIN_SIZE, self.VC.CASCADE_CLASSIFIER_MIN_SIZE), maxSize=(self.VC.CASCADE_CLASSIFIER_MAX_SIZE, \
                    self.VC.CASCADE_CLASSIFIER_MAX_SIZE))
        else:
            casc_upper_left_points_and_sizes_list = cascade_classifier.detectMultiScale(gray_frame, self.VC.CASCADE_CLASSIFIER_SCALE_FACTOR, self.VC.CASCADE_CLASSIFIER_MIN_NEIGHBOURS, \
                minSize=(self.VC.CASCADE_CLASSIFIER_MIN_SIZE, self.VC.CASCADE_CLASSIFIER_MIN_SIZE), maxSize=(self.VC.CASCADE_CLASSIFIER_MAX_SIZE, self.VC.CASCADE_CLASSIFIER_MAX_SIZE))

        for casc_upper_left_point in casc_upper_left_points_and_sizes_list:
            casc_center_point = (casc_upper_left_point[0] + int(casc_upper_left_point[2]/2), casc_upper_left_point[1] + int(casc_upper_left_point[3]/2))
            casc_center_points_and_sizes_list.append((casc_center_point[0], casc_center_point[1], casc_upper_left_point[2], casc_upper_left_point[3]))

        if self.VC.USE_REFINED_OBJECTS_LIST:
            # this piece of code is only used with refined_objects_listy
            if len(casc_center_points_and_sizes_list) >= self.VC.MAX_DETECTED_OBJECTS:
                casc_center_points_and_sizes_list = casc_center_points_and_sizes_list[0:self.VC.MAX_DETECTED_OBJECTS]
            self.process_refined_objects_list(unprocessed_frame, casc_center_points_and_sizes_list)
            for ready_refined_object in self.ready_refined_objects_list:
                cascade_sizes_list.append((width, height))
                if not self.is_scene_change_significant():
                    cv2.rectangle(frame, (ready_refined_object.get_upper_left_x(), ready_refined_object.get_upper_left_y()), (ready_refined_object.get_lower_right_x(), \
                        ready_refined_object.get_lower_right_y()), self.VC.COLOR_BLUE, self.VC.LINE_THICKNESS)
                    cv2.drawMarker(frame, (ready_refined_object.get_x(), ready_refined_object.get_y()), self.VC.COLOR_BLUE, cv2.MARKER_CROSS, self.VC.MARKER_SIZE, self.VC.LINE_THICKNESS)
                    cv2.putText(frame, "{}p".format(ready_refined_object.get_height()), (ready_refined_object.get_x(), ready_refined_object.get_y()), cv2.FONT_HERSHEY_SIMPLEX, \
                        self.VC.TEXT_SIZE_SMALL, self.VC.COLOR_BLUE, self.VC.LINE_THICKNESS)
            if not self.is_scene_change_significant() and self.VC.PROCESS_FROM_CAM:
                cv2.putText(frame, "Detected: {}".format(len(self.ready_refined_objects_list)), (self.VC.DETECTED_OBJECTS_POS_X, self.VC.DETECTED_OBJECTS_POS_Y), cv2.FONT_HERSHEY_SIMPLEX, \
                    self.VC.TEXT_SIZE, self.VC.COLOR_BLUE, self.VC.LINE_THICKNESS)
        else:
            if self.are_new_detection_results_showing():
                self.old_casc_center_points_and_sizes_list = casc_center_points_and_sizes_list
            else:
                try: casc_center_points_and_sizes_list = self.old_casc_center_points_and_sizes_list
                except: self.old_casc_center_points_and_sizes_list = casc_center_points_and_sizes_list
            if len(casc_center_points_and_sizes_list) >= self.VC.MAX_DETECTED_OBJECTS:
                casc_center_points_and_sizes_list = casc_center_points_and_sizes_list[0:self.VC.MAX_DETECTED_OBJECTS]
            for (center_point_x, center_point_y, width, height) in casc_center_points_and_sizes_list:
                cascade_sizes_list.append((width, height))
                if not self.is_scene_change_significant():
                    cv2.rectangle(frame, (center_point_x - int(width/2), center_point_y - int(height/2)), (center_point_x + int(width/2), center_point_y + int(height/2)), self.VC.COLOR_BLUE, \
                        self.VC.LINE_THICKNESS)
                    cv2.drawMarker(frame, (center_point_x, center_point_y), self.VC.COLOR_BLUE, cv2.MARKER_CROSS, self.VC.MARKER_SIZE, self.VC.LINE_THICKNESS)
                    cv2.putText(frame, "{}p".format(height), (center_point_x, center_point_y), cv2.FONT_HERSHEY_SIMPLEX, self.VC.TEXT_SIZE_SMALL, self.VC.COLOR_BLUE, self.VC.LINE_THICKNESS)
            if not self.is_scene_change_significant() and self.VC.PROCESS_FROM_CAM:
                cv2.putText(frame, "Detected: {}".format(len(casc_center_points_and_sizes_list)), (self.VC.DETECTED_OBJECTS_POS_X, self.VC.DETECTED_OBJECTS_POS_Y), \
                    cv2.FONT_HERSHEY_SIMPLEX, self.VC.TEXT_SIZE, self.VC.COLOR_BLUE, self.VC.LINE_THICKNESS)

        return (frame, gray_frame, equalized_gray_frame)



    # the next piece of code is only used with refined_objects_list

    class ObjectDetectedAsTemplate:

        def __init__(self, x, y, width, height):
            self.x = x
            self.y = y
            self.width = width
            self.height = height
            self.ponder = 1

        def calculate_distance(self, other): return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

        def average_attributes(self, other):
            self.x = int(self.x * self.ponder + other.x * other.ponder / (self.ponder + other.ponder))
            self.y = int(self.y * self.ponder + other.y * other.ponder / (self.ponder + other.ponder))
            self.width = int(self.width * self.ponder + other.width * other.ponder / (self.ponder + other.ponder))
            self.height = int(self.height * self.ponder + other.height * other.ponder / (self.ponder + other.ponder))
            self.ponder = self.ponder + other.ponder

        def get_x(self): return self.x

        def get_y(self): return self.y

        def get_ponder(self): return self.ponder

        def get_width(self): return self.width

        def get_height(self): return self.height

        def get_upper_left_x(self): return self.x - int(self.width/2)

        def get_upper_left_y(self): return self.y - int(self.height/2)

        def get_lower_right_x(self): return self.x + int(self.width/2)

        def get_lower_right_y(self): return self.y + int(self.height/2)


    class ObjectDetectedAsHoughCircle:

        def __init__(self, x, y, radius):
            self.x = x
            self.y = y
            self.radius = radius
            self.ponder = 1

        def calculate_distance(self, other): return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

        def average_attributes(self, other):
            self.x = int(self.x * self.ponder + other.x * other.ponder / (self.ponder + other.ponder))
            self.y = int(self.y * self.ponder + other.y * other.ponder / (self.ponder + other.ponder))
            self.radius = int(self.radius * self.ponder + other.radius * other.ponder / (self.ponder + other.ponder))
            self.ponder = self.ponder + other.ponder

        def get_x(self): return self.x

        def get_y(self): return self.y

        def get_ponder(self): return self.ponder

        def get_radius(self): return self.radius


    def set_fps(self, fps):
        self.fps = fps


    def reset_refined_objects_list(self):
        self.num_of_frames_processed = 0
        self.num_of_frames_group_processed = 0
        self.refined_objects_list = []
        self.ready_refined_objects_list = []


    def process_refined_objects_list(self, unprocessed_frame, detected_data_list):
        num_of_frames_to_process_before_refined_is_ready = self.fps * self.VC.TIME_TO_PROCESS_BEFORE_REFINED_IS_READY
        num_of_frames_to_group_process_before_resetting_refined = self.fps * self.VC.TIME_TO_GROUP_PROCESS_BEFORE_RESETTING_REFINED

        if self.is_detection_mode_changed() or self.is_scene_change_significant():
            self.reset_refined_objects_list()

        if self.num_of_frames_group_processed < num_of_frames_to_group_process_before_resetting_refined:
            if self.num_of_frames_processed < num_of_frames_to_process_before_refined_is_ready:
                self.update_refined_objects_list(detected_data_list)
                self.num_of_frames_processed += 1
            else:
                if self.refined_objects_list:
                    average_refined_objects_ponder = sum([refined_object.get_ponder() for refined_object in self.refined_objects_list]) / len(self.refined_objects_list)
                    if isinstance(self.refined_objects_list[0], self.ObjectDetectedAsTemplate):
                        average_refined_objects_width = sum([refined_object.get_width() for refined_object in self.refined_objects_list]) / len(self.refined_objects_list)
                        average_refined_objects_height = sum([refined_object.get_height() for refined_object in self.refined_objects_list]) / len(self.refined_objects_list)
                        for refined_object in self.refined_objects_list:
                            if refined_object.get_ponder() <= average_refined_objects_ponder * self.VC.MIN_PONDER_FACTOR or refined_object.get_width() <= average_refined_objects_width * \
                                self.VC.MIN_WIDTH_AND_HEIGHT_FACTOR or refined_object.get_width() >= average_refined_objects_width * self.VC.MAX_WIDTH_AND_HEIGHT_FACTOR or \
                                refined_object.get_height() <= average_refined_objects_height * self.VC.MIN_WIDTH_AND_HEIGHT_FACTOR or refined_object.get_height() >= \
                                average_refined_objects_height * self.VC.MAX_WIDTH_AND_HEIGHT_FACTOR:
                                del self.refined_objects_list[self.refined_objects_list.index(refined_object)]
                                del refined_object
                    elif isinstance(self.refined_objects_list[0], self.ObjectDetectedAsHoughCircle):
                        average_refined_objects_radius = sum([refined_object.get_radius() for refined_object in self.refined_objects_list]) / len(self.refined_objects_list)
                        for refined_object in self.refined_objects_list:
                            if refined_object.get_ponder() <= average_refined_objects_ponder * self.VC.MIN_PONDER_FACTOR or refined_object.get_radius() <= average_refined_objects_radius * \
                                self.VC.MIN_RADIUS_FACTOR or refined_object.get_radius() >= average_refined_objects_radius * self.VC.MAX_RADIUS_FACTOR:
                                del self.refined_objects_list[self.refined_objects_list.index(refined_object)]
                                del refined_object
                self.ready_refined_objects_list = self.refined_objects_list
                self.num_of_frames_processed = 0


    def update_refined_objects_list(self, detected_data_list):
        if detected_data_list is not None:
            for detected_data in detected_data_list:
                # if the objects are detected as templates, then they are of the form (x_center, y_center, template_width, template_height)
                if len(detected_data) == 4:
                    detected_object = self.ObjectDetectedAsTemplate(detected_data[0], detected_data[1], detected_data[2], detected_data[3])
                # if the objects are detected as Hough circles, then they are of the form (x_center, y_center, radius)
                elif len(detected_data) == 3:
                    detected_object = self.ObjectDetectedAsHoughCircle(detected_data[0], detected_data[1], detected_data[2])
                if not self.refined_objects_list:
                    self.refined_objects_list.append(detected_object)
                else:
                    detected_object_similar_to_some_refined_object = False
                    for refined_object in self.refined_objects_list:
                        if refined_object.calculate_distance(detected_object) <= self.VC.MIN_DISTANCE_BETWEEN_CENTERS:
                            refined_object.average_attributes(detected_object)
                            detected_object_similar_to_some_refined_object = True
                            break
                    if not detected_object_similar_to_some_refined_object:
                        self.refined_objects_list.append(detected_object)
                del detected_object