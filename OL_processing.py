import cv2
import time
import os
import glob


class WebcamVideoProcessing:

    def __init__(self, video_const, video_algorithms, video_input=None, video_output=None, video_trackbars=None):
        self.VC = video_const
        self.VA = video_algorithms

        if self.VC.PROCESS_FROM_CAM:
            self.VI = video_input.start()
            self.VO = video_output.start()
            self.VT = video_trackbars
            self.VT.set_detection_mode_trackbar(self.VC.TRACKBAR_DETECTION_MODE)
            self.VT.set_mask_videos_mode_trackbar(self.VC.TRACKBAR_MASK_VIDEOS_MODE)
            self.reset_fps()
            self.new_detection_mode = None

        self.gray_temp_images_list = self.make_gray_template_images_list()
        self.binary_temp_images_list = self.make_binary_template_images_list()
        self.concatenated_gray_temp_images = self.concatenate_gray_template_images()
        self.concatenated_binary_temp_images = self.concatenate_binary_template_images()


    def reset_fps(self):
        self.start_time = time.time()
        self.new_time = self.start_time
        self.new_fps = 0
        self.counted_frames = 0


    def calculate_fps(self):
        self.old_time = self.new_time
        self.old_fps = self.new_fps
        self.elapsed_time = round(float(time.time() - self.start_time), 4)

        if round(float(time.time() - self.old_time), 1) > self.VC.MIN_SHOW_FPS_INTERVAL:
            self.new_time = time.time()
            if self.elapsed_time > 0:
                self.new_fps = round(self.counted_frames / self.elapsed_time, 1)
            else:
                self.new_fps = 0
        else:
            self.new_fps = self.old_fps

        return self.new_fps


    def make_gray_template_images_list(self):
        self.gray_temp_images_list = []
        temp_image_files = glob.glob(self.VC.GRAY_TEMPLATE_IMAGE_FILES_PATH)

        for temp_image_file in temp_image_files:
            gray_temp_image = cv2.imread(temp_image_file, cv2.IMREAD_GRAYSCALE)
            temp_width = gray_temp_image.shape[1]
            temp_height = gray_temp_image.shape[0]
            for temp_scale in self.VC.TEMPLATE_SCALES_LIST:
                gray_temp_image_scaled = cv2.resize(gray_temp_image, (int(temp_width * temp_scale), int(temp_height * temp_scale)))
                self.gray_temp_images_list.append(gray_temp_image_scaled)

        return self.gray_temp_images_list


    def make_binary_template_images_list(self):
        self.binary_temp_images_list = []
        temp_image_files = glob.glob(self.VC.BINARY_TEMPLATE_IMAGE_FILES_PATH)

        for temp_image_file in temp_image_files:
            binary_temp_image = cv2.imread(temp_image_file, cv2.IMREAD_GRAYSCALE)
            temp_width = binary_temp_image.shape[1]
            temp_height = binary_temp_image.shape[0]
            for temp_scale in self.VC.TEMPLATE_SCALES_LIST:
                binary_temp_image_scaled = cv2.resize(binary_temp_image, (int(temp_width * temp_scale), int(temp_height * temp_scale)))
                self.binary_temp_images_list.append(binary_temp_image_scaled)

        return self.binary_temp_images_list


    def concatenate_gray_template_images(self):
        temp_image_files = glob.glob(self.VC.GRAY_TEMPLATE_IMAGE_FILES_PATH)
        original_gray_temp_images_list = []

        for temp_image_file in temp_image_files:
            gray_temp_image = cv2.imread(temp_image_file, cv2.IMREAD_GRAYSCALE)
            original_gray_temp_images_list.append(gray_temp_image)
        self.concatenated_gray_temp_images = cv2.hconcat(original_gray_temp_images_list)

        return self.concatenated_gray_temp_images


    def concatenate_binary_template_images(self):
        temp_image_files = glob.glob(self.VC.BINARY_TEMPLATE_IMAGE_FILES_PATH)
        original_binary_temp_images_list = []

        for temp_image_file in temp_image_files:
            binary_temp_image = cv2.imread(temp_image_file, cv2.IMREAD_GRAYSCALE)
            original_binary_temp_images_list.append(binary_temp_image)
        self.concatenated_binary_temp_images = cv2.hconcat(original_binary_temp_images_list)

        return self.concatenated_binary_temp_images


    def loop_video(self):
        (ret, last_frame) = self.VI.read()
        # last_frame = cv2.flip(last_frame, self.VC.FLIP_HORIZONTALLY)

        while True:
            if not self.VI.is_started() or not self.VO.is_started():
                self.VI.stop()
                self.VO.stop()
                break

            (ret, frame) = self.VI.read()
            # frame = cv2.flip(frame, self.VC.FLIP_HORIZONTALLY)
            self.VA.set_fps(self.new_fps)
            self.VA.check_scene_change_significancy(last_frame, frame)
            self.VA.check_new_detection_results_showing()
            self.old_detection_mode = self.new_detection_mode

            if self.VA.is_scene_change_significant():
                self.VA.update_detection_results_showing()

            if self.VC.TRACKBAR_DETECTION_MODE == self.VC.TRACKBAR_ALL_DETECTION_MODES[0]:
                (processed_frame, processed_first_mask_frame) = (frame.copy(), frame.copy())
                processed_second_mask_frame = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)
            elif self.VC.TRACKBAR_DETECTION_MODE == self.VC.TRACKBAR_ALL_DETECTION_MODES[1]:
                (processed_frame, processed_first_mask_frame) = self.VA.match_gray_templates(frame.copy(), self.gray_temp_images_list)
                processed_second_mask_frame = self.concatenated_gray_temp_images
            elif self.VC.TRACKBAR_DETECTION_MODE == self.VC.TRACKBAR_ALL_DETECTION_MODES[2]:
                (processed_frame, processed_first_mask_frame) = self.VA.match_binary_templates(frame.copy(), self.binary_temp_images_list)
                processed_second_mask_frame = self.concatenated_binary_temp_images
            elif self.VC.TRACKBAR_DETECTION_MODE == self.VC.TRACKBAR_ALL_DETECTION_MODES[3]:
                (processed_frame, processed_first_mask_frame, processed_second_mask_frame) = self.VA.detect_hough_circles_with_canny_edges(frame.copy())
            elif self.VC.TRACKBAR_DETECTION_MODE == self.VC.TRACKBAR_ALL_DETECTION_MODES[4]:
                (processed_frame, processed_first_mask_frame, processed_second_mask_frame) = self.VA.detect_hough_circles_with_adaptive_thresh(frame.copy())
            elif self.VC.TRACKBAR_DETECTION_MODE == self.VC.TRACKBAR_ALL_DETECTION_MODES[5]:
                (processed_frame, processed_first_mask_frame, processed_second_mask_frame) = self.VA.classify_with_haar_cascade(frame.copy())

            self.VC.TRACKBAR_DETECTION_MODE = self.VT.get_detection_mode_trackbar()
            self.VC.TRACKBAR_MASK_VIDEOS_MODE = self.VT.get_mask_videos_mode_trackbar()
            self.VC.SHOW_MASK_VIDEOS = self.VC.TRACKBAR_MASK_VIDEOS_MODE
            self.new_detection_mode = self.VC.TRACKBAR_DETECTION_MODE
            if self.new_detection_mode != self.old_detection_mode:
                self.VA.set_detection_mode_change(True)
                self.reset_fps()
            else:
                self.VA.set_detection_mode_change(False)
            self.counted_frames += 1
            self.new_fps = self.calculate_fps()
            if self.VA.are_new_detection_results_showing():
                self.VA.update_detection_results_showing()
            last_frame = frame

            cv2.putText(processed_frame, "{}".format(self.new_detection_mode), (self.VC.DETECTION_MODE_POS_X, self.VC.DETECTION_MODE_POS_Y), cv2.FONT_HERSHEY_SIMPLEX, self.VC.TEXT_SIZE, \
                self.VC.COLOR_BLUE, self.VC.LINE_THICKNESS*2)
            cv2.putText(processed_frame, "Max. FPS: {}".format(self.new_fps), (self.VC.FPS_POS_X, self.VC.FPS_POS_Y), cv2.FONT_HERSHEY_SIMPLEX, self.VC.TEXT_SIZE, self.VC.COLOR_BLUE, \
                self.VC.LINE_THICKNESS)
            self.VO.set_frame(processed_frame)
            self.VO.set_first_mask_frame(processed_first_mask_frame)
            self.VO.set_second_mask_frame(processed_second_mask_frame)
            cv2.waitKey(self.VC.MIN_FRAME_TIME)


    def loop_images_in_folder(self):
        unprocessed_image_files = glob.glob(self.VC.UNPROCESSED_IMAGE_FILES_PATH)
        processed_images_folder = self.VC.PROCESSED_IMAGES_FOLDER_PATH
        num_of_processed_image = 1
        for detection_mode in self.VC.TRACKBAR_ALL_DETECTION_MODES:
            if not os.path.exists(processed_images_folder + detection_mode):
                os.makedirs(processed_images_folder + detection_mode)

        for unprocessed_image_file in unprocessed_image_files:
            image = cv2.imread(unprocessed_image_file, cv2.IMREAD_UNCHANGED)
            for detection_mode in self.VC.TRACKBAR_ALL_DETECTION_MODES:
                if detection_mode == self.VC.TRACKBAR_ALL_DETECTION_MODES[0]:
                    (processed_image, processed_first_mask_image) = (image.copy(), image.copy())
                    processed_second_mask_image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
                elif detection_mode == self.VC.TRACKBAR_ALL_DETECTION_MODES[1]:
                    (processed_image, processed_first_mask_image) = self.VA.match_gray_templates(image.copy(), self.gray_temp_images_list)
                    processed_second_mask_image = self.concatenated_gray_temp_images
                elif detection_mode == self.VC.TRACKBAR_ALL_DETECTION_MODES[2]:
                    (processed_image, processed_first_mask_image) = self.VA.match_binary_templates(image.copy(), self.binary_temp_images_list)
                    processed_second_mask_image = self.concatenated_binary_temp_images
                elif detection_mode == self.VC.TRACKBAR_ALL_DETECTION_MODES[3]:
                    (processed_image, processed_first_mask_image, processed_second_mask_image) = self.VA.detect_hough_circles_with_canny_edges(image.copy())
                elif detection_mode == self.VC.TRACKBAR_ALL_DETECTION_MODES[4]:
                    (processed_image, processed_first_mask_image, processed_second_mask_image) = self.VA.detect_hough_circles_with_adaptive_thresh(image.copy())
                elif detection_mode == self.VC.TRACKBAR_ALL_DETECTION_MODES[5]:
                    (processed_image, processed_first_mask_image, processed_second_mask_image) = self.VA.classify_with_haar_cascade(image.copy())
                cv2.imwrite(processed_images_folder + detection_mode + "/" + str(num_of_processed_image) + ".jpg", processed_image)
                cv2.waitKey(self.VC.MIN_FRAME_TIME)
            num_of_processed_image += 1