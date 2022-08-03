import cv2


class WebcamVideoTrackbars:

    def __init__(self, video_const):
        self.VC = video_const

        cv2.namedWindow(self.VC.TRACKBARS_WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.resizeWindow(self.VC.TRACKBARS_WINDOW_NAME, (self.VC.TRACKBAR_WINDOW_WIDTH, self.VC.TRACKBAR_WINDOW_HEIGHT))
        cv2.moveWindow(self.VC.TRACKBARS_WINDOW_NAME, self.VC.TRACKBARS_WINDOW_POS_X, self.VC.TRACKBARS_WINDOW_POS_Y)


    def set_detection_mode_trackbar(self, trackbar_detection_mode):
        def nothing(trackbar_var): pass

        lowest_detection_mode_value = 0
        highest_detection_mode_value = len(self.VC.TRACKBAR_ALL_DETECTION_MODES) - 1

        cv2.createTrackbar(self.VC.TRACKBAR_DETECTION_NAME, self.VC.TRACKBARS_WINDOW_NAME, lowest_detection_mode_value, highest_detection_mode_value, nothing)

        for detection_mode in self.VC.TRACKBAR_ALL_DETECTION_MODES:
            if trackbar_detection_mode == detection_mode:
                detection_mode_value = self.VC.TRACKBAR_ALL_DETECTION_MODES.index(detection_mode)

        cv2.setTrackbarPos(self.VC.TRACKBAR_DETECTION_NAME, self.VC.TRACKBARS_WINDOW_NAME, detection_mode_value)


    def get_detection_mode_trackbar(self):
        detection_mode_value = cv2.getTrackbarPos(self.VC.TRACKBAR_DETECTION_NAME, self.VC.TRACKBARS_WINDOW_NAME)

        for detection_mode in self.VC.TRACKBAR_ALL_DETECTION_MODES:
            if detection_mode_value == self.VC.TRACKBAR_ALL_DETECTION_MODES.index(detection_mode):
                trackbar_detection_mode = detection_mode

        return trackbar_detection_mode


    def set_mask_videos_mode_trackbar(self, trackbar_mask_videos_mode):
        def nothing(trackbar_var): pass

        lowest_mask_videos_mode_value = 0
        highest_mask_videos_mode_value = len(self.VC.TRACKBAR_ALL_MASK_VIDEOS_MODES) - 1

        cv2.createTrackbar(self.VC.TRACKBAR_MASK_VIDEOS_NAME, self.VC.TRACKBARS_WINDOW_NAME, lowest_mask_videos_mode_value, highest_mask_videos_mode_value, nothing)

        for detection_mode in self.VC.TRACKBAR_ALL_MASK_VIDEOS_MODES:
            if trackbar_mask_videos_mode == detection_mode:
                mask_videos_mode_value = self.VC.TRACKBAR_ALL_MASK_VIDEOS_MODES.index(detection_mode)

        cv2.setTrackbarPos(self.VC.TRACKBAR_MASK_VIDEOS_NAME, self.VC.TRACKBARS_WINDOW_NAME, mask_videos_mode_value)


    def get_mask_videos_mode_trackbar(self):
        mask_videos_mode_value = cv2.getTrackbarPos(self.VC.TRACKBAR_MASK_VIDEOS_NAME, self.VC.TRACKBARS_WINDOW_NAME)

        for detection_mode in self.VC.TRACKBAR_ALL_MASK_VIDEOS_MODES:
            if mask_videos_mode_value == self.VC.TRACKBAR_ALL_MASK_VIDEOS_MODES.index(detection_mode):
                trackbar_mask_videos_mode = detection_mode

        return trackbar_mask_videos_mode