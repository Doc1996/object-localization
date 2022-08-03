import os


class WebcamVideoConstants:

    def __init__(self):
        self.PROCESS_FROM_CAM = True
        self.CAM_FROM_PHONE = False
        self.PHONE_CAM_URL = "http://192.168.43.1:8080/video"
        self.CAM_SOURCE = 0
        self.CAM_WIDTH = 320  # 160, 320 or 640 pixels
        self.CAM_HEIGHT = int(self.CAM_WIDTH * 0.75)  # 120, 240 or 480 pixels
        if self.CAM_WIDTH == 320:
            self.TEMPLATE_SCALE_FACTOR = 1
        else:
            self.TEMPLATE_SCALE_FACTOR = self.CAM_WIDTH / 320
        self.FLIP_HORIZONTALLY = 1
        self.MIN_FRAME_TIME = 1  # time in milliseconds
        self.MIN_SHOW_FPS_INTERVAL = 0.05  # time in seconds
        self.MIN_SHOW_RESULTS_INTERVAL = 0.05  # time in seconds
        self.KEY_ESC = 27  # the number of key that stops the execution of program, in this case ESC

        self.UNPROCESSED_IMAGE_FILES_PATH = os.path.dirname(os.path.abspath(__file__)).replace(os.sep, "/") + "/unprocessed images/*"  # it has an asterisk to include all files
        self.PROCESSED_IMAGES_FOLDER_PATH = os.path.dirname(os.path.abspath(__file__)).replace(os.sep, "/") + "/processed images/"


        self.MIN_OTSU_THRESH_VALUE = 0  # for 0 the function finds the optimal value by itself
        self.MAX_OTSU_THRESH_VALUE = 255
        self.MAX_ADAPTIVE_THRESH_VALUE = 255
        self.ADAPTIVE_THRESH_WINDOW_SIZE = 11  # optimal value is 11
        self.ADAPTIVE_THRESH_MEAN_OFFSET = 5  # optimal value is 5
        self.MIN_CANNY_THRESH_VALUE = 0
        self.MAX_CANNY_THRESH_VALUE = 255
        self.CANNY_SIGMA_VALUE = 0.33  # optimal value is 0.33
        self.MIN_BINARY_THRESH_VALUE_FOR_ABSDIFF = 50  # optimal value is 50
        self.MAX_BINARY_THRESH_VALUE_FOR_ABSDIFF = 255

        self.SHOW_MASK_VIDEOS = False
        self.VIDEO_WINDOW_NAME = "video"
        self.VIDEO_WINDOW_POS_X = 0
        self.VIDEO_WINDOW_POS_Y = 0
        self.FIRST_MASK_VIDEO_WINDOW_NAME = "first mask video"
        self.FIRST_MASK_VIDEO_WINDOW_POS_X = self.CAM_WIDTH + 2
        self.FIRST_MASK_VIDEO_WINDOW_POS_Y = 0
        self.SECOND_MASK_VIDEO_WINDOW_NAME = "second mask video"
        self.SECOND_MASK_VIDEO_WINDOW_POS_X = 0
        self.SECOND_MASK_VIDEO_WINDOW_POS_Y = self.CAM_HEIGHT + 32
        self.TRACKBARS_WINDOW_NAME = "trackbars"
        self.TRACKBARS_WINDOW_POS_X = self.CAM_WIDTH + 2
        self.TRACKBARS_WINDOW_POS_Y = self.CAM_HEIGHT + 32
        self.TRACKBAR_WINDOW_WIDTH = self.CAM_WIDTH
        self.TRACKBAR_WINDOW_HEIGHT = int(self.CAM_HEIGHT / 3)

        self.TRACKBAR_DETECTION_NAME = "detection"
        self.TRACKBAR_ALL_DETECTION_MODES = ["No Detection", "Gray Template Matching", "Binary Template Matching", "Canny Edges Hough", "Adaptive Threshold Hough", "Haar Cascade"]
        self.TRACKBAR_DETECTION_MODE = self.TRACKBAR_ALL_DETECTION_MODES[0]
        self.TRACKBAR_MASK_VIDEOS_NAME = "masks"
        self.TRACKBAR_ALL_MASK_VIDEOS_MODES = [False, True]
        self.TRACKBAR_MASK_VIDEOS_MODE = True

        self.DETECTION_MODE_POS_X = 10
        self.DETECTION_MODE_POS_Y = 25
        self.FPS_POS_X = 10
        self.FPS_POS_Y = self.CAM_HEIGHT - 15
        self.DETECTED_OBJECTS_POS_X = int(self.CAM_WIDTH / 2) + 30
        self.DETECTED_OBJECTS_POS_Y = self.CAM_HEIGHT - 15

        # let the original templates have dimensions of 64x64 pixels, so they will be scaled down, because in a 320x240 pixel image, those larger than 64x64 pixels won't be needed
        # all original templates must have the same dimensions, because otherwise you cannot use the function to combine templates into one image for display
        # scaling must move from larger dimensions to smaller ones because when small dimensions only find a part of an object, that part can still be accepted as the whole object
        self.THREADED_TEMPLATE_MATCHING = True
        self.GRAY_TEMPLATE_IMAGE_FILES_PATH = os.path.dirname(os.path.abspath(__file__)).replace(os.sep, "/") + "/templates/gray/*"  # it has an asterisk to include all files
        self.BINARY_TEMPLATE_IMAGE_FILES_PATH = os.path.dirname(os.path.abspath(__file__)).replace(os.sep, "/") + "/templates/binary/*"  # it has an asterisk to include all files
        self.TEMPLATE_SCALES_LIST = [template_scale * self.TEMPLATE_SCALE_FACTOR for template_scale in [0.8, 0.675, 0.55, 0.475, 0.4]]
        # it is intended to use [0.55, 0.65, 0.75] so that you can see how reliable the detection is, but when it is not required, then anything above 0.55 should not be used
        self.TEMPLATE_MATCHING_GRAY_THRESHS_LIST = [0.55]  # može ići samo do 0.55 jer inače pihvaća i ono što nisu traženi objekti
        # it is intended to use [0.45, 0.55, 0.65, 0.75] so that you can see how reliable the detection is, but when it is not required, then anything above 0.45 should not be used
        self.TEMPLATE_MATCHING_BINARY_THRESHS_LIST = [0.45]  # can go up to 0.45 because it is quite robust
        self.MAX_DETECTED_OBJECTS = 30

        # for close objects to be detected, their slight overlap must be allowed, which is why the distances from the centers are set to 1/3
        self.MIN_OBJECTS_IN_SCENE_HEIGHT = 4
        self.MAX_OBJECTS_IN_SCENE_HEIGHT = 8
        self.PYR_MEAN_SHIFT_SP = 5  # optimal value is 5
        self.PYR_MEAN_SHIFT_SR = 10  # optimal value is 10
        self.HOUGH_RATIO_OF_RESOLUTION = 1  # optimal value is 1
        self.HOUGH_MIN_RADIUS = int(self.CAM_HEIGHT / self.MAX_OBJECTS_IN_SCENE_HEIGHT / 2)
        self.HOUGH_MAX_RADIUS = int(self.CAM_HEIGHT / self.MIN_OBJECTS_IN_SCENE_HEIGHT / 2)
        self.HOUGH_MIN_DISTANCE_BETWEEN_CENTERS = int(self.HOUGH_MIN_RADIUS * 2 * 2/3)
        self.HOUGH_UPPER_CANNY_THRESH = 30  # optimal value is 30
        self.HOUGH_UPPER_CENTER_THRESH = 30  # optimal value is 30

        self.MIN_NUM_OF_LASTING_HOUGH_RADII = self.MAX_DETECTED_OBJECTS
        self.MAX_NUM_OF_LASTING_HOUGH_RADII = self.MAX_DETECTED_OBJECTS * 2
        self.HOUGH_MIN_AVERAGE_RADIUS_FACTOR = 0.66
        self.HOUGH_MAX_AVERAGE_RADIUS_FACTOR = 1.5

        self.USE_EQUALIZED_GRAY_FRAME = False
        self.CASCADE_CLASSIFIERS_FILES_LIST = ["cascade_16_gray.xml", "cascade_16_gray_equalized.xml", "cascade_18_gray.xml", "cascade_18_gray_equalized.xml", "cascade_20_gray.xml", \
            "cascade_20_gray_equalized.xml"]
        self.CASCADE_CLASSIFIER_FILE = self.CASCADE_CLASSIFIERS_FILES_LIST[0]  # the cascade_16_gray.xml classifier gives the best results
        self.CASCADE_CLASSIFIER_FILE_PATH = os.path.dirname(os.path.abspath(__file__)).replace(os.sep, "/") + "/classifiers/" + self.CASCADE_CLASSIFIER_FILE
        self.CASCADE_CLASSIFIER_SCALE_FACTOR = 1.01  # optimal value for classifiers is 1.01
        if self.CASCADE_CLASSIFIER_FILE in [self.CASCADE_CLASSIFIERS_FILES_LIST[0], self.CASCADE_CLASSIFIERS_FILES_LIST[1]]:
            self.CASCADE_CLASSIFIER_MIN_NEIGHBOURS = 8  # optimal value for classifiers with 16 stages is 8
        elif self.CASCADE_CLASSIFIER_FILE in [self.CASCADE_CLASSIFIERS_FILES_LIST[2], self.CASCADE_CLASSIFIERS_FILES_LIST[3]]:
            self.CASCADE_CLASSIFIER_MIN_NEIGHBOURS = 4  # optimal value for classifiers with 18 stages is 4
        elif self.CASCADE_CLASSIFIER_FILE in [self.CASCADE_CLASSIFIERS_FILES_LIST[4], self.CASCADE_CLASSIFIERS_FILES_LIST[5]]:
            self.CASCADE_CLASSIFIER_MIN_NEIGHBOURS = 2  # optimal value for classifiers with 20 stages is 2
        self.CASCADE_CLASSIFIER_MIN_SIZE = int(self.CAM_HEIGHT / self.MAX_OBJECTS_IN_SCENE_HEIGHT / (3/2))  # the min. dim. is one third smaller than the min. diameter of Hough circles
        self.CASCADE_CLASSIFIER_MAX_SIZE = int(self.CAM_HEIGHT / self.MIN_OBJECTS_IN_SCENE_HEIGHT)  # the max. dimension is equal to the max. diameter of the Hough circles

        self.USE_REFINED_OBJECTS_LIST = False
        self.MIN_PONDER_FACTOR = 0.2
        self.MIN_RADIUS_FACTOR = 0.66
        self.MAX_RADIUS_FACTOR = 1.5
        self.MIN_WIDTH_AND_HEIGHT_FACTOR = 0.66
        self.MAX_WIDTH_AND_HEIGHT_FACTOR = 1.5
        self.TIME_TO_PROCESS_BEFORE_REFINED_IS_READY = 0.05  # time in seconds
        self.NUM_OF_GROUP_PROCESSINGS_BEFORE_RESETTING_REFINED = 5
        self.TIME_TO_GROUP_PROCESS_BEFORE_RESETTING_REFINED = self.TIME_TO_PROCESS_BEFORE_REFINED_IS_READY * self.NUM_OF_GROUP_PROCESSINGS_BEFORE_RESETTING_REFINED
        self.MIN_DISTANCE_BETWEEN_CENTERS = int(self.HOUGH_MIN_DISTANCE_BETWEEN_CENTERS / 4)
        self.MAX_CONSECUTIVE_FRAMES_DIFFERENCE = 0.05  # value that multiplied by hundered corresponds to percentages

        self.TEXT_SIZE_SMALL = 0.35
        self.TEXT_SIZE = 0.5
        self.MARKER_SIZE = 10
        self.LINE_THICKNESS = 1
        self.COLOR_WHITE = (255, 255, 255)
        self.COLOR_BLACK = (0, 0, 0)
        self.COLOR_BLUE = (255, 0, 0)
        self.COLOR_CYAN = (255, 255, 0)
        self.COLOR_GREEN = (0, 255, 0)
        self.COLOR_YELLOW = (0, 255, 255)
        self.COLOR_RED = (0, 0, 255)
        self.COLOR_MAGENTA = (255, 0, 255)