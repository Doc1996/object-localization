import cv2
import os


# minimum parameters will be 1.01 and 1, and maximum 1.05 and 10

# the smaller the first and second parameters, the more objects are detected and the execution is slower (there are also more false-positives and multiple detected objects), and vice versa

# the best first parameter is 1.01 because it is fast enough and detects objects most accurately

# min_dim must not be more than 20, and max_dim must not be less than 60

# it is better not to detect some objects than to detect false-positives

# each classifier can be applied to both grayscale and equalized grayscale images, but each is also slightly better for grayscale images

# classifiers with 16 stages are generally better than those with 18 and 20 because they are less rigorous in accepting features, so with the increase of the second parameter, false-positives
# are removed, but correctly detected objects are not

# classifiers based on grayscale images are generally better than those based on grayscale equalized histograms because they don't find as many false-positives

# cascade_16_gray.xml is best used with values 1.01 and 8   # rang 1
# cascade_16_gray_equalized.xml is best used with values 1.01 and 8   # rang 2
# cascade_18_gray.xml is best used with values 1.01 and 4   # rang 3
# cascade_18_gray_equalized.xml is best used with values 1.01 and 4   # rang 4
# cascade_20_gray.xml is usable only for values 1.01 and 2 (it is too rigorous in accepting features)   # rang 5
# cascade_20_gray_equalized.xml is usable only for values 1.01 and 2 (it is too rigorous in accepting features)   # rang 6


caps_cascade_classifier_list = ["cascade_16_gray.xml", "cascade_16_gray_equalized.xml", "cascade_18_gray.xml", "cascade_18_gray_equalized.xml", \
    "cascade_20_gray.xml", "cascade_20_gray_equalized.xml"]

caps_cascade_classifier_index = 0  ### enter the classifier index from the list

caps_cascade_classifier = cv2.CascadeClassifier(os.path.dirname(os.path.abspath(__file__)).replace(os.sep, "/") + "/classifiers/" + caps_cascade_classifier_list[caps_cascade_classifier_index])

cv2.namedWindow("video", cv2.WINDOW_AUTOSIZE)
cv2.namedWindow("trackbars", cv2.WINDOW_AUTOSIZE)
cv2.resizeWindow("trackbars", (320, 165))

def nothing(trackbar_var): pass

cv2.createTrackbar("percent", "trackbars", 1, 5, nothing)
cv2.createTrackbar("neighbours", "trackbars", 8, 10, nothing)
cv2.createTrackbar("min dim", "trackbars", 20, 30, nothing)
cv2.createTrackbar("max dim", "trackbars", 60, 80, nothing)

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

while True:
    ret, frame = capture.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # gray_frame = cv2.equalizeHist(gray_frame)

    scale_factor = 1 + cv2.getTrackbarPos("percent", "trackbars") / 100
    neighbours = cv2.getTrackbarPos("neighbours", "trackbars")
    min_dim = cv2.getTrackbarPos("min dim", "trackbars")
    max_dim = cv2.getTrackbarPos("max dim", "trackbars")
    detected_objects = caps_cascade_classifier.detectMultiScale(gray_frame, scale_factor, neighbours, minSize=(min_dim, min_dim), maxSize=(max_dim, max_dim))

    for (x, y, w, h) in detected_objects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)
    cv2.imshow("video", frame)
    if cv2.waitKey(1) == 27:
        break

capture.release()
cv2.destroyAllWindows()