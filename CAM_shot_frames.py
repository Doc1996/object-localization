import cv2
import time
import os


if not os.path.exists(os.path.dirname(os.path.abspath(__file__)).replace(os.sep, "/") + "/unprocessed images/"):
    os.makedirs(os.path.dirname(os.path.abspath(__file__)).replace(os.sep, "/") + "/unprocessed images/")

num_of_saved_frame = 1
cv2.namedWindow("video", cv2.WINDOW_AUTOSIZE)

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

while True:
    ret, frame = capture.read()

    cv2.imshow("video", frame)
    if cv2.waitKey(1) == ord("s"):
        cv2.imwrite(os.path.dirname(os.path.abspath(__file__)).replace(os.sep, "/") + "/unprocessed images/" + str(num_of_saved_frame) + ".jpg", frame)
        cv2.putText(frame, "Shot Frame " + str(num_of_saved_frame), (10, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.imshow("video", frame)
        cv2.waitKey(1)
        num_of_saved_frame += 1
        time.sleep(0.2)

capture.release()
cv2.destroyAllWindows()