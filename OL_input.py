import cv2
import threading


class WebcamVideoInput:

	def __init__(self, video_const):
		self.VC = video_const

		if self.VC.CAM_FROM_PHONE:
			self.stream = cv2.VideoCapture(self.VC.PHONE_CAM_URL)
		else:
			self.stream = cv2.VideoCapture(self.VC.CAM_SOURCE)

		self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, self.VC.CAM_WIDTH)
		self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, self.VC.CAM_HEIGHT)
		(self.ret, self.frame) = self.stream.read()
		self.started = False


	def get_frame(self): return self.frame


	def is_started(self):
		return self.started


	def start(self):
		if not self.started:
			self.started = True
			self.thread = threading.Thread(target=self.update, args=())
			self.thread.daemon = True
			self.thread.start()
			return self


	def read(self): return (self.ret, self.frame)


	def update(self):
		while self.started:
			(self.ret, self.frame) = self.stream.read()


	def stop(self):
		self.started = False

		if self.thread.is_alive():
			self.thread.join()
		self.stream.release()