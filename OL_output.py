import cv2
import threading


class WebcamVideoOutput:

	def __init__(self, video_const, processed_frame, processed_first_mask_frame=None, processed_second_mask_frame=None):
		self.VC = video_const
		self.frame = processed_frame
		self.first_mask_frame = processed_first_mask_frame
		self.second_mask_frame = processed_second_mask_frame
		self.started = False


	def set_frame(self, processed_frame):
		self.frame = processed_frame


	def set_first_mask_frame(self, processed_first_mask_frame):
		self.first_mask_frame = processed_first_mask_frame


	def set_second_mask_frame(self, processed_second_mask_frame):
		self.second_mask_frame = processed_second_mask_frame


	def is_started(self): return self.started


	def start(self):
		if not self.started:
			self.started = True
			self.thread = threading.Thread(target=self.update, args=())
			self.thread.daemon = True
			self.thread.start()
			return self


	def update(self):
		while self.started:
			cv2.imshow(self.VC.VIDEO_WINDOW_NAME, self.frame)

			if self.first_mask_frame is not None and self.second_mask_frame is not None:
				if self.VC.SHOW_MASK_VIDEOS:
					cv2.imshow(self.VC.FIRST_MASK_VIDEO_WINDOW_NAME, self.first_mask_frame)
					cv2.imshow(self.VC.SECOND_MASK_VIDEO_WINDOW_NAME, self.second_mask_frame)
				else:
					try:
						cv2.destroyWindow(self.VC.FIRST_MASK_VIDEO_WINDOW_NAME)
						cv2.destroyWindow(self.VC.SECOND_MASK_VIDEO_WINDOW_NAME)
					except:
						pass

			if cv2.waitKey(self.VC.MIN_FRAME_TIME) == self.VC.KEY_ESC:
				self.started = False


	def stop(self):
		self.started = False

		if self.thread.is_alive():
			self.thread.join()
			self.stream.release()