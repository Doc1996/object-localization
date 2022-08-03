from OL_constants import WebcamVideoConstants
from OL_input import WebcamVideoInput
from OL_output import WebcamVideoOutput
from OL_processing import WebcamVideoProcessing
from OL_trackbars import WebcamVideoTrackbars
from OL_algorithms import WebcamVideoAlgorithms


def main():
    video_const = WebcamVideoConstants()
    video_algorithms = WebcamVideoAlgorithms(video_const)

    if video_const.PROCESS_FROM_CAM:
        video_input = WebcamVideoInput(video_const)
        video_output = WebcamVideoOutput(video_const, video_input.get_frame())
        video_trackbars = WebcamVideoTrackbars(video_const)
        video_processing = WebcamVideoProcessing(video_const, video_algorithms, video_input, video_output, video_trackbars)
        video_processing.loop_video()
    else:
        video_processing = WebcamVideoProcessing(video_const, video_algorithms)
        video_processing.loop_images_in_folder()


if __name__ == "__main__":
    main()