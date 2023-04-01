import cv2
import numpy as np

def preprocess(input, downsampled_fps):
    """
    Preprocess the video to get the downsampled frames.
    :param input: the path of the video to be processed
    :param downsampled_fps: the fps of the downsampled video
    return: a dict with keys: video_name, video_frames, n_frame, picks
    """
    if not isinstance(input, str):
        raise TypeError(f'input should be a str,'
                        f'  but got {type(input)}')
    frames = []
    picks = []
    cap = cv2.VideoCapture(input)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_multiple = original_fps//downsampled_fps
    # self.frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_idx = 0
    while (cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_multiple == 0:
            frames.append(frame)
            picks.append(frame_idx)
        frame_idx += 1
    n_frame = frame_idx

    result = {
        'video_name': input,
        'video_frames': np.array(frames),
        'n_frame': n_frame,
        'picks': np.array(picks)
    }
    return result