import numpy as np

def get_change_points(video_feat, n_frame):
    """
    Create change points for the video.
    """

    change_points = list(range(0, n_frame, n_frame//5))

    temp_change_points = []
    # for idx in range(1, len(change_points) - 1):
    #     segment = [change_points[idx], change_points[idx + 1] - 1]
    #     if idx == len(change_points) - 2:
    #         segment = [change_points[idx], change_points[idx + 1]]

    #     temp_change_points.append(segment)
    # change_points = np.array(list(temp_change_points))

    for start, end in zip(change_points[:-1], change_points[1:]):
        segment = [start, end]
        temp_change_points.append(segment)
    change_points = np.array(list(temp_change_points))

    temp_n_frame_per_seg = []
    for change_points_idx in range(len(change_points)):
        n_frame = change_points[change_points_idx][1] - change_points[
            change_points_idx][0]
        temp_n_frame_per_seg.append(n_frame)
    n_frame_per_seg = np.array(list(temp_n_frame_per_seg))

    return change_points, n_frame_per_seg

def get_shot_importance(all_shot_bound, all_scores, all_nframes, all_positions, all_urls):
    """ Generate the automatic machine summary, based on the video shots; the frame importance scores; the number of
    frames in the original video and the position of the sub-sampled frames of the original video.
    :param list[np.ndarray] all_shot_bound: The video shots for all the -original- testing videos.
    :param list[np.ndarray] all_scores: The calculated frame importance scores for all the sub-sampled testing videos.
    :param list[np.ndarray] all_nframes: The number of frames for all the -original- testing videos.
    :param list[np.ndarray] all_positions: The position of the sub-sampled frames for all the -original- testing videos.
    :param list[np.ndarray] all_ulrs: The urls of all the videos.
    :return: A list containing the indices of the selected frames for all the -original- testing videos.
    """
    all_shots_infos = []
    for video_index in range(len(all_scores)):
        # Get shots' boundaries
        shot_bound = all_shot_bound[video_index]  # [number_of_shots, 2]
        frame_init_scores = all_scores[video_index]
        n_frames = all_nframes[video_index]
        positions = all_positions[video_index]
        url = all_urls[video_index]

        # Compute the importance scores for the initial frame sequence (not the sub-sampled one)
        frame_scores = np.zeros(n_frames, dtype=np.float32)
        if positions.dtype != int:
            positions = positions.astype(np.int32)
        if positions[-1] != n_frames:
            positions = np.concatenate([positions, [n_frames]])
        for i in range(len(positions) - 1):
            pos_left, pos_right = positions[i], positions[i + 1]
            if i == len(frame_init_scores):
                frame_scores[pos_left:pos_right] = 0
            else:
                frame_scores[pos_left:pos_right] = frame_init_scores[i]

        # Compute shot-level importance scores by taking the average importance scores of all frames in the shot
        shots_score = []
        shots_len = []
        for shot in shot_bound:
            shot_infos = {'change_point': shot,
                            'score': (frame_scores[shot[0]:shot[1] + 1].mean()).item(),
                            'length': shot[1] - shot[0] + 1}
            shots_score.append((frame_scores[shot[0]:shot[1] + 1].mean()).item())
            shots_len.append(shot[1] - shot[0] + 1)
        shot_infos = {"video_url": url,
                      "segments_delimitation": shot_bound,
                      "segments_score": shots_score,
                      "segments_length": shots_len}
        all_shots_infos.append(shot_infos)

    return all_shots_infos