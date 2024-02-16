# Based on https://github.com/sayakpaul/single-video-curation-svd/blob/main/video_preprocessing_optical_flow_score.ipynb

import cv2
import numpy as np

class apiFarneback:
    def __init__(self) -> None:
        """
        Here goes the initialization
        """
        pass

    def preparedata(self, file_path: str, fps: int):
        """
        Here extract the frames of the video
        """
        cap = cv2.VideoCapture(file_path)
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        target_fps = min(frame_rate, fps)

        frames = []
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % int(frame_rate / target_fps) == 0:
                frames.append(frame)
            frame_count += 1

        cap.release()
        return frames

    def __call__(self, frames):
        """
        and here calculate the flow map, average, etc...
        """
        prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        flow_maps = []
        
        params = dict(
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )

        for frame in frames[1:]:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow_map = cv2.calcOpticalFlowFarneback(prev_gray, gray, flow=None, **params)
            flow_maps.append(flow_map)
            prev_gray = gray
            
        downscaled_maps = [
            cv2.resize(
                flow, (16, int(flow.shape[0] * (16 / flow.shape[1]))), interpolation=cv2.INTER_AREA
            ) 
            for flow in flow_maps
        ]
        average_flow_map = np.mean(np.array(downscaled_maps), axis=0)
        return np.mean(average_flow_map)