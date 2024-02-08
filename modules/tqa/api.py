# https://github.com/VQAssessment/DOVER
# Using DOVER - 2211.04894
# Dover.pth - https://github.com/QualityAssessment/DOVER/releases/download/v0.1.0/DOVER.pth

import yaml
import torch
import numpy as np
from dover.models import DOVER
from dover.datasets import (
    UnifiedFrameSampler,
    spatial_temporal_view_decomposition,
)

class apiDover:
    def __init__(self, opt: str, device: str):
        """
        Wrapper for the DOVER Video Quality Assessment Model
            opt (`str`):
                Path to the DOVER yaml file
            device (`str`):
                Device to use for the model
        """

        with open(opt, "r") as f:
            self.opt = yaml.safe_load(f)
        # Model Loading

        self.evaluator = DOVER(**self.opt["model"]["args"]).to(device)
        self.evaluator.load_state_dict(
            torch.load(self.opt["test_load_path"], map_location=device)
        )

        self.mean, self.std = (
            torch.FloatTensor([123.675, 116.28, 103.53]),
            torch.FloatTensor([58.395, 57.12, 57.375]),
        )

        self.device = torch.device(device)
        self.temporal_samplers = {}

        self.dopt = self.opt["data"]["val-l1080p"]["args"]

        for stype, sopt in self.dopt["sample_types"].items():
            if "t_frag" not in sopt:
                # resized temporal sampling for TQE in DOVER
                self.temporal_samplers[stype] = UnifiedFrameSampler(
                    sopt["clip_len"], sopt["num_clips"], sopt["frame_interval"]
                )
            else:
                # temporal sampling for AQE in DOVER
                self.temporal_samplers[stype] = UnifiedFrameSampler(
                    sopt["clip_len"] // sopt["t_frag"],
                    sopt["t_frag"],
                    sopt["frame_interval"],
                    sopt["num_clips"],
                )

    def fuse_results(self, results: list):
        t, a = (results[1] - 0.1107) / 0.07355, (results[0] + 0.08285) / 0.03774
        x = t * 0.6104 + a * 0.3896
        return {
            "aesthetic": 1 / (1 + np.exp(-a)),
            "technical": 1 / (1 + np.exp(-t)),
            "overall": 1 / (1 + np.exp(-x)),
        }

    def preparedata(self, video_path: str):
        """
        Call this function to prepare the video before running it through the model, shouldn't touch the device
            video_path (`str`):
                Path to the video file
        """

        opt = self.opt
        dopt = self.dopt
        mean = self.mean
        std = self.std

        views, _ = spatial_temporal_view_decomposition(
            video_path, dopt["sample_types"], self.temporal_samplers
        )

        for k, v in views.items():
            num_clips = dopt["sample_types"][k].get("num_clips", 1)
            views[k] = (
                ((v.permute(1, 2, 3, 0) - mean) / std)
                .permute(3, 0, 1, 2)
                .reshape(v.shape[0], num_clips, -1, *v.shape[2:])
                .transpose(0, 1)
                #.to(device) <- disabled because this function (preparedata) is meant for preprocessing
            )

        return views

    def __call__(self, views) -> dict:
        """
        Call this function to run the video through the model
            views (`dict`):
                Dictionary of views, output of `preparedata`
        """
        evaluator = self.evaluator

        for k, v in views.items():
            views[k] = v.to(self.device)
            
        with torch.no_grad():
            results = evaluator(views, reduce_scores=False)
            results = [np.mean(l.cpu().numpy()) for l in results]

        return self.fuse_results(results)