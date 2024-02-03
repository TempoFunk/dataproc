# Blattman. 2023, Stable Video
# https://arxiv.org/abs/2311.15127 
# Page 3, Section 3.1
# Note: Fig. 2, Left Graph, Blue Line represent the ammount of separated clips obtained via metadata (wasn't clear enough)
# https://github.com/sayakpaul/single-video-curation-svd
# https://www.scenedetect.com/docs/latest/api.html

import multiprocessing as mp
from scenedetect import ContentDetector, open_video, SceneManager

def sceneRetrivalWorker(
    index: int,
    inDataPipe: mp.Queue,
    outDataPipe: mp.Queue
):
    sceneManager = SceneManager()
    sceneManager.add_detector(ContentDetector())
    
    while True:
        vidPath = inDataPipe.get() # str

        if vidPath is None:
            break

        sceneVideo = open_video(vidPath)
        sceneManager.detect_scenes(sceneVideo)
        sceneList = sceneManager.get_scene_list()
        
        outDataPipe.put(sceneList)