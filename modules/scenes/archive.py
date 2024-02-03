import multiprocessing as mp
import tempfile
from scenedetect import ContentDetector, open_video, SceneManager
import cv2
import time

### start config ###
TARGET_FPS = 20
TARGET_RES = (512,512)
vidPath = "/home/diffusers/workspace_dep/dataproc/tomscott.mp4"
### end config ###

sceneManager = SceneManager()
sceneManager.add_detector(ContentDetector())

# #vidPath = inDataPipe.get() # str

# preproc_init = time.time()

# ## Video Pre-processing
# tempFile = tempfile.NamedTemporaryFile(suffix='.mp4')
# vidCap = cv2.VideoCapture(vidPath)
# vidOut = cv2.VideoWriter(tempFile.name, cv2.VideoWriter_fourcc(*'DIVX'), TARGET_FPS, TARGET_RES)
# frameSkip = int(vidCap.get(cv2.CAP_PROP_FPS)/TARGET_FPS)
# frameCount = 0

# while True:
#     ret = vidCap.grab()
#     if not ret:
#         break

#     if frameCount % frameSkip == 0:
#         resizedFrame = cv2.resize(
#             vidCap.retrieve()[1], TARGET_RES, interpolation=cv2.INTER_LANCZOS4
#         )
#         vidOut.write(resizedFrame)

#     frameCount += 1

# vidOut.release()
# vidCap.release()

# preproc_end = time.time()

# print("Preproc time: ", preproc_end - preproc_init)

scene_init = time.time()
## Scene Analyzing
sceneVideo = open_video(vidPath)
sceneManager.detect_scenes(sceneVideo)
sceneList = sceneManager.get_scene_list()
scene_end = time.time()
print(sceneList)
print("Scene time:", scene_end - scene_init)