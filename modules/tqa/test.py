import os

folder_path = "/home/diffusers/workspace_dep/dataproc/samples/videos"
file_paths = []

for root, dirs, files in os.walk(folder_path):
    for file in files:
        file_path = os.path.join(root, file)
        file_paths.append(file_path)

from api import apiDover
model = apiDover("/home/diffusers/workspace_dep/dataproc/modules/tqa/dover.yml", "cuda")

data = list()

for file_path in file_paths:
    print(file_path)
    data.append(
        (file_path, model.preparedata(file_path))
    )

datalen = len(data)
repeats = 10

import time

init_time = time.time()

scoreboard = []

for i in range(repeats):
    for name, entry in data:
        print(name, model(entry))

        if i == 0:
            scoreboard.append((name, model(entry)))

# sort
scoreboard.sort(key=lambda x: x[1]["overall"], reverse=True)

# print time taken per iteration
print((time.time() - init_time) / (datalen * repeats))

# print scoreboard fancy
for name, entry in scoreboard:
    print(name, entry)