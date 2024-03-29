import os
os.environ['HF_HOME'] = '/data1/cache/hf'

import jsonlines
from datasets import load_dataset
from datetime import datetime
from tqdm import tqdm
import json

MIN_SECS = 1
MAX = 2000000

curr_id = None
prior_data = None
timestamps = None

pb = tqdm(total=2000000)
count = 0

def cn(time):
    return (datetime.strptime(time, "%H:%M:%S.%f") - datetime(1900, 1, 1)).total_seconds()

cache = list()

with open('/data4/hdvila-100m.json') as f:
    for line in f:
        data = json.loads(line)
        if data["video_id"] != curr_id:
            # Save metadata from previous video
            if prior_data != None:
                prior_data: dict
                prior_data.pop("span_start")
                prior_data.pop("span_end")
                prior_data.pop("caption")
                prior_data.pop("clip_id")

                # loop through timestamps and get shortest one (above 1 sec)
                shortest = None
                for timestamp in timestamps:
                    try:
                        start = cn(timestamp[0])
                        end = cn(timestamp[1])

                        if end - start >= MIN_SECS:
                            if shortest == None:
                                shortest = timestamp
                            elif end - start < cn(shortest[1]) - cn(shortest[0]):
                                shortest = timestamp
                    except:
                        pass

                if shortest != None:
                    prior_data["span_start"] = shortest[0]
                    prior_data["span_end"] = shortest[1]
                    prior_data["caption"] = shortest[2]
                    cache.append(prior_data)
                else:
                    print("No valid timestamps found for video", prior_data["video_id"])

                pb.update(1)
                count += 1

                if count >= MAX:
                    break

            curr_id = data["video_id"]
            timestamps = []
            prior_data = data
        timestamps.append([data["span_start"], data["span_end"], data["caption"]])

print("saving")
with jsonlines.Writer(open("/data4/nwds2.jsonl", "w")) as writer:
    for data in cache:
        writer.write(data)