import jsonlines
from datasets import load_dataset
from tqdm import tqdm

MAX = 1000000

og_dataset = load_dataset("TempoFunk/hdvila-100M", split="train", streaming=True)

#{'clip_id': 'KylzcWjpTaE.25.mp4', 'video_id': 'KylzcWjpTaE', 'url': 'https://www.youtube.com/watch?v=KylzcWjpTaE', 'span_start': '00:04:30.570', 'span_end': '00:04:37.730', 'caption': 'a group of women sitting on a couch with balloons'}

curr_id = None
prior_data = None
timestamps = None

pb = tqdm(total=MAX)
count = 0

with jsonlines.Writer(open("/data1/nwds.jsonl", "w")) as writer:
    for data in og_dataset:
        if data["video_id"] != curr_id:
            # Save metadata from previous video
            if prior_data != None:
                prior_data: dict
                prior_data.pop("span_start")
                prior_data.pop("span_end")
                prior_data.pop("caption")
                prior_data.pop("clip_id")

                prior_data["timestamps"] = timestamps
                writer.write(prior_data)
                pb.update(1)
                count += 1
                if count >= MAX:
                    break
            curr_id = data["video_id"]
            timestamps = []
            prior_data = data
        timestamps.append([data["span_start"], data["span_end"], data["caption"]])
