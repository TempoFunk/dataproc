import multiprocessing as mp
import subprocess
from tqdm import tqdm
import requests
import os
import json

"""
yt-dlp --format "best[height<=360]" https://www.youtube.com/watch?v=323t_f4gMgw --external-downloader ffmpeg --external-downloader-args "ffmpeg_i:-ss 00:10:11 -to 00:20:25"
"""

DATASET_PATH = "/mnt/disk1/data/hdvila1m_single_short/mp4"
CORES_AVAIL = 240
VID_RES = 360
MAX_VIDEOS = 1010000 # 1.01M # float("inf")


pattern = '"playabilityStatus":{"status":"ERROR","reason":"Video unavailable"'
def try_site(url):
    request = requests.get(url)
    return False if pattern in request.text else True


def worker(data: dict):
    try:
        video_id = data["video_id"]
        url = data["url"]
        span_start = data["span_start"]
        span_end = data["span_end"]
        caption = data["caption"]

        if not try_site(url):
            # print(f"Video Unavailable: {video_id}")
            return 0

        command = [
            "yt-dlp", "--format", f"best[height<={VID_RES}]",
            url,
            "--external-downloader", "ffmpeg",
            "--external-downloader-args", f"ffmpeg_i:-ss {span_start} -to {span_end}",
            "-o", f"{DATASET_PATH}/{video_id}.mp4",
        ]

        try:
            completed_process = subprocess.run(
                command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        except subprocess.CalledProcessError as e:
            # print("Error occurred:", e)
            # print("Command output:")
            # print(e.stdout)
            # print("Command error output:")
            # print(e.stderr)
            return 0
        
        open(f"{DATASET_PATH}/{video_id}.txt", "w").write(caption)
        return 1

    except Exception as e:
        # print(f"Error: {e}")
        return 0


def main():
    done = 0

    if os.path.exists(DATASET_PATH) is False:
        os.makedirs(DATASET_PATH)

    with open("/data4/nwds2.jsonl") as f:
        dataset = [json.loads(line) for line in f]
    
    loader = iter(dataset)

    num_processes = CORES_AVAIL
    print("Number of processes:", num_processes)

    pool = mp.Pool(processes=num_processes)

    with tqdm(total=MAX_VIDEOS) as pbar:
        for x in pool.imap_unordered(worker, loader):
            pbar.update(x)
            done += x
            if done >= MAX_VIDEOS:
                break

    pool.close()
    pool.join()


if __name__ == "__main__":
    main()
