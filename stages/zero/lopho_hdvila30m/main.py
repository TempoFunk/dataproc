import multiprocessing as mp
import pytube
import tempfile
import subprocess
from datasets import load_dataset
from datetime import timedelta
from tqdm import tqdm

"""
ffmpeg -i /home/diffusers/workspace_dep/test/video.mp4 -filter_complex 
"[0:v]trim=start=0:end=3,setpts=PTS-STARTPTS[va1]; 
[0:v]trim=start=3:end=6,setpts=PTS-STARTPTS[va2]" 
-map "[va1]" va1.mp4 -map "[va2]" va2.mp4
"""

DATASET_PATH = "/data4/hdvila1m_group/data/"
CORES_AVAIL = 250
MAX_VIDEOS = 1000000 # 1M # float("inf")

def extract_subclips(input_file, video_id, timestamps, output_path, cpu_core):

    filters = list()
    maps = list()

    clip_idx = 0

    for t_start, t_end, caption in timestamps:

        start_time = int(
            timedelta(
                hours=int(t_start[0:2]),
                minutes=int(t_start[3:5]),
                seconds=int(t_start[6:8]),
            ).total_seconds()
        )
        end_time = int(
            timedelta(
                hours=int(t_end[0:2]),
                minutes=int(t_end[3:5]),
                seconds=int(t_end[6:8])
            ).total_seconds()
        )

        filters.append(
            f"[0:v]trim=start={start_time}:end={end_time},setpts=PTS-STARTPTS[va{clip_idx}]"
        )

        mps = ["-map", f"[va{clip_idx}]", f"{output_path}/{video_id}.{clip_idx}.mp4"]
        maps.extend(mps)

        open(f"{output_path}/{video_id}.{clip_idx}.txt", "w").write(caption)

        clip_idx += 1
    # command = f'/home/diffusers/.local/bin/ffmpeg -i {input_file} -filter_complex "{filters}" {maps} -y'

    command = [
        "taskset", "-c", f"{cpu_core}",
        "ffmpeg", "-y",
        "-i", input_file,
        "-filter_complex", "; ".join(filters),
        *maps,
    ]

    try:
        completed_process = subprocess.run(
            command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    except subprocess.CalledProcessError as e:
        print("Error occurred:", e)
        print("Command output:")
        print(e.stdout)
        print("Command error output:")
        print(e.stderr)
    else:
        if completed_process.stderr:
            print("Command error output (if any):")
            print(completed_process.stderr)
    return


def worker(data: dict):
    # {"video_id": "PQxIjfO9ZhE", "url": "https://www.youtube.com/watch?v=PQxIjfO9ZhE", "timestamps": [["00:00:07.980", "00:00:16.000", "a diagram of a constellation with lines and dots"], ["00:00:16.000", "00:00:21.520", "a diagram of a constellation with lines and dots"]]}

    try:
        video_id = data["video_id"]
        url = data["url"]
        timestamps = data["timestamps"]
        worker_id = mp.current_process().name.split("-")[-1]
        print("Worker", worker_id)

        yt = pytube.YouTube(url)

        try:
            yt.check_availability()
        except Exception as e:
            print(f"Video Unavailable: {e}")
            return
        stream = yt.streams.filter(progressive=True).get_by_resolution("360p")

        with tempfile.TemporaryDirectory() as tempdir:
            stream.download(output_path=tempdir, filename=f"{video_id}.mp4")
            try:
                extract_subclips(
                    input_file=f"{tempdir}/{video_id}.mp4",
                    video_id=video_id,
                    timestamps=timestamps,
                    output_path=f"{DATASET_PATH}/",
                    cpu_core=worker_id,
                )
            except Exception as e:
                print(f"Error: {e}")
        return
    except Exception as e:
        print(f"Error: {e}")
        return


def main():
    dataset = load_dataset("chavinlo/hdvila1m_group", split="train", streaming=True)

    num_processes = CORES_AVAIL
    print("Number of processes:", num_processes)

    pool = mp.Pool(processes=num_processes)

    with tqdm(total=MAX_VIDEOS) as pbar:
        for _ in pool.imap_unordered(worker, dataset):
            pbar.update(1)

    pool.close()
    pool.join()


if __name__ == "__main__":
    main()
