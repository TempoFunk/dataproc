import multiprocessing as mp
import json
import torch
import cv2
import threading
import os
import numpy as np
from datetime import timedelta
from tqdm import tqdm
from scenedetect import ContentDetector, SceneManager, FrameTimecode
from scenedetect.backends import VideoCaptureAdapter
from queue import Empty
from wrapt_timeout_decorator import *
from typing import List

GPUS_IDX = [4, 5, 6, 7]
DATASET_PATH = "/data4/hdvila1m_group/data/"
OUT_PATH = "/home/diffusers/workspace_dep/dataproc/stages/zero/lopho_hdvila30m/testdata" # "/data4/hdvila1m_group/capssimilarity/test1/"
CORES_AVAIL = 32

GPU_MAX_BATCH_SIZE = 32
GPU_CLIPVIP_CONF = "/home/diffusers/workspace_dep/dataproc/modules/clipvip_clean/cfg/pretrain_clipvip_base_16.json"
GPU_TASK_RESET = 1000

CPU_TASK_RESET = 400
CPU_FRAME_COUNT = 6

def gpu_worker(
        gpu_idx: int,
        input_pipe: mp.Queue,
        output_pipes: List[mp.Queue],
        finished: mp.Value
    ):
    from transformers import CLIPTokenizerFast
    from torchvision import transforms
    
    import sys
    sys.path.append('/home/diffusers/workspace_dep/dataproc/modules/clipvip_clean')
    from modelClipVip import VidCLIP
    from constants import norm_mean, norm_std, input_res
    from src.utils.load_save import load_state_dict_with_mismatch

    #print(f"[GPU {gpu_idx}] Starting...")

    ### Configuration Start ###
    
    config = json.load(open(GPU_CLIPVIP_CONF))

    model = VidCLIP(
        clip_config=config["clip_config"],
        clip_weights=config["clip_weights"],
        clip_vision_additional_config=config["clip_vision_additional_config"]
    )

    # I don't think sharing the tokenizer would be a good idea
    tokenizer = CLIPTokenizerFast.from_pretrained(config["clip_config"])

    normalize = transforms.Normalize(mean=norm_mean, std=norm_std)
    transform = transforms.Compose([
                transforms.Resize(input_res, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(input_res),
                normalize,
            ])
    
    load_state_dict_with_mismatch(model, config['e2e_weights_path'])
    device = torch.device(f"cuda:{gpu_idx}")
    model = model.to(device)
    model = model.to(torch.float16)

    #print(f"[GPU {gpu_idx}] Model loaded")

    ### Configuration End ###

    """
    {
        "video": numpy array (needs to be permuted),
        "text": str,
        "timestamps": tuple,
        "worker_id": int,
    }
    """

    def cpu_transfer_and_put(
            idx: int, # id of batch, not worker
            batch: list, 
            video_feature: torch.Tensor, 
            text_feature: torch.Tensor,
            output_pipes: List[mp.Queue]
        ):
        #print(f"[GPU {gpu_idx}] Transferring and putting {idx}/{len(batch)}")
        output_pipes[batch[idx]["worker_id"]].put(
            {
                "video": video_feature.to("cpu"),
                "text": text_feature.to("cpu"),
                "timestamps": batch[idx]["timestamps"],
                "worker_id": batch[idx]["worker_id"],
            }
        )
        #print(f"[GPU {gpu_idx}] Transferred and put {idx}/{len(batch)}")

    processes = list()
    tasks_done = 0
    active = True

    #print(f"[GPU {gpu_idx}] Starting with processing...")

    while active:
        if tasks_done > GPU_TASK_RESET:
            #print(f"[GPU {gpu_idx}] Resetting...")
            active = False
            with finished.get_lock():
                finished.value = 0
            break

        batch = list()

        while len(batch) < GPU_MAX_BATCH_SIZE:
            # if input_pipe.empty():
            #     break
            try:
                data = input_pipe.get(timeout=0.2)
                if data is None:
                    active = False
                    with finished.get_lock():
                        finished.value = 1
                    break
            except Empty:
                break
            batch.append(data)

        if len(batch) == 0:
            continue

        #print(f"[GPU {gpu_idx}] Processing batch of size {len(batch)} - Current Pipe length: {input_pipe.qsize()}")

        ## Video
        video_batch = torch.stack(
                        [
                            transform(
                                torch.from_numpy(
                                    x["video"] # (F, H, W, C)
                                ).permute(0, 3, 1, 2).float() / 255.
                            ) 
                        for x in batch]
                    ).to(device)
        video_batch = video_batch.to(torch.float16)
        
        ## Text
        text_batch = [x["text"] for x in batch]
        tokn_batch = tokenizer.batch_encode_plus(
                        text_batch,
                        max_length=config["max_txt_len"],
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt"
                    )
        txtids_batch = tokn_batch.input_ids.to(device)
        txtmsk_batch = tokn_batch.attention_mask.to(device)

        #print(f"[GPU {gpu_idx}] Data came from workers {list(set([x['worker_id'] for x in batch]))}")

        #print(f"[GPU {gpu_idx}] Forwarding...")

        ## Forward
        with torch.no_grad():
            video_feature = model.forward_video(video_batch)
            text_feature = model.forward_text(txtids_batch, txtmsk_batch)

        #print(f"[GPU {gpu_idx}] Forwarded")

        for idx in range(len(batch)):
            p = threading.Thread(target=cpu_transfer_and_put, args=(idx, batch, video_feature[idx], text_feature[idx], output_pipes))
            processes.append(p)
            p.start()

    for p in processes:
        p.join()

def gpu_worker_manager(
        gpu_idx: int,
        input_pipe: mp.Queue,
        output_pipes: List[mp.Queue]
    ):
    
    finished = mp.Value('i', 0)
    
    while True:
        if finished.value == 1:
            break

        p = mp.Process(target=gpu_worker, args=(gpu_idx, input_pipe, output_pipes, finished))
        p.start()
        p.join()

def cpu_worker(
        cpu_idx: int, # cpu_idx = core number = worker number
        paths_pipe: mp.Queue,
        input_pipe: mp.Queue, # global one, as in input for the gpu
        output_pipe: mp.Queue,
        progress_pipe: mp.Queue,
        finished: mp.Value
    ):

    #print(f"[CPU {cpu_idx}] Starting...")

    frame_count = CPU_FRAME_COUNT

    active = True
    tasks_done = 0

    #print(f"[CPU {cpu_idx}] Starting with processing...")

    while active:
        if tasks_done > CPU_TASK_RESET:
            active = False
            with finished.get_lock():
                finished.value = 0
            break

        if paths_pipe.empty():
            continue

        path = paths_pipe.get()

        if path is None:
            active = False
            input_pipe.put(None) # Signal the GPUs to stop
            with finished.get_lock():
                finished.value = 1
            break

        #print(f"[CPU {cpu_idx}] Task {tasks_done} - Opening video {os.path.basename(path)}")
        sceneManager = SceneManager()
        sceneManager.add_detector(ContentDetector())
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        sceneManager.detect_scenes(VideoCaptureAdapter(cap))
        sceneList = sceneManager.get_scene_list()

        if len(sceneList) == 0:
            #print(f"[CPU {cpu_idx}] No scenes detected")
            continue
        else:
            #print(f"[CPU {cpu_idx}] {len(sceneList)} scenes detected")
            pass

        subclips = list()

        for scene in sceneList:
            start_time, end_time = scene
            start_time: FrameTimecode; end_time: FrameTimecode
            start_frame = start_time.get_frames()
            end_frame = end_time.get_frames()
            step_size = max(1, (end_frame - start_frame) // frame_count)
            frame_indices = list(range(start_frame, end_frame, step_size))[:frame_count]

            while len(frame_indices) != frame_count:
                frame_indices.append(frame_indices[-1])
            
            frames = list()
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    # frame shape is (H, W, C)
                else:
                    #print("Frame not read")
                    break

            subclip = np.stack(frames)
            subclips.append({
                "frames": subclip,
                "timestamps": (start_frame, end_frame)
                }
            )

            #print(f"[CPU {cpu_idx}] Subclip {len(subclips)} done")

        caption = open(f"{path[:-4]}.txt").read()
        n_subclips = len(subclips)

        for subclip_idx in range(n_subclips):
            input_pipe.put(
                {
                    "video": subclips[subclip_idx]["frames"],
                    "text": caption,
                    "timestamps": subclips[subclip_idx]["timestamps"],
                    "worker_id": cpu_idx,
                }
            )

        #print(f"[CPU {cpu_idx}] Data sent to GPUs")

        del subclips

        video_features = list()
        text_feature = None # Same across all

        for _ in range(n_subclips):
            data = output_pipe.get()
            video_features.append(
                {
                    "video": data["video"],
                    "timestamps": data["timestamps"],
                }
            )
            text_feature = data["text"]
            #print(f"[CPU {cpu_idx}] Recieved {len(video_features)}/{n_subclips} from GPUs")
        
        #print(f"[CPU {cpu_idx}] Data received from GPUs")

        highest_similarity = 0
        highest_similarity_idx = 0

        for vidfeat_idx in range(len(video_features)):
            #print(f"[CPU {cpu_idx}] Calculating similarity {vidfeat_idx}/{len(video_features)}, video features shape: {video_features[vidfeat_idx]['video'].shape}, text feature shape: {text_feature.shape}")
            similarity = torch.cosine_similarity(
                video_features[vidfeat_idx]["video"].unsqueeze(0),
                text_feature.unsqueeze(0)
            )
            if similarity > highest_similarity:
                highest_similarity = similarity
                highest_similarity_idx = vidfeat_idx

        #print(f"[CPU {cpu_idx}] Highest similarity: {highest_similarity}")

        preffered_clip = video_features[highest_similarity_idx]["timestamps"]
        frame_indices = list(range(preffered_clip[0], preffered_clip[1]))

        frames = list()

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
            else:
                #print("Frame not read")
                break

        cap.release()

        #print(f"[CPU {cpu_idx}] Writing video {os.path.basename(path)}")

        out = cv2.VideoWriter(
            f"{OUT_PATH}/{os.path.basename(path)}",
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (frames[0].shape[1], frames[0].shape[0])
        )

        for frame in frames:
            out.write(frame)

        out.release()

        open(f"{OUT_PATH}/{os.path.basename(path)[:-4]}.txt", "w").write(caption)

        tasks_done += 1
        progress_pipe.put(1)

        #print(f"[CPU {cpu_idx}] Task done {tasks_done}/{CPU_TASK_RESET}")

    #print(f"[CPU {cpu_idx}] Finished")

def cpu_worker_manager(
        cpu_idx: int, # cpu_idx = core number = worker number
        paths_pipe: mp.Queue,
        input_pipe: mp.Queue, # global one, as in input for the gpu
        output_pipe: mp.Queue,
        progress_pipe: mp.Queue,
    ):

    finished = mp.Value('i', 0)

    while True:
        if finished.value == 1:
            break

        p = mp.Process(target=cpu_worker, args=(cpu_idx, paths_pipe, input_pipe, output_pipe, progress_pipe, finished))
        p.start()
        p.join()

def scan_and_filter(path: str, ext: str = ".mp4"):
    opposite_ext = ".txt" if ext == ".mp4" else ".mp4"
    for x in os.scandir(path):
        if x.is_file() and x.name.endswith(ext) and os.path.exists(f"{x.path[:-4]}{opposite_ext}"):
            yield x.path

def scan_and_count(path: str, ext: str = ".mp4"):
    count = 0
    opposite_ext = ".txt" if ext == ".mp4" else ".mp4"
    for x in os.scandir(path):
        if x.is_file() and x.name.endswith(ext) and os.path.exists(f"{x.path[:-4]}{opposite_ext}"):
            count += 1
    return count

def scan_worker(path: str, paths_pipe: mp.Queue, pipe_limit: int):
    gen = scan_and_filter(path)
    while True:
        if paths_pipe.qsize() < pipe_limit:
            try:
                paths_pipe.put(next(gen))
            except StopIteration:
                break

def pb_worker(progress_pipe: mp.Queue, total: int):
    pbar = tqdm(total=total)
    while True:
        progress = progress_pipe.get()
        if progress is None:
            break
        pbar.update(progress)
    pbar.close()

def main():
    #print("Number of worker processes:", CORES_AVAIL)
    #print("Number of GPUs available:", len(GPUS_IDX))

    #print(f"Scanning {DATASET_PATH}...")
    total = scan_and_count(DATASET_PATH)
    #print(f"Total: {total}")

    ## Pipes
    paths_pipe = mp.Queue() # Paths to files

    input_pipe = mp.Queue() # Single, global pipeline
    
    output_pipes = list()
    for idx in range(CORES_AVAIL):
        output_pipes.append(mp.Queue())

    progress_pipe = mp.Queue()

    ## Workers
        
    ### Scanner
    scan_process = mp.Process(target=scan_worker, args=(DATASET_PATH, paths_pipe, (CORES_AVAIL*4),))

    ### CPUs
    cpu_processes = list()
    for idx in range(CORES_AVAIL):
        p = mp.Process(target=cpu_worker_manager, args=(idx, paths_pipe, input_pipe, output_pipes[idx], progress_pipe,))
        cpu_processes.append(p)
    
    ### GPUs
    gpu_processes = list()
    for idx in GPUS_IDX:
        p = mp.Process(target=gpu_worker_manager, args=(idx, input_pipe, output_pipes))
        gpu_processes.append(p)

    ### Progress Bar
    pb_process = mp.Process(target=pb_worker, args=(progress_pipe, total))

    ## Start
    #print("Starting Progress Bar")
    pb_process.start()
    #print("Starting Scanner")
    scan_process.start()
    #print("Starting CPUs")
    for p in cpu_processes:
        p.start()
    #print("Starting GPUs")
    for p in gpu_processes:
        p.start()
    #print("All processes started")

    ## Join
    scan_process.join()
    for p in cpu_processes:
        p.join()
    for p in gpu_processes:
        p.join()
    progress_pipe.put(None)
    pb_process.join()
    #print("All workers joined")

if __name__ == "__main__":
    main()