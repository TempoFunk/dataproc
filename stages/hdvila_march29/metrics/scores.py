import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['HF_HOME'] = '/data1/cache/hf'

from functools import partial
import random
import math
import json
import datasets
import torch
from torch.utils.data import DataLoader
from PIL import Image, ImageOps
import numpy as np
from tqdm.auto import tqdm
from safetensors.torch import save_file
import accelerate
from video_utils import read_video_cv2, read_video_mediapy
import torch.multiprocessing as mp

fps = 24
width = 576
height = 320
input_dir = '/data4/hdvila1m_single_short/mp4'
model_path = "/home/diffusers/workspace_dep/dataproc/modules/tqa/dover.yml"

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

accelerator = accelerate.Accelerator()

vae: AutoencoderKL = AutoencoderKL.from_pretrained(model, subfolder = 'vae').eval().requires_grad_(False)
vae = vae.to(device = accelerator.device)
text_encoder: CLIPTextModel = CLIPTextModel.from_pretrained(model, subfolder = 'text_encoder').eval().requires_grad_(False)
text_encoder = text_encoder.to(device = accelerator.device)
tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(model, subfolder = 'tokenizer')


@torch.no_grad()
def collate_fn(
        batch: list[dict[str, str]] | list[str],
        min_frames: int,
        max_frames: int,
        output_dir_stats: str,
        width: int,
        height: int,
        fps: int,
        caption: bool = True,
        video: bool = True,
        raw: bool = False,
        raw_video_ext: str = '.mp4'
) -> dict[str, torch.Tensor]:
    data = []
    for s in batch:
        video_id = os.path.splitext(os.path.basename(s))[0]
        exists = False
        for odir in output_dirs:
            if os.path.isfile(os.path.join(odir, video_id + '.safetensors')):
                exists = True
                continue
        if exists:
            data.append({ 'video': None })
            continue
        if raw:
            video_path = os.path.splitext(s)[0] + raw_video_ext
            with open(s, 'r') as f:
                p = f.read().strip()
            s = { 'id': video_id, 'video': video_path, 'caption': p }
        dataframe = { 'id': s['id'] }
        if caption:
            dataframe['caption'] = s['caption']
        if video:
            x = read_video_cv2(s['video'])
            if x is None:
                dataframe['video'] = None
                data.append(dataframe)
                continue
            video_frames = x[0]
            video_metadata = x[1]
            vfps = video_metadata.fps
            # resample to target fps
            step = max(1, round(vfps / fps))
            video_frames = video_frames[::step]
            num_frames = len(video_frames)
            if num_frames >= min_frames:
                video_slice = 0
                if max_frames > 0 and max_frames < num_frames:
                    # random slice of total required frames
                    video_slice = random.randint(0, num_frames - max_frames)
                    video_frames = video_frames[video_slice:video_slice + max_frames]
                with open(os.path.join(output_dir_stats, video_id + '.json'), 'w') as f:
                    json.dump({
                        'id': video_id,
                        'num_images': video_metadata.num_images,
                        'shape': video_metadata.shape,
                        'fps': video_metadata.fps,
                        'caption': s['caption'],
                        'slice': [video_slice*step, video_slice*step + min(max_frames, num_frames)*step],
                        'frame_stepping': step,
                        'num_frames': min(max_frames, num_frames)
                    }, f)
                # resize
                video_frames = np.array([
                        np.asarray(ImageOps.fit(
                                Image.fromarray(x),
                                (width, height),
                                method = Image.Resampling.LANCZOS
                        ))
                        for x in video_frames
                ])
                # f h w c -> f c h w
                video_frames = torch.from_numpy(video_frames).permute(0, 3, 1, 2)
                video_frames = video_frames.to(torch.float32).mul(2).div(255).sub(1)
                dataframe['video'] = video_frames
            else:
                dataframe['video'] = None
        data.append(dataframe)
    return data

def worker_init_fn(_: int):
    wseed = torch.initial_seed() % (2**32-2) # max val for random 2**32 - 1
    random.seed(wseed)
    np.random.seed(wseed)

#ds = datasets.load_dataset(input_dir, streaming = False, split = datasets.Split.TRAIN, trust_remote_code = True)
#ds = sorted(ds, key = lambda d: d['id'])
ds: list[str] = sorted([ os.path.join(input_dir, x) for x in os.listdir(input_dir) if x.endswith('.txt') ])

dataloader = DataLoader(
        ds,
        batch_size = 1,
        worker_init_fn = worker_init_fn,
        collate_fn = partial(
                collate_fn,
                min_frames = min_frames,
                max_frames = max_frames,
                output_dir_stats = output_dir_stats,
                width = width,
                height = height,
                fps = fps,
                video = True,
                caption = True,
                raw = True,
                raw_video_ext = '.mp4'
        ),
        num_workers = 2,
        pin_memory = False,
        shuffle = False
)

vae, text_encoder, dataloader = accelerator.prepare(vae, text_encoder, dataloader)

@torch.no_grad()
def encode_batch(
        output_dir: str,
        batch: list[dict[str, str | torch.Tensor]],
        batch_size: int = 16,
) -> None:
    output_data = []
    for b in batch:
        if b['video'] is None:
            continue
        videoid = b['id']
        frames = b['video'].to(device = vae.device)
        prompt = b['caption']
        means = []
        stds = []
        tokens = tokenizer(
                prompt,
                padding = 'max_length',
                max_length = 77,
                truncation = True,
                return_tensors = 'pt'
        ).input_ids.to(
                device = text_encoder.device,
                memory_format = torch.contiguous_format
        )
        with accelerator.autocast():
            text_embed = text_encoder(tokens)
        text_encoding = text_embed.last_hidden_state.to(memory_format = torch.contiguous_format, dtype = torch.float32, device = 'cpu')
        for sb in torch.split(frames, batch_size):
            with accelerator.autocast():
                y = vae.encode(sb).latent_dist
            means.append(y.mean)
            stds.append(y.std)
        means = torch.cat(means).to(memory_format = torch.contiguous_format, dtype = torch.float32, device = 'cpu')
        stds = torch.cat(stds).to(memory_format = torch.contiguous_format, dtype = torch.float32, device = 'cpu')
        output_data.append({'data': {'mean': means, 'std': stds, 'text_encoding': text_encoding}, 'path': os.path.join(output_dir, videoid + '.safetensors')})
    return output_data

if accelerator.is_main_process:
    for d in output_dirs:
        os.makedirs(d, exist_ok = True)
    os.makedirs(output_dir_stats, exist_ok = True)

save_queue = mp.Queue()
output_queue = mp.Queue()
def save_worker(save_queue: mp.Queue, output_queue: mp.Queue):
    while True:
        x = save_queue.get()
        if x is None:
            output_queue.put(None)
            return
        for y in x:
            save_file(y['data'], y['path'])
            output_queue.put(y['path'])
save_workers: list[mp.Process] = []
num_save_workers = 2
for _ in range(num_save_workers):
    p = mp.Process(target = save_worker, args = (save_queue, output_queue))
    p.start()
    save_workers.append(p)

i = 0
pbar = tqdm(dataloader, desc = str(accelerator.device.index), dynamic_ncols = True, position = accelerator.device.index)
accelerator.wait_for_everyone()
for k,b in enumerate(pbar):
    output_dir = output_dirs[i]
    i = (i + 1) % len(output_dirs)
    output_data = encode_batch(
            output_dir = output_dir,
            batch = b,
            batch_size = 32
    )
    save_queue.put(output_data)
    pbar.set_postfix_str(f'{save_queue.qsize(), output_queue.qsize()}')

for p in save_workers:
    save_queue.put(None)
save_queue.put(None)
for p in save_workers:
    p.join()


