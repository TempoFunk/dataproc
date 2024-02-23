import json
import torch
import decord
from modelClipVip import VidCLIP
from src.utils.load_save import load_state_dict_with_mismatch
from transformers import CLIPTokenizerFast
from torchvision import transforms
from scenedetect import ContentDetector, open_video, SceneManager, FrameTimecode

config = json.load(open("/home/diffusers/workspace_dep/dataproc/modules/clipvip/clipvip_clean/cfg/pretrain_clipvip_base_32.json"))

model = VidCLIP(
    clip_config=config["clip_config"],
    clip_weights=config["clip_weights"],
    clip_vision_additional_config=config["clip_vision_additional_config"]
)

tokenizer = CLIPTokenizerFast.from_pretrained(config["clip_config"])

norm_mean=(0.48145466, 0.4578275, 0.40821073)
norm_std=(0.26862954, 0.26130258, 0.27577711)
input_res=(224, 224)

normalize = transforms.Normalize(mean=norm_mean, std=norm_std)
transform = transforms.Compose([
            transforms.Resize(input_res, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(input_res),
            normalize,
        ])

load_state_dict_with_mismatch(model, config['e2e_weights_path'])
device = torch.device("cuda:4")
model.to(device)

PAIR_PATH = "/home/diffusers/workspace_dep/dataproc/modules/clipvip/samples/beZQsCwuBl0.30"

vr = decord.VideoReader(f"{PAIR_PATH}.mp4", ctx=decord.cpu(0))

# PySceneDetect
sceneManager = SceneManager()
sceneManager.add_detector(ContentDetector())
sceneVideo = open_video(f"{PAIR_PATH}.mp4")
sceneManager.detect_scenes(sceneVideo)
sceneList = sceneManager.get_scene_list()

subclips = []
subclips_indices = []

print(sceneList)

frame_count = 12

for scene in sceneList:
    start_time, end_time = scene
    start_time: FrameTimecode; end_time: FrameTimecode
    start_frame = start_time.get_frames()
    end_frame = end_time.get_frames()

    step_size = max(1, (end_frame - start_frame) // frame_count)
    frame_indices = list(range(start_frame, end_frame, step_size))[:frame_count]
    print(frame_indices, len(frame_indices), step_size, start_frame, end_frame)

    while len(frame_indices) != frame_count:
        frame_indices.append(frame_indices[-1])

    subclip = torch.from_numpy(
                vr.get_batch(frame_indices).asnumpy()
                ).permute(0, 3, 1, 2).float() / 255.
    
    subclip = transform(subclip)
    subclips.append(subclip)
    subclips_indices.append({
        "start": round(start_time.get_seconds(), 2),
        "end": round(end_time.get_seconds(), 2)
    })

caption = open(f"{PAIR_PATH}.txt").read()

# given the length, extract 12 frames from the video
# total_frame_num = len(vr)
# frame_idx = list(range(0, total_frame_num, total_frame_num//12))

# video = torch.from_numpy(vr.get_batch(frame_idx).asnumpy())
# video = video.permute(0, 3, 1, 2).float() / 255.
# video = transform(video)
video = torch.stack(subclips)

batch_enc = tokenizer.batch_encode_plus(
            [caption * 3],
            max_length=config["max_txt_len"],
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

text_input_ids = batch_enc.input_ids  # (B, L)
text_input_mask = batch_enc.attention_mask  # (B, L)

print("Shapes: ", video.shape, text_input_ids.shape, text_input_mask.shape)

video = video.to(device)
text_input_ids = text_input_ids.to(device)
text_input_mask = text_input_mask.to(device)

# x = model(
#     video = video,
#     text_input_ids = text_input_ids,
#     text_input_mask = text_input_mask
# )

import time

init_time = time.time()

video_feature = model.forward_video(video)
text_feature = model.forward_text(text_input_ids, text_input_mask)

print("Time taken: ", time.time() - init_time)

print(video_feature.shape, text_feature.shape)

print(caption)

for x in range(len(subclips_indices)):
    # There are multiple videos and only one text, compare them all
    similarity = torch.nn.functional.cosine_similarity(video_feature[x].unsqueeze(0), text_feature).item()
    print(round(similarity, 2), " - ", subclips_indices[x])


# for i in range(len(possible_texts)):
#     similarity = torch.nn.functional.cosine_similarity(video_feature, text_feature[i].unsqueeze(0)).item()
#     print(possible_texts[i], similarity)

# similarity = torch.nn.functional.cosine_similarity(x['vis_features'], x['text_features']).item()

# print(similarity)