"""
Adapted from https://github.com/Elvenson/stable-diffusion-keras-ft/blob/main/inference.py
"""
import argparse

import keras_cv
from PIL import Image

from trainer import load_sd_lora_layer

from keras_cv.models.stable_diffusion.diffusion_model import DiffusionModel

MAX_PROMPT_LENGTH = 77

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import argparse

import tensorflow as tf

from keras_cv.models.stable_diffusion.diffusion_model import DiffusionModel
from keras_cv.models.stable_diffusion.image_encoder import ImageEncoder
from keras_cv.models.stable_diffusion.noise_scheduler import NoiseScheduler
from tensorflow.keras import mixed_precision


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to do inference a Stable Diffusion model."
    )

    parser.add_argument("--prompt", default=None, type=str)
    parser.add_argument("--img_height", default=256, type=int)
    parser.add_argument("--img_width", default=256, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--num_steps", default=50, type=int)
    parser.add_argument("--checkpoint", default=None, type=str, help="Model checkpoint for loading model weights.")
    parser.add_argument("--lora", action="store_true", help="Whether to load loRA layer.")
    parser.add_argument("--lora_rank", default=4, type=int)
    parser.add_argument("--lora_alpha", default=4, type=float)
    parser.add_argument("--log_dir", default=None, type=str)
    parser.add_argument("--exp", default=None, type=str)

    return parser.parse_args()

def run(args):
    
    
    model = keras_cv.models.StableDiffusion(
        img_height=256, img_width=256
    )
    
    if args.lora:
        print("Loading LoRA layer")
        
        load_sd_lora_layer(
          model.diffusion_model, 
          args.img_height, 
          args.img_width,
          rank=args.lora_rank, 
          alpha=args.lora_alpha
        )

    
    model.diffusion_model.load_weights(args.checkpoint)

    print("Begin generating images")
    images = model.text_to_image(
        prompt=args.prompt,
        num_steps=args.num_steps,
        batch_size=args.batch_size
    )

    for idx, img in enumerate(images):
        image_path = f"ttest-{idx}_lora4-lr_1e6-b3-greenskirt.png"
        Image.fromarray(img).save(image_path)


if __name__ == "__main__":
    args = parse_args()
    run(args)
