import argparse

import keras_cv
from PIL import Image
import numpy as np
import pandas as pd
from skimage import io
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.transforms import functional as F
import torch
from trainer import load_sd_lora_layer

from keras_cv.models.stable_diffusion.diffusion_model import DiffusionModel
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.transforms import functional as F


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
    parser.add_argument("--lr", default=1e-5, type=float)



    return parser.parse_args()

def read_csv_and_return_pairs(file_path):
    df = pd.read_csv(file_path)
    
    if 'caption' not in df.columns or 'image_path' not in df.columns:
        raise ValueError("CSV does not contain 'k' and 'v' columns")
    
    pairs = df.sample(5, random_state = 41)
    
    return pairs.to_dict('records') 



def preprocess_image(image):
    image = torch.tensor(image).unsqueeze(0)
    image = image.permute(0, 3, 1, 2) / 255.0
    return F.center_crop(image, (256, 256))



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

    
    print("here")
    model.diffusion_model.load_weights(args.checkpoint)

    print("Begin generating images")

    test_images = read_csv_and_return_pairs("resize_imgs/test_dataset.csv")
    real_ims = []
    for i in range(3):
      real_im = io.imread('resize_imgs/' + test_images[i]['image_path'])
      real_ims.append(real_im)
    real_images = torch.cat([preprocess_image(real_im) for real_im in real_ims])


    gen_images = []
    for i in range(3):
      prompt = test_images[i]['caption']

      im_i = model.text_to_image(
          prompt=prompt,
          num_steps=50,
          batch_size=1
      )

      gen_images.append(im_i)

    for idx, img in enumerate(gen_images):
        image_path = "store_testv/" + f"n{idx}" +f"lora{args.lora_rank}" + f"_lr{args.lr}" + f"_b3" +".png"
        Image.fromarray(img[0]).save(image_path)

    real_images = torch.cat([preprocess_image(real_im) for real_im in real_ims])
    base_imgs = torch.cat([preprocess_image(base_i[0]) for base_i in gen_images])
    fid = FrechetInceptionDistance(normalize=True)
    fid.update(real_images, real=True)
    fid.update(base_imgs, real=False)

    print(f"FID: {float(fid.compute())}")
    return float(fid.compute())
    

    


if __name__ == "__main__":
    args = parse_args()
    run(args)