# Fashion Synthesis Assistant Tool based on Stable Diffusion Model
## Contributors: Amrita Moturi, Mariana Vasquez, Maya Zheng

This project presents a fashion synthesis assistant tool that allows users to generate fashion images by simply entering a text prompt. The tool is developed based on the Stable Diffusion model, which is effective for text-to-image generation. By fine tuning the diffusion model on a product fashion dataset using low-rank adaptation (LoRA), our fine-tuned model can synthesize visually appealing images of clothing. The experiments demonstrate that fine-tuning a diffusion model on a fashion data can improve the quality of text-to-image generation tailored to our specific task and LoRA is a robust technique for the fine-tuning process. Access the resized images that we used in our implementation at https://drive.google.com/drive/folders/1x9SiWiCkKKO-5lWNpJSs4Ka4KpsXP_uF?usp=sharing. To run our experiments, clone this repository in Google Colab, add the following files to a folder "stable-diffusion-keras-ft" in Google Drive and follow the steps in KerasCV_loRA.ipynb

KerasCV Baseline Model: https://keras.io/guides/keras_cv/generate_images_with_stable_diffusion/

## Files
datasets_c.py is used to generate the training and validation datasets used for training. This was adapted from https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py

layers.py is the implementation of a LoRA layer, adapted from https://github.com/Elvenson/stable-diffusion-keras-ft/blob/main/layers.py

trainer_c.py includes the trainer used for training as well as the loading of LoRA layers into a diffusion model. 

finetune_c.py is the training script. 
Example use: python finetune_c.py --img_height 256 --img_width 256 --batch_size 4 --num_epochs 20 --ema 0 --lr 1e-04 --augmentation --lora --lora_rank 4 --lora_alpha 8 

inference.py 
Adapted from https://github.com/Elvenson/stable-diffusion-keras-ft/blob/main/inference.py 
Used to generate images from prompts for a specific model. 
Example use: python inference.py --prompt "a pink strapless top" --img_height 256 --img_width 256 --batch_size 1 --checkpoint 'ckpt_epochs_20_res_256_mp_False.h5' --num_steps 50

inference_with_fid.py 
This includes the implementation for FrechetInceptionDistance calculations for a specific model in comparison to the selected test data.



## Acknowledgements
Credits to https://github.com/Elvenson/stable-diffusion-keras-ft/tree/main 
