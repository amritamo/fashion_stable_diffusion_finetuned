# Fashion Synthesis Assistant Tool based on Stable Diffusion Model
## Contributors: Amrita Moturi, Mariana Vasquez, Maya Zheng

This project presents a fashion synthesis assistant tool that allows users to generate fashion images by simply entering a text prompt. The tool is developed based on the Stable Diffusion model, which is effective for text-to-image generation. By fine tuning the diffusion model on a product fashion dataset using low-rank adaptation (LoRA), our fine-tuned model can synthesize visually appealing images of clothing. The experiments demonstrate that fine-tuning a diffusion model on a fashion data can improve the quality of text-to-image generation tailored to our specific task and LoRA is a robust technique for the fine-tuning process. Access the resized images that we used in our implementation at https://drive.google.com/drive/folders/1x9SiWiCkKKO-5lWNpJSs4Ka4KpsXP_uF?usp=sharing. To run our experiments, clone this repository in Google Colab, add the following files to a folder "stable-diffusion-keras-ft" in Google Drive and follow the steps in KerasCV_loRA.ipynb

KerasCV Baseline Model: https://keras.io/guides/keras_cv/generate_images_with_stable_diffusion/

##Files
datasets_c.py is used to generate the training and validation datasets used for training. This was adapted from https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py

layers.py is the implementation of a LoRA layer, adapted from https://github.com/Elvenson/stable-diffusion-keras-ft/blob/main/layers.py

trainer_c.py includes the trainer used for training as well as the loading of LoRA layers into a diffusion model. 




##Acknowledgements
Credits to https://github.com/Elvenson/stable-diffusion-keras-ft/tree/main 
