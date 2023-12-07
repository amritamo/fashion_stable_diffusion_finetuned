"""
Adapted from https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py
"""

import os
from typing import Dict, Tuple

import keras_cv
import numpy as np
import pandas as pd
import tensorflow as tf
from keras_cv.models.stable_diffusion.clip_tokenizer import SimpleTokenizer
from keras_cv.models.stable_diffusion.text_encoder import TextEncoder
from sklearn.model_selection import train_test_split

PADDING_TOKEN = 49407
MAX_PROMPT_LENGTH = 77
AUTO = tf.data.AUTOTUNE
POS_IDS = tf.convert_to_tensor([list(range(MAX_PROMPT_LENGTH))], dtype=tf.int32)
DEFAULT_DATA_ARCHIVE = "resize_imgs"


class DatasetUtils:
    def __init__(
            self,
            dataset_archive: str = None,
            batch_size: int = 4,
            img_height: int = 256,
            img_width: int = 256,
    ):
        self.tokenizer = SimpleTokenizer()
        self.text_encoder = TextEncoder(MAX_PROMPT_LENGTH)
        self.augmenter = keras_cv.layers.Augmenter(
            layers=[
                keras_cv.layers.CenterCrop(img_height, img_width),
                keras_cv.layers.RandomFlip(),
                tf.keras.layers.Rescaling(scale=1.0 / 127.5, offset=-1),
            ]
        )

        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width

        data_path = DEFAULT_DATA_ARCHIVE
        np.random.seed(42)
        self.data_frame = pd.read_csv(os.path.join(data_path, "dataTr.csv"))
        self.data_frame = self.data_frame.sample(850, random_state = 42)
        self.data_frame["image_path"] = self.data_frame["image_path"].apply(
            lambda x: os.path.join(data_path, x)
        )

    def process_text(self, caption: str) -> np.ndarray:
        tokens = self.tokenizer.encode(caption)
        tokens = tokens + [PADDING_TOKEN] * (MAX_PROMPT_LENGTH - len(tokens))
        return np.array(tokens)

    def process_image(
            self, image_path: tf.Tensor, tokenized_text: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        image = tf.io.read_file(image_path)
        image = tf.io.decode_png(image, 3)
        image = tf.image.resize(image, (self.img_height, self.img_width))
        return image, tokenized_text

    def apply_augmentation(
            self, image_batch: tf.Tensor, token_batch: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        return self.augmenter(image_batch), token_batch

    def run_text_encoder(
            self, image_batch: tf.Tensor, token_batch: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        # Since the text encoder will remain frozen we can precompute it.
        return (
            image_batch,
            token_batch,
            self.text_encoder([token_batch, POS_IDS], training=False),
        )

    def prepare_dict(
            self,
            image_batch: tf.Tensor,
            token_batch: tf.Tensor,
            encoded_text_batch: tf.Tensor,
    ) -> Dict[str, tf.Tensor]:
        return {
            "images": image_batch,
            "tokens": token_batch,
            "encoded_text": encoded_text_batch,
        }

    def prepare_dataset(self, augmentation=True) -> tf.data.Dataset:
        
        all_captions = list(self.data_frame["caption"].values)
        tokenized_texts = np.empty((len(self.data_frame), MAX_PROMPT_LENGTH))
        for i, caption in enumerate(all_captions):
            tokenized_texts[i] = self.process_text(caption)

        image_paths = np.array(self.data_frame["image_path"])

        train_images, val_images, train_texts, val_texts = train_test_split(
            image_paths, tokenized_texts, test_size=0.1, random_state=42
        )


        train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_texts))
        train_dataset = train_dataset.shuffle(self.batch_size * 10)
        train_dataset = train_dataset.map(self.process_image, num_parallel_calls=AUTO).batch(
            self.batch_size
        )
        if augmentation:
            train_dataset = train_dataset.map(self.apply_augmentation, num_parallel_calls=AUTO)
        train_dataset = train_dataset.map(self.run_text_encoder, num_parallel_calls=AUTO)
        train_dataset = train_dataset.map(self.prepare_dict, num_parallel_calls=AUTO)

        val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_texts))
        val_dataset = val_dataset.shuffle(self.batch_size * 10)
        val_dataset = val_dataset.map(self.process_image, num_parallel_calls=AUTO).batch(
            self.batch_size
        )
        if augmentation:
            val_dataset = val_dataset.map(self.apply_augmentation, num_parallel_calls=AUTO)
        val_dataset = val_dataset.map(self.run_text_encoder, num_parallel_calls=AUTO)
        val_dataset = val_dataset.map(self.prepare_dict, num_parallel_calls=AUTO)
        return train_dataset.prefetch(AUTO), val_dataset.prefetch(AUTO)