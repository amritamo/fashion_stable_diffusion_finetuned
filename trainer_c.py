""" Adapted from https://github.com/Elvenson/stable-diffusion-keras-ft/blob/main/trainer.py
"""
from typing import Union
from sklearn.utils import validation

import tensorflow as tf
import keras_cv
import tensorflow.experimental.numpy as tnp
from keras_cv.models.stable_diffusion.noise_scheduler import NoiseScheduler
from keras_cv.models.stable_diffusion.stable_diffusion import MAX_PROMPT_LENGTH
from tensorflow import keras

from layers import LoraLayer


def load_sd_lora_layer(
        diffusion_model: tf.keras.Model,
        # diffusion_model: keras_cv.models.stable_diffusion,
        img_height: int,
        img_width: int,
        rank: int,
        alpha: float,
):
    for l in diffusion_model.layers:
        if "spatial_transformer" in l.name:

            l.transformer_block.attn1.to_q = LoraLayer(
                l.transformer_block.attn1.to_q,
                rank=rank,
                alpha=alpha,
                use_bias=False,
                trainable=True
            )
           
            
            l.transformer_block.attn1.to_k = LoraLayer(
                l.transformer_block.attn1.to_k,
                rank=rank,
                alpha=alpha,
                use_bias=False,
                trainable=True
            )


            l.transformer_block.attn1.to_v = LoraLayer(
                l.transformer_block.attn1.to_v,
                rank=rank,
                alpha=alpha,
                use_bias=False,
                trainable=True
            )
            l.transformer_block.attn1.out_proj = LoraLayer(
                l.transformer_block.attn1.out_proj,
                rank=rank,
                alpha=alpha,
                use_bias=False,
                trainable=True
            )

            l.transformer_block.attn2.to_q = LoraLayer(
                l.transformer_block.attn2.to_q,
                rank=rank,
                alpha=alpha,
                use_bias=False,
                trainable=True
            )
            l.transformer_block.attn2.to_k = LoraLayer(
                l.transformer_block.attn2.to_k,
                rank=rank,
                alpha=alpha,
                use_bias=False,
                trainable=True
            )
            l.transformer_block.attn2.to_v = LoraLayer(
                l.transformer_block.attn2.to_v,
                rank=rank,
                alpha=alpha,
                use_bias=False,
                trainable=True
            )
            l.transformer_block.attn2.out_proj = LoraLayer(
                l.transformer_block.attn2.out_proj,
                rank=rank,
                alpha=alpha,
                use_bias=False,
                trainable=True
            )
            

    # Forward pass to register new LoRA layers.
    latent = tf.random.normal([1, img_height // 8, img_width // 8, 4], 0, 1)
    t_emb = tf.random.normal([1, 320], 0, 1)
    context = tf.random.normal([1, MAX_PROMPT_LENGTH, 768], 0, 1)
    diffusion_model.predict_on_batch([latent, t_emb, context])

    # Freeze all layers except LoRA layers.
    for layer in diffusion_model._flatten_layers():
        lst_of_sublayers = list(layer._flatten_layers())

        if len(lst_of_sublayers) == 1:  # "leaves of the model"
            if layer.name in ["lora_a", "lora_b"]:
                layer.trainable = True
            else:
                layer.trainable = False


# Load LoRA layer for Stable Diffusion model
class Trainer(tf.keras.Model):
    # Adapted from https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py

    def __init__(
            self,
            diffusion_model: tf.keras.Model,
            vae: tf.keras.Model,
            noise_scheduler: NoiseScheduler,
            pretrained_ckpt: Union[str, None],
            mp: bool,
            ema=0.9999,
            max_grad_norm=1.0,
            lora=False,
            lora_rank=2,
            lora_alpha=4.,
            **kwargs,
    ):
        super().__init__(**kwargs)

        if lora and ema > 0.0:
            raise ValueError(
                "LoRA layer does not support exponential moving averaging learning (ema) for now."
            )

        self.diffusion_model = diffusion_model

        if lora:
            load_sd_lora_layer(self.diffusion_model, vae.input_shape[1], vae.input_shape[2],
                               rank=lora_rank, alpha=lora_alpha)

        if pretrained_ckpt is not None:
            self.diffusion_model.load_weights(pretrained_ckpt)
            print(
                f"Loading the provided checkpoint to initialize the diffusion model: {pretrained_ckpt}..."
            )

        self.vae = vae
        self.noise_scheduler = noise_scheduler

        self.vae.trainable = False
        self.mp = mp
        self.max_grad_norm = max_grad_norm

        if ema > 0.0:
            self.ema = tf.Variable(ema, dtype="float32")
            self.optimization_step = tf.Variable(0, dtype="int32")
            self.ema_diffusion_model = keras.models.clone_model(self.diffusion_model)
            self.do_ema = True
        else:
            self.do_ema = False

    def train_step(self, inputs):
        images = inputs["images"]
        encoded_text = inputs["encoded_text"]
        bsz = tf.shape(images)[0]

        with tf.GradientTape() as tape:
            # Project image into the latent space.
            latents = self.sample_from_encoder_outputs(self.vae(images, training=False))
            latents = latents * 0.18215

            # Sample noise that we'll add to the latents.
            noise = tf.random.normal(tf.shape(latents))

            # Sample a random timestep for each image.
            timesteps = tnp.random.randint(
                0, self.noise_scheduler.train_timesteps, (bsz,)
            )

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process).
            noisy_latents = self.noise_scheduler.add_noise(
                tf.cast(latents, noise.dtype), noise, timesteps
            )

            # Get the target for loss depending on the prediction type
            # just the sampled noise for now.
            target = noise  # noise_schedule.predict_epsilon == True

            # Can be implemented:
            # https://github.com/huggingface/diffusers/blob/9be94d9c6659f7a0a804874f445291e3a84d61d4/src/diffusers/schedulers/scheduling_ddpm.py#L352

            # Predict the noise residual and compute loss
            timestep_embeddings = tf.map_fn(
                lambda t: self.get_timestep_embedding(t), timesteps, dtype=tf.float32
            )
            timestep_embeddings = tf.squeeze(timestep_embeddings, 1)
            model_pred = self.diffusion_model(
                [noisy_latents, timestep_embeddings, encoded_text], training=True
            )
            loss = self.compiled_loss(target, model_pred)
            if self.mp:
                loss = self.optimizer.get_scaled_loss(loss)

        # Update parameters of the diffusion model.
        trainable_vars = self.diffusion_model.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        if self.mp:
            gradients = self.optimizer.get_unscaled_gradients(gradients)
        if self.max_grad_norm > 0.0:
            gradients = [tf.clip_by_norm(g, self.max_grad_norm) for g in gradients]
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # EMA.
        if self.do_ema:
            self.ema_step()

        metrics = {metric.name: metric.result() for metric in self.metrics}
        metrics["loss"] = loss


        
        return metrics

    
    def test_step(self, validation_data):
        print("validating")
        images = validation_data["images"]
        encoded_text = validation_data["encoded_text"]
        bsz = tf.shape(images)[0]

        latents = self.sample_from_encoder_outputs(self.vae(images, training=False))
        latents = latents * 0.18215

        # Sample noise that we'll add to the latents.
        noise = tf.random.normal(tf.shape(latents))

        # Sample a random timestep for each image.
        timesteps = tf.random.uniform(
            shape=(bsz,), minval=0, maxval=self.noise_scheduler.train_timesteps, dtype=tf.int32
        )

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process).
        noisy_latents = self.noise_scheduler.add_noise(
            tf.cast(latents, noise.dtype), noise, timesteps
        )

        # Get the target for loss depending on the prediction type
        target = noise

        # Predict the noise residual and compute loss
        timestep_embeddings = tf.map_fn(
            lambda t: self.get_timestep_embedding(t), timesteps, dtype=tf.float32
        )
        timestep_embeddings = tf.squeeze(timestep_embeddings, 1)
        model_pred = self.diffusion_model(
            [noisy_latents, timestep_embeddings, encoded_text], training=False  # Set training to False
        )
        val_loss = self.compiled_loss(target, model_pred)

        metrics = {metric.name: metric.result() for metric in self.metrics}
        metrics["val_loss"] = val_loss
        print("val_loss", val_loss)
        return metrics


    def get_timestep_embedding(self, timestep, dim=320, max_period=10000):
        # Taken from
        # # https://github.com/keras-team/keras-cv/blob/ecfafd9ea7fe9771465903f5c1a03ceb17e333f1/keras_cv/models/stable_diffusion/stable_diffusion.py#L481
        half = dim // 2
        log_max_period = tf.math.log(tf.cast(max_period, tf.float32))
        freqs = tf.math.exp(-log_max_period * tf.range(0, half, dtype=tf.float32) / half)
        args = tf.convert_to_tensor([timestep], dtype=tf.float32) * freqs
        embedding = tf.concat([tf.math.cos(args), tf.math.sin(args)], 0)
        embedding = tf.reshape(embedding, [1, -1])
        return embedding  # Excluding the repeat.

    def get_decay(self, optimization_step):
        value = (1 + optimization_step) / (10 + optimization_step)
        value = tf.cast(value, dtype=self.ema.dtype)
        return 1 - tf.math.minimum(self.ema, value)

    def ema_step(self):
        self.optimization_step.assign_add(1)
        self.ema.assign(self.get_decay(self.optimization_step))

        for weight, ema_weight in zip(
                self.diffusion_model.trainable_variables,
                self.ema_diffusion_model.trainable_variables,
        ):
            tmp = self.ema * (ema_weight - weight)
            ema_weight.assign_sub(tmp)

    def sample_from_encoder_outputs(self, outputs):
        mean, logvar = tf.split(outputs, 2, axis=-1)
        logvar = tf.clip_by_value(logvar, -30.0, 20.0)
        std = tf.exp(0.5 * logvar)
        sample = tf.random.normal(tf.shape(mean), dtype=mean.dtype)
        return mean + std * sample

    def save_weights(self, filepath, overwrite=True, save_format=None, options=None):
        # Overriding to help with the `ModelCheckpoint` callback.
        if self.do_ema:
            self.ema_diffusion_model.save_weights(
                filepath=filepath,
                overwrite=overwrite,
                save_format=save_format,
                options=options,
            )
        else:
            self.diffusion_model.save_weights(
                filepath=filepath,
                overwrite=overwrite,
                save_format=save_format,
                options=options,
            )

    
