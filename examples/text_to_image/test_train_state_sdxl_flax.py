import os
from pprint import pprint

import flax.linen as nn
import jax.lib
import jax.numpy as jnp
import jaxlib.xla_extension
import optax
from flax.training import train_state

from diffusers import FlaxUNet2DConditionModel


def main():
    print("stabilityai/stable-diffusion-xl-base-1.0")
    unet, params = FlaxUNet2DConditionModel.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        subfolder="unet",
        dtype=jnp.float16,
    )
    params = unet.to_bf16(params)
    # (`jnp.ndarray`): (batch, channel, height, width) noisy inputs tensor
    sample = jnp.zeros((1, 4, 128, 128), dtype=jnp.float16)

    #  (`jnp.ndarray` or `float` or `int`): timesteps
    timesteps = jnp.zeros((1,), dtype=jnp.float16)

    # (batch_size, sequence_length, hidden_size) encoder hidden
    encoder_hidden_states = jnp.zeros((1, 1, 2048), dtype=jnp.float16)

    added_cond_kwargs = {
        "text_embeds": jnp.zeros((1, 1280), dtype=jnp.float16),
        "time_ids": jnp.zeros((1, 6), dtype=jnp.float16),
    }

    tabulate_fn = nn.tabulate(unet, jax.random.key(0))
    # print(
    #     FlaxUNet2DConditionModel().tabulate(
    #         jax.random.key(0),
    #         sample,
    #         timesteps,
    #         encoder_hidden_states,
    #         added_cond_kwargs,
    #         compute_flops=True,
    #         compute_vjp_flops=True,
    #     )
    # )
    # tabulate_fn = unet.tabulate(
    #     jax.random.key(0),
    #     sample,
    #     timesteps,
    #     encoder_hidden_states,
    #     added_cond_kwargs,
    #     compute_flops=True,
    #     compute_vjp_flops=True,
    # )
    tab = tabulate_fn(sample, timesteps, encoder_hidden_states, added_cond_kwargs)
    print(tab)
    adamw = optax.adamw(learning_rate=optax.constant_schedule(1e-5))
    try:
        state = init_fn(unet, params, adamw)
    except jaxlib.xla_extension.XlaRuntimeError as e:
        # RESOURCE_EXHAUSTED: Error allocating device buffer: Attempting to allocate 50.00M. That was not possible. There are 27.86M free.; (0x0x0_HBM0)
        print(e)


def init_fn(unet, params, adamw):
    return train_state.TrainState.create(apply_fn=unet.__call__, params=params, tx=adamw)


if __name__ == "__main__":
    main()
