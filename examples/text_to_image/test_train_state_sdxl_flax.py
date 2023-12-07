import os

import jax.lib
import jaxlib.xla_extension
import optax
from flax.training import train_state

from diffusers import FlaxUNet2DConditionModel


def main():
    jax.profiler.start_trace("prf")
    unet, params = FlaxUNet2DConditionModel.from_pretrained(
        "pcuenq/stable-diffusion-xl-base-1.0-flax",
        subfolder="unet",
    )
    adamw = optax.adamw(learning_rate=optax.constant_schedule(1e-5))
    try:
        state = init_fn(unet, params, adamw)
    except jaxlib.xla_extension.XlaRuntimeError as e:
        # RESOURCE_EXHAUSTED: Error allocating device buffer: Attempting to allocate 50.00M. That was not possible. There are 27.86M free.; (0x0x0_HBM0)
        print(e)
    jax.profiler.stop_trace()


def init_fn(unet, params, adamw):
    return train_state.TrainState.create(apply_fn=unet.__call__, params=params, tx=adamw)


if __name__ == "__main__":
    main()
