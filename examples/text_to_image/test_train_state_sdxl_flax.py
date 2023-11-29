import jax.numpy as jnp
import optax
from flax.training import train_state

from diffusers import FlaxUNet2DConditionModel


# I cannot create TrainState for SDXL on TPU v3-8 with 335 GB RAM
# create_sdxl will allocate about ~20GB of memory and then will throw error RESOURCE_EXHAUSTED
# It states that it have problem with allocating 50M of memory, despite fact that ~310GB is still free.
print("initialize sdxl-base FlaxUNet2DConditionModel")
unet, unet_params = FlaxUNet2DConditionModel.from_pretrained(
    "pcuenq/stable-diffusion-xl-base-1.0-flax",
    from_pt=False,
    subfolder="unet",
    dtype=jnp.bfloat16,
)
constant_scheduler = optax.constant_schedule(1e-5)
adamw = optax.adamw(learning_rate=constant_scheduler, b1=0.9, b2=0.999, eps=1e-08, weight_decay=1e-2)
optimizer = optax.chain(optax.clip_by_global_norm(1.0), adamw)

print("******* TRYING TO CREATE TrainState AND THIS WILL THROW RESOURCE_EXHAUSTED")
state = train_state.TrainState.create(apply_fn=unet.__call__, params=unet_params, tx=optimizer)
# RESOURCE_EXHAUSTED: Error allocating device buffer: Attempting to allocate 50.00M. That was not possible. There are 27.86M free.; (0x0x0_HBM0)
print(f"Never goes here :( {state=}")
