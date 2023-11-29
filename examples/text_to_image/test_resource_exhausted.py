import argparse
import functools
import logging
import math
import os
import random
from pathlib import Path

import jax
import jax.numpy as jnp
import jaxlib.xla_extension
import numpy as np
import optax
import torch
import torch.utils.checkpoint
import transformers
from datasets import load_dataset
from flax import jax_utils
from flax.training import train_state
from flax.training.common_utils import shard
from huggingface_hub import create_repo, upload_folder
from torchvision import transforms
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    CLIPImageProcessor,
    CLIPTokenizer,
    FlaxCLIPTextModel,
    FlaxCLIPTextModelWithProjection,
    set_seed,
)

from diffusers import (
    FlaxAutoencoderKL,
    FlaxDDPMScheduler,
    FlaxPNDMScheduler,
    FlaxStableDiffusionXLPipeline,
    FlaxUNet2DConditionModel,
)
from diffusers.utils import check_min_version


def get_optimizer():
    constant_scheduler = optax.constant_schedule(1e-5)
    adamw = optax.adamw(learning_rate=constant_scheduler, b1=0.9, b2=0.999, eps=1e-08, weight_decay=1e-2)
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), adamw)
    return optimizer


def create_sd():
    print("initialize sd goes smoothly")
    unet, unet_params = FlaxUNet2DConditionModel.from_pretrained(
        "duongna/stable-diffusion-v1-4-flax",
        from_pt=False,
        subfolder="unet",
        dtype=jnp.bfloat16,
    )

    state = train_state.TrainState.create(apply_fn=unet.__call__, params=unet_params, tx=get_optimizer())
    print("created state for sd2-1")
    return state


def create_sdxl():
    print("initialize sdxl-base FlaxUNet2DConditionModel")
    unet, unet_params = FlaxUNet2DConditionModel.from_pretrained(
        "pcuenq/stable-diffusion-xl-base-1.0-flax",
        from_pt=False,
        subfolder="unet",
        dtype=jnp.bfloat16,
    )

    print("******* TRYING TO CREATE TrainState WILL THROW RESOURCE_EXHAUSTED")
    state = train_state.TrainState.create(apply_fn=unet.__call__, params=unet_params, tx=get_optimizer())
    print("Never goes here :(")
    return state


def main():
    # I cannot create TrainState for SDXL on TPU v3-8 with 335 GB RAM
    # create_sdxl will allocate about ~20GB of memory and then will throw error RESOURCE_EXHAUSTED
    # It states that it have problem with allocating 50M of memory, despite fact that ~310GB is still free.
    # try:
    #     create_sdxl()
    # except jaxlib.xla_extension.XlaRuntimeError as e:
    #     # RESOURCE_EXHAUSTED: Error allocating device buffer: Attempting to allocate 50.00M. That was not possible. There are 27.86M free.; (0x0x0_HBM0)
    #     print(e)

    # weird part: i can easly create 10 TrainStates for SD that will consume 200GB so i have free memory
    sd_states = [create_sd() for _ in range(10)]
    print(sd_states)


if __name__ == "__main__":
    main()
