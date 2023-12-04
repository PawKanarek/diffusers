import os

import jax
import jax.lib.xla_bridge
import jax.numpy as jnp
import jaxlib.xla_client
import jaxlib.xla_extension
import optax
from flax import linen as nn
from flax.training import train_state


class SingleLayerModel(nn.Module):
    features: int

    def setup(self):
        self.dense = nn.Dense(self.features, kernel_init=nn.initializers.normal(), name="dense")

    def __call__(self, x):
        x = self.dense(x)
        return x


def main():
    rng = jax.random.PRNGKey(0)
    adamw = optax.adamw(learning_rate=optax.constant_schedule(1e-5))

    model = SingleLayerModel(features=128)
    params = model.init(rng, jnp.ones((1, int(1e7))))

    states = []
    try:
        while 1:
            state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=adamw)
            states.append(state)
            jax.profiler.save_device_memory_profile(f"prof/memory_profile_{len(states)}.prof")
            print(f"profiles created {len(states)}")
    except jaxlib.xla_extension.XlaRuntimeError as e:
        print(f"Memory exhausted. created {len(states)} states")
        jax.profiler.save_device_memory_profile("prof/memory_profile_last.prof")
        print(e)


if __name__ == "__main__":
    os.makedirs("prof", exist_ok=True)
    main()
