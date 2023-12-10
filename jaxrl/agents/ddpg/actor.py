from typing import Tuple

import jax.numpy as jnp

from jaxrl.datasets import Batch
from jaxrl.networks.common import InfoDict, Model, Params


def update(actor: Model, critic: Model,
           batch: Batch, optimism: jnp.ndarray) -> Tuple[Model, InfoDict]:
    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        actions = actor.apply({'params': actor_params}, batch.observations)
        q1, q2 = critic(batch.observations, actions)
        q_lb = (q1 + q2) / 2 - jnp.squeeze(optimism) * jnp.abs(q1 - q2) / 2
        actor_loss = -q_lb.mean()
        return actor_loss, {'actor_loss': actor_loss}

    new_actor, info = actor.apply_gradient(actor_loss_fn)

    return new_actor, info
