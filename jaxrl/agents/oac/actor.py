from typing import Tuple

import jax.numpy as jnp

from jaxrl.datasets import Batch
from jaxrl.networks.common import InfoDict, Model, Params, PRNGKey, tree_norm


def update(key: PRNGKey, actor: Model, critic: Model, temp: Model,
           batch: Batch, beta_lb: float) -> Tuple[Model, InfoDict]:
    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        dist = actor.apply({'params': actor_params}, batch.observations)
        actions = dist.sample(seed=key)
        log_probs = dist.log_prob(actions)
        q1, q2 = critic(batch.observations, actions)
        q_mean = (q1 + q2) / 2
        q_std = jnp.abs(q1 - q2) / 2
        q = q_mean - beta_lb * q_std
        actor_loss = (log_probs * temp() - q).mean()
        return actor_loss, {
            'actor_loss': actor_loss,
            'entropy': -log_probs.mean(),
            'actor_pnorm': tree_norm(actor_params),
            'actor_action': jnp.mean(jnp.abs(actions))
        }

    new_actor, info = actor.apply_gradient(actor_loss_fn)
    info['actor_gnorm'] = info.pop('grad_norm')

    return new_actor, info