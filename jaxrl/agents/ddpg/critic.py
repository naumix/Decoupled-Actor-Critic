from typing import Tuple

import jax.numpy as jnp

import jax
from jaxrl.datasets import Batch
from jaxrl.networks.common import InfoDict, Model, Params, tree_norm, PRNGKey


def update(key: PRNGKey, target_actor: Model, critic: Model, target_critic: Model, batch: Batch,
           discount: float, optimism: jnp.ndarray) -> Tuple[Model, InfoDict]:
    #add seeding
    next_actions = target_actor(batch.next_observations)
    noise = jnp.clip((jax.random.normal(key=key, shape=jnp.shape(next_actions)) * 0.2), a_min=-0.2, a_max=0.2)
    next_actions = jnp.clip((next_actions + noise), a_min=-1.0, a_max=1.0)
    next_q1, next_q2 = target_critic(batch.next_observations, next_actions)
    next_q = (next_q1 + next_q2) / 2 - jnp.squeeze(optimism) * (next_q1 - next_q2) / 2
    target_q = batch.rewards + discount * batch.masks * next_q

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        critic_fn = lambda actions: critic.apply({'params': critic_params},
                                                 batch.observations, actions)
        def _critic_fn(actions):
            q1, q2 = critic_fn(actions)
            return 0.5*(q1 + q2).mean(), (q1, q2)

        (_, (q1, q2)), action_grad = jax.value_and_grad(_critic_fn,
                                                        has_aux=True)(
                                                            batch.actions)
        critic_loss = ((q1 - target_q)**2 + (q2 - target_q)**2).mean()
        return critic_loss, {
            'critic_loss': critic_loss,
            'q1': q1.mean(),
            'q2': q2.mean(),
            'r': batch.rewards.mean(),
            'critic_pnorm': tree_norm(critic_params),
            'critic_agnorm': jnp.sqrt((action_grad ** 2).sum(-1)).mean(0)
        }

    new_critic, info = critic.apply_gradient(critic_loss_fn)
    info['critic_gnorm'] = info.pop('grad_norm')

    return new_critic, info
        
