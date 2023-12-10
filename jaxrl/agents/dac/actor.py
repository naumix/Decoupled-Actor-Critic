from typing import Tuple

import jax.numpy as jnp
import flax.linen as nn
from jaxrl.datasets import Batch
from jaxrl.networks.common import InfoDict, Model, Params, PRNGKey, tree_norm


def update_conservative(
        key: PRNGKey, actor_c: Model, critic: Model, temp: Model, 
        batch: Batch, beta_lb: float
           ) -> Tuple[Model, InfoDict]:
    def actor_c_loss_fn(actor_c_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        dist = actor_c.apply({'params': actor_c_params}, batch.observations, return_params=False)
        actions = dist.sample(seed=key)
        log_probs = dist.log_prob(actions)
        q1, q2 = critic(batch.observations, actions)
        if beta_lb == 1.0:
            q = jnp.minimum(q1, q2)
        else:
            q = (q1 + q2) / 2 - beta_lb * jnp.abs(q1 - q2) / 2
        actor_loss = (log_probs * temp()).mean() - q.mean()
        return actor_loss, {
            'actor_loss': actor_loss,
            'entropy': -log_probs.mean(),
            'actor_pnorm': tree_norm(actor_c_params),
            'actor_action': jnp.mean(jnp.abs(actions)),
        }
    new_actor_c, info = actor_c.apply_gradient(actor_c_loss_fn)
    info['actor_c_gnorm'] = info.pop('grad_norm')
    return new_actor_c, info

def update_optimistic(
        key: PRNGKey, actor_c: Model, actor_o: Model, critic: Model, optimism: Model, 
        regularizer: Model, batch: Batch, std_multiplier: float, action_dim: float
        ) -> Tuple[Model, InfoDict]:
    def actor_o_loss_fn(actor_o_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        _, mu_c, std_c = actor_c(batch.observations, temperature=1.0, return_params=True)
        dist, mu_o, std_ox = actor_o.apply({'params': actor_o_params}, observations=batch.observations, temperature=std_multiplier, return_params=True)
        std_o = std_ox / std_multiplier
        actions = dist.sample(seed=key)
        q1, q2 = critic(batch.observations, actions)
        kl = (jnp.log(std_c/std_o) + (std_o ** 2 + (mu_o - mu_c) ** 2)/(2 * std_c ** 2) - 1/2).sum(-1)
        q_ub = (q1 + q2) / 2 + optimism() * jnp.abs(q1 - q2) / 2
        actor_e_loss = (-q_ub).mean() + jnp.clip(regularizer(), a_max=20.0) * kl.mean()    
        return actor_e_loss, {
            'actor_o_loss': actor_e_loss,
            'kl': kl.mean()/action_dim,
            'actor_mu_diff': jnp.mean(jnp.abs(nn.tanh(mu_c) - nn.tanh(mu_o))),
            'actor_o_pnorm': tree_norm(actor_o_params),
            'std': std_c.mean(),
            'std_e': std_o.mean(),
            'Q_mean': ((q1 + q2) / 2).mean(),
            'Q_std': (jnp.abs(q1 - q2) / 2).mean(),
            'Qloss': (-q_ub).mean(),
            'KLoss': kl.mean(),
        }
    new_actor_o, info = actor_o.apply_gradient(actor_o_loss_fn)
    info['actor_o_gnorm'] = info.pop('grad_norm')
    return new_actor_o, info