import functools
from typing import Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import os

from jaxrl.agents.dac import temperature
from jaxrl.agents.dac.actor import update_conservative, update_optimistic
from jaxrl.agents.dac.critic import target_update
from jaxrl.agents.dac.critic import update as update_critic
from jaxrl.datasets import Batch
from jaxrl.networks import critic_net, policies
from jaxrl.networks.common import InfoDict, Model, PRNGKey
        
@functools.partial(jax.vmap, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, None, None, None, None, None, None, None, None))
def _update(
    rng: PRNGKey, actor_c: Model, actor_o: Model, critic: Model, target_critic: Model, temp: Model, optimism: Model, regularizer: Model, 
    batch: Batch, discount: float, tau: float, target_entropy: float, target_kl: float,
    beta_lb: float, beta_critic: float, std_multiplier: float, action_dim: float
) -> Tuple[PRNGKey, Model, Model, Model, Model, Model, Model, Model, InfoDict]:

    rng, key = jax.random.split(rng)
    new_critic, critic_info = update_critic(key, actor_c, critic, target_critic, temp, batch, discount, beta_critic, soft_critic=True)
    new_target_critic = target_update(new_critic, target_critic, tau)
    rng, key = jax.random.split(rng)        
    new_actor_c, actor_c_info = update_conservative(key, actor_c, new_critic, temp, batch, beta_lb)
    rng, key = jax.random.split(rng)
    new_actor_o, actor_o_info = update_optimistic(key, new_actor_c, actor_o, new_critic, optimism, regularizer, batch, std_multiplier, action_dim)
    new_temp, alpha_info = temperature.update_temperature(temp, actor_c_info['entropy'], target_entropy)
    new_optimism, optimism_info = temperature.update_optimism(optimism, actor_o_info['kl'], target_kl)
    new_regularizer, regularizer_info = temperature.update_regularizer(regularizer, actor_o_info['kl'], target_kl)
    
    return rng, new_actor_c, new_actor_o, new_critic, new_target_critic, new_temp, new_optimism, new_regularizer, {
        **critic_info,
        **actor_c_info,
        **actor_o_info,
        **alpha_info,
        **optimism_info,
        **regularizer_info,
    }

@functools.partial(jax.jit, static_argnames=('discount', 'tau', 'target_entropy', 'target_kl', 'beta_lb', 'beta_critic', 'std_multiplier', 'action_dim', 'num_updates'))
def _do_multiple_updates(
        rng: PRNGKey, actor_c: Model, actor_o: Model, critic: Model, target_critic: Model,
                         temp: Model, optimism: Model, regularizer: Model, batches: Batch, discount: float, tau: float, target_entropy: float, target_kl: float,
                         beta_lb: float, beta_critic: float, std_multiplier: float, action_dim: float,
                         step, num_updates: int
                         ) -> Tuple[PRNGKey, Model, Model, Model, Model, Model, Model, Model, InfoDict]:
    def one_step(i, state):
        step, rng, actor_c, actor_o, critic, target_critic, temp, optimism, regularizer, info = state
        step = step + 1
        new_rng, new_actor_c, new_actor_o, new_critic, new_target_critic, new_temp, new_optimism, new_regularizer, info = _update(
                rng, actor_c, actor_o, critic, target_critic, temp, optimism, regularizer,
                jax.tree_map(lambda x: jnp.take(x, i, axis=1), batches), discount, tau, target_entropy, target_kl,
                beta_lb, beta_critic, std_multiplier, action_dim)
        return step, new_rng, new_actor_c, new_actor_o, new_critic, new_target_critic, new_temp, new_optimism, new_regularizer, info
    step, rng, actor_c, actor_o, critic, target_critic, temp, optimism, regularizer, info = one_step(0, (step, rng, actor_c, actor_o, critic, target_critic, temp, optimism, regularizer, {}))
    return jax.lax.fori_loop(1, num_updates, one_step,
                             (step, rng, actor_c, actor_o, critic, target_critic, temp, optimism, regularizer, info))


class DACLearner(object):
    def __init__(self,
                 seed: int,
                 observations: jnp.ndarray,
                 actions: jnp.ndarray,
                 actor_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 temp_lr: float = 3e-4,
                 adjustment_lr: float = 3e-5, 
                 adjustment_beta: float = 0.5, 
                 hidden_dims: Sequence[int] = (256, 256),
                 discount: float = 0.99,
                 tau: float = 0.005,
                 target_entropy: Optional[float] = None,
                 init_temperature: float = 1.0,
                 init_optimism: float = 0.5,
                 init_regularizer: float = 1.0,
                 num_seeds: int = 5,
                 beta_lb: float = 1.0,
                 std_multiplier: float = 1.25,
                 target_kl: float = 0.5,
                 init_copy: int = 60000,
                 copy_interval: int = 500000,
                 ) -> None:
        self.seeds = jnp.arange(seed, seed + num_seeds)
        action_dim = actions.shape[-1]

        if target_entropy is None:
            self.target_entropy = - action_dim / 2
        else:
            self.target_entropy = target_entropy

        self.tau = tau
        self.discount = discount
        self.beta_lb = beta_lb
        self.beta_critic = 1.0
        self.std_multiplier = std_multiplier
        self.init_copy = init_copy
        self.copy_interval = copy_interval
        self.action_dim = float(action_dim)
        self.hidden_dims_c = (256,256,256)
        self.hidden_dims_a = (256,256,256)
        
        self.target_kl = target_kl
        
        def _init_models(seed):
            rng = jax.random.PRNGKey(seed)
            rng, actor_c_key, actor_o_key, critic_key, temp_key, optimism_key, regularizer_key = jax.random.split(rng, 7)
            actor_def = policies.NormalTanhPolicy(self.hidden_dims_a, action_dim)
            actor_c = Model.create(actor_def,
                                 inputs=[actor_c_key, observations],
                                 tx=optax.adam(learning_rate=actor_lr))
            actor_o_ = Model.create(actor_def,
                                 inputs=[actor_o_key, observations],
                                 tx=optax.adam(learning_rate=actor_lr))
            critic_def = critic_net.DoubleCriticLN(self.hidden_dims_c)
            critic = Model.create(critic_def,
                                  inputs=[critic_key, observations, actions],
                                  tx=optax.adam(learning_rate=critic_lr))
            target_critic = Model.create(critic_def, inputs=[critic_key, observations, actions])
            temp = Model.create(temperature.TemperatureOffset(init_value=init_temperature),
                                inputs=[temp_key],
                                tx=optax.adam(learning_rate=temp_lr))
            optimism = Model.create(temperature.TemperatureOffset(init_value=init_optimism, offset=beta_lb),
                                inputs=[optimism_key],
                                tx=optax.adam(learning_rate=adjustment_lr, b1=adjustment_beta))
            regularizer = Model.create(temperature.TemperatureOffset(init_value=init_regularizer),
                                inputs=[regularizer_key],
                                tx=optax.adam(learning_rate=adjustment_lr, b1=adjustment_beta))
            actor_o = target_update(actor_c, actor_o_, 0.0)
            return actor_c, actor_o, critic, target_critic, temp, optimism, regularizer, rng
        
        def _init_actor_o(rng, actor_c):
            _, actor_o_key, optimism_key, regularizer_key = jax.random.split(rng, 4)
            actor_def = policies.NormalTanhPolicy(self.hidden_dims_a, action_dim)
            actor_o = Model.create(actor_def,
                                 inputs=[actor_o_key, observations],
                                 tx=optax.adam(learning_rate=actor_lr))
            new_actor_o = target_update(actor_c, actor_o, 0.0)
            new_optimism = Model.create(temperature.TemperatureOffset(init_value=init_optimism, offset=beta_lb),
                                inputs=[optimism_key],
                                tx=optax.adam(learning_rate=adjustment_lr, b1=adjustment_beta))
            new_regularizer = Model.create(temperature.TemperatureOffset(init_value=init_regularizer, offset=0.0),
                                inputs=[regularizer_key],
                                tx=optax.adam(learning_rate=adjustment_lr, b1=adjustment_beta))
            return new_actor_o, new_optimism, new_regularizer
        
        self.init_models = jax.jit(jax.vmap(_init_models))
        self.init_actor_o = jax.jit(jax.vmap(_init_actor_o))
        self.actor_c, self.actor_o, self.critic, self.target_critic, self.temp, self.optimism, self.regularizer, self.rng = self.init_models(self.seeds)
        self.step = 1
        
    def reinitialize(self):
        self.actor_o, self.optimism, self.regularizer = self.init_actor_o(self.rng, self.actor_c)

    def save_state(self, path: str):
        self.actor_c.save(os.path.join(path, 'actor_c'))
        self.actor_o.save(os.path.join(path, 'actor_o'))
        self.critic.save(os.path.join(path, 'critic'))
        self.target_critic.save(os.path.join(path, 'target_critic'))
        self.temp.save(os.path.join(path, 'temp'))
        self.optimism.save(os.path.join(path, 'optimism'))
        self.regularizer.save(os.path.join(path, 'regularizer'))
        with open(os.path.join(path, 'step'), 'w') as f:
            f.write(str(self.step))

    def load_state(self, path: str):
        self.actor_c = self.actor_c.load(os.path.join(path, 'actor_c'))
        self.actor_o = self.actor_o.load(os.path.join(path, 'actor_o'))
        self.critic = self.critic.load(os.path.join(path, 'critic'))
        self.target_critic = self.target_critic.load(os.path.join(path, 'target_critic'))
        self.temp = self.temp.load(os.path.join(path, 'temp'))
        self.optimism = self.temp.load(os.path.join(path, 'optimism'))
        self.regularizer = self.temp.load(os.path.join(path, 'regularizer'))
        # Restore the step counter
        with open(os.path.join(path, 'step'), 'r') as f:
            self.step = int(f.read())

    def sample_actions(self,
                       observations: np.ndarray,
                       temperature: float = 1.0) -> jnp.ndarray:
        rng, actions = policies.sample_actions(self.rng, self.actor_c.apply_fn,
                                               self.actor_c.params, observations,
                                               temperature)
        self.rng = rng
        actions = np.asarray(actions)
        return np.clip(actions, -1, 1)
    
    def sample_actions_o(self,
                       observations: np.ndarray) -> jnp.ndarray:
        rng, actions = policies.sample_actions(self.rng, self.actor_o.apply_fn,
                                               self.actor_o.params, observations,
                                               self.std_multiplier)
        self.rng = rng
        actions = np.asarray(actions)
        return np.clip(actions, -1, 1)
        
    def update(self, batch: Batch, training_step: int, num_updates: int = 1) -> InfoDict:
        if training_step % 25000 == 0:
            self.reinitialize()
        
        step, rng, actor_c, actor_o, critic, target_critic, temp, optimism, regularizer, info = _do_multiple_updates(
            self.rng, self.actor_c, self.actor_o, self.critic, self.target_critic, self.temp, self.optimism, self.regularizer,
            batch, self.discount, self.tau, self.target_entropy, self.target_kl, 
            self.beta_lb, self.beta_critic, self.std_multiplier, self.action_dim,
            self.step, num_updates)
            
        self.step = step
        self.rng = rng
        self.actor_c = actor_c
        self.actor_o = actor_o
        self.critic = critic
        self.target_critic = target_critic
        self.temp = temp
        self.optimism = optimism
        self.regularizer = regularizer
        return info

    def reset(self):
        self.step = 1
        self.actor_c, self.actor_o, self.critic, self.target_critic, self.temp, self.optimism, self.regularizer, self.rng = self.init_models(self.seeds)
