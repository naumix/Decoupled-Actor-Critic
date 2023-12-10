from typing import Tuple
from jaxrl.networks.common import InfoDict, Model
from flax import linen as nn
import jax.numpy as jnp

class TemperatureOffset(nn.Module):
    init_value: float = 1.0
    offset: float = 0.0
    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_temp = self.param('log_temp',
                              init_fn=lambda key: jnp.full(
                                  (), jnp.log(self.init_value)))
        return jnp.exp(log_temp) - self.offset

def update_optimism(
        optimism: Model, empirical_kl: float, target_kl: float, beta_lb: float = 1.0
        ) -> Tuple[Model, InfoDict]:
    def optimism_loss_fn(optimism_params):
        beta_ub = optimism.apply({'params': optimism_params})
        optimism_loss = (beta_ub + beta_lb) * (empirical_kl - target_kl).mean() - jnp.clip(beta_ub, a_max=-0.999)
        return optimism_loss, {'beta_ub': beta_ub, 'optimism_loss': optimism_loss}
    new_beta, info = optimism.apply_gradient(optimism_loss_fn)
    info.pop('grad_norm')
    return new_beta, info

def update_regularizer(
        regularizer: Model, empirical_kl: float, target_kl: float
        ) -> Tuple[Model, InfoDict]:
    def regularizer_loss_fn(regularizer_params):
        kl_weight = regularizer.apply({'params': regularizer_params})
        regularizer_loss = -kl_weight * (empirical_kl - target_kl).mean()
        return regularizer_loss, {'kl_weight': kl_weight, 'regularizer_loss': regularizer_loss}
    new_regularizer, info = regularizer.apply_gradient(regularizer_loss_fn)
    info.pop('grad_norm')
    return new_regularizer, info

def update_temperature(
        temp: Model, entropy: float, target_entropy: float
        ) -> Tuple[Model, InfoDict]:
    def temperature_loss_fn(temp_params):
        temperature = temp.apply({'params': temp_params})
        temp_loss = temperature * (entropy - target_entropy).mean()
        return temp_loss, {'temperature': temperature, 'temp_loss': temp_loss}
    new_temp, info = temp.apply_gradient(temperature_loss_fn)
    info.pop('grad_norm')
    return new_temp, info