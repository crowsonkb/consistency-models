import jax
import jax.numpy as jnp
from jax.tree_util import Partial

from .utils import rb


def l2_metric(x, y):
    return jnp.sum((x - y) ** 2)


# TODO: Rescale based on tmin and tmax
def uniform_weight(t):
    return jnp.ones_like(t)


# TODO: Rescale based on tmin and tmax
def cosine_weight(t):
    return (t**2 + 1) * jnp.pi / 2


batchmean = jax.vmap(jnp.mean)


def hvp(fun, x, v):
    return jax.jvp(jax.grad(fun), (x,), (v,))[1]


def consistency_loss(
    params,
    x0,
    t,
    noise,
    model_fun,
    weight_fun=uniform_weight,
    metric_fun=l2_metric,
    teacher_fun=None,
    stopgrad=False,
):
    xt = x0 + noise * rb(t, x0)
    if teacher_fun is None:
        eps = noise
    else:
        eps = (xt - teacher_fun(xt, t)) / rb(t, x0)
    out, out_jvp = jax.jvp(Partial(model_fun, params), (xt, t), (eps, jnp.ones_like(t)))
    if stopgrad:
        out_hvp = jax.lax.stop_gradient(hvp(lambda y: metric_fun(y, out), out, out_jvp))
        return jnp.mean(weight_fun(t) * batchmean(out * out_hvp))
    out_hvp = hvp(lambda y: metric_fun(out, y), out, out_jvp)
    return jnp.mean(weight_fun(t) * batchmean(out_jvp * out_hvp)) / 2


def score_matching_loss(params, x0, t, noise, model_fun, weight_fun=uniform_weight):
    xt = x0 + noise * rb(t, x0)
    eps = (xt - model_fun(params, xt, t)) / rb(t, x0)
    return jnp.mean(weight_fun(t) * batchmean((eps - noise) ** 2))
