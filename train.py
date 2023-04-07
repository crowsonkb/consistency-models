#!/usr/bin/env python3

"""Trains a consistency model."""

import argparse
import math
from typing import Any

from einshape import jax_einshape as einshape
import flax
import jax
from jax.experimental.pjit import pjit
from jax.experimental import mesh_utils, multihost_utils
import jax.numpy as jnp
from jax.tree_util import Partial
import numpy as np
import optax
from PIL import Image
from rich import print
from rich.traceback import install

import consistency_models as cm
import jax_local_cluster


class Normalize(flax.struct.PyTreeNode):
    mean: jax.Array
    std: jax.Array

    def forward(self, x):
        return (x - self.mean) / self.std

    def inverse(self, x):
        return x * self.std + self.mean

    def logdet_forward(self, x):
        return -jnp.sum(jnp.broadcast_arrays(x, jnp.log(self.std))[1])

    def logdet_inverse(self, x):
        return jnp.sum(jnp.broadcast_arrays(x, jnp.log(self.std))[1])


class ModelState(flax.struct.PyTreeNode):
    params: Any
    params_ema: Any
    state: Any
    opt_state: Any


class TrainState(flax.struct.PyTreeNode):
    step: jax.Array
    key: jax.Array
    state_s: ModelState
    state_c: ModelState


def main():
    install()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", type=str, default="run", help="the run name")
    parser.add_argument("--seed", type=int, default=43292, help="the random seed")
    args = parser.parse_args()

    try:
        jax.distributed.initialize()
    except ValueError:
        pass

    key = jax.random.PRNGKey(args.seed)

    batch_size_per_device = 64
    batch_size_per_process = batch_size_per_device * jax.local_device_count()
    batch_size = batch_size_per_device * jax.device_count()

    if jax.process_index() == 0:
        print("Processes:", jax.process_count())
        print("Devices:", jax.device_count())
        print("Batch size per device:", batch_size_per_device)
        print("Batch size per process:", batch_size_per_process)
        print("Batch size:", batch_size, flush=True)

    devices = mesh_utils.create_device_mesh((jax.device_count(),))
    mesh = jax.sharding.Mesh(devices, axis_names=("n",))
    pspec_image = jax.sharding.PartitionSpec("n", None, None, None)

    target = {"train": (None, None), "val": (None, None)}
    d_ = flax.serialization.from_bytes(target, open("mnist.msgpack", "rb").read())
    d_ = jnp.array(d_["train"][0])
    dataset = d_ / 255
    size = dataset.shape[1:3]
    ch = dataset.shape[3]
    del d_

    model_s = cm.model.TransformerModel(
        width=512, depth=8, wavelet_levels=2, dtype=jnp.bfloat16
    )
    key, subkey = jax.random.split(key)
    variables = model_s.init(subkey, jnp.zeros([1, *size, ch]), jnp.zeros((1,)))
    state, params = variables.pop("params")
    mask = model_s.weight_decay_mask(params)
    opt_s = optax.adamw(2e-4, b2=0.99, weight_decay=1e-2, mask=mask)
    opt_state = opt_s.init(params)
    state_s = ModelState(params, jax.tree_map(jnp.copy, params), state, opt_state)
    if jax.process_index() == 0:
        print("Score model parameters:", cm.utils.tree_size(params))
    del variables, state, params, opt_state

    model_c = cm.model.TransformerModel(
        width=512, depth=8, wavelet_levels=2, dtype=jnp.bfloat16
    )
    key, subkey = jax.random.split(key)
    variables = model_c.init(subkey, jnp.zeros([1, *size, ch]), jnp.zeros((1,)))
    state, params = variables.pop("params")
    mask = model_c.weight_decay_mask(params)
    opt_c = optax.adamw(2e-4, b2=0.99, weight_decay=1e-2, mask=mask)
    opt_state = opt_c.init(params)
    state_c = ModelState(params, jax.tree_map(jnp.copy, params), state, opt_state)
    if jax.process_index() == 0:
        print("Consistency model parameters:", cm.utils.tree_size(params))
    del variables, state, params, opt_state

    key, subkey = jax.random.split(key)
    state = TrainState(
        step=jnp.array(0, jnp.int32),
        key=subkey,
        state_s=state_s,
        state_c=state_c,
    )
    del state_s, state_c

    # TODO: use decode() and encode() instead
    normalize = Normalize(jnp.array(0.5), jnp.array(0.25))

    def ema_decay(step):
        return jnp.minimum(0.9999, 1 - (step + 1) ** -(2 / 3))

    @Partial(pjit, in_axis_resources=(None, pspec_image), donate_argnums=0)
    def update(state, x):
        # Prepare data
        x = normalize.forward(x)

        # Sample timesteps and noise
        key, *keys = jax.random.split(state.key, 5)
        u = jax.random.uniform(keys[0], x.shape[:1])
        # TODO: make this configurable
        t = jnp.tan((u * 0.998 + 0.001) * jnp.pi / 2)
        weight_fun = cm.cosine_weight
        noise = jax.random.normal(keys[1], x.shape)

        # Update score model
        def model_fun_s(params, xt, t):
            return model_s.apply(
                {"params": params, **state.state_s.state},
                xt,
                t,
                deterministic=False,
                rngs={"dropout": keys[2]},
            )

        loss_s, grad_s = jax.value_and_grad(cm.score_matching_loss)(
            state.state_s.params,
            x,
            t,
            noise,
            model_fun=model_fun_s,
            weight_fun=weight_fun,
        )
        updates, opt_state_s = opt_s.update(
            grad_s, state.state_s.opt_state, state.state_s.params
        )
        params_s = optax.apply_updates(state.state_s.params, updates)
        params_s_ema = cm.utils.ema_update(
            state.state_s.params_ema, params_s, ema_decay(state.step)
        )
        state_s = state.state_s.replace(
            params=params_s, params_ema=params_s_ema, opt_state=opt_state_s
        )

        # Update consistency model
        def model_fun_c(params, xt, t):
            return model_c.apply(
                {"params": params, **state.state_c.state},
                xt,
                t,
                deterministic=False,
                rngs={"dropout": keys[3]},
            )

        loss_c, grad_c = jax.value_and_grad(cm.consistency_loss)(
            state.state_c.params,
            x,
            t,
            noise,
            model_fun=model_fun_c,
            weight_fun=weight_fun,
            metric_fun=cm.l2_metric,
            teacher_fun=Partial(model_fun_s, params_s_ema),
            stopgrad=False,
        )
        updates, opt_state_c = opt_c.update(
            grad_c, state.state_c.opt_state, state.state_c.params
        )
        params_c = optax.apply_updates(state.state_c.params, updates)
        params_c_ema = cm.utils.ema_update(
            state.state_c.params_ema, params_c, ema_decay(state.step)
        )
        state_c = state.state_c.replace(
            params=params_c, params_ema=params_c_ema, opt_state=opt_state_c
        )

        # Assemble new training state
        state = state.replace(
            step=state.step + 1, key=key, state_s=state_s, state_c=state_c
        )
        aux = {"loss_s": loss_s, "loss_c": loss_c}
        return state, aux

    @Partial(pjit, out_shardings=jax.sharding.PartitionSpec(None), static_argnums=2)
    def sample(state, key, n):
        tmax = 160.0

        key, subkey = jax.random.split(key, 2)
        xt = jax.random.normal(subkey, (n, *size, ch)) * tmax
        t = jnp.full((n,), tmax)
        x0 = model_c.apply(
            {"params": state.state_c.params_ema, **state.state_c.state}, xt, t
        )

        x0 = normalize.inverse(x0)
        return x0

    def demo(state, key):
        rows = 10
        cols = 10
        n = rows * cols
        n_adj = math.ceil(n / jax.device_count()) * jax.device_count()
        x0 = sample(state, key, n_adj)
        with jax.spmd_mode("allow_all"):
            x0 = x0[:n]
            grid = einshape("(ab)hwc->(ah)(bw)c", x0, a=rows, b=cols)
            grid = np.array(jnp.round(jnp.clip(grid * 255, 0, 255)).astype(jnp.uint8))
        if jax.process_index() == 0:
            if ch == 1:
                grid = grid[..., 0]
            Image.fromarray(grid).save(f"{args.name}_demo_{step:08}.png")
            print("📸 Output demo grid!", flush=True)

    step = state.step.item()
    perf_ctr = cm.utils.PerfCounter()

    @Partial(pjit, out_axis_resources=pspec_image)
    def select_from_dataset(key, dataset):
        idx = jax.random.choice(key, len(dataset), [batch_size])
        return dataset[idx]

    try:
        while True:
            key, *keys = jax.random.split(key, 3)
            if step % 500 == 0:
                with mesh:
                    demo(state, keys[0])
            # TODO: implement saving
            # if step > 0 and step % 20000 == 0:
            #     if jax.process_index() == 0:
            #         save(train_state, key)
            with mesh:
                x = select_from_dataset(keys[1], dataset)
                state, aux = update(state, x)
            average_time = perf_ctr.update()
            if step % 25 == 0:
                if jax.process_index() == 0:
                    print(
                        f'step: {step}, loss_s: {aux["loss_s"].item():g}, loss_c: {aux["loss_c"].item():g}, {1 / average_time:g} it/s',
                        flush=True,
                    )
            step += 1
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
