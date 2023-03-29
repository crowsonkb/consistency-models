from einshape import jax_einshape as einshape
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax_wavelets as jw

from .utils import rb


class FourierFeatures(nn.Module):
    features: int
    std: float = 1.0

    @nn.compact
    def __call__(self, x):
        assert self.features % 2 == 0
        kernel = self.param(
            "kernel",
            nn.initializers.normal(self.std),
            (x.shape[-1], self.features // 2),
        )
        x = 2 * jnp.pi * x @ kernel
        return jnp.concatenate([jnp.cos(x), jnp.sin(x)], axis=-1)


class TransformerBlock(nn.Module):
    dropout_rate: float = 0.0
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x, z, deterministic=True):
        init = nn.initializers.variance_scaling(1.0, "fan_in", "normal")
        init_out = nn.initializers.zeros

        # Self attention
        x_skip = x
        x = nn.LayerNorm(name="self_attn_norm")(x)
        q = nn.Dense(x.shape[-1], kernel_init=init, dtype=self.dtype, name="query")(x)
        k = nn.Dense(x.shape[-1], kernel_init=init, dtype=self.dtype, name="key")(x)
        v = nn.Dense(x.shape[-1], kernel_init=init, dtype=self.dtype, name="value")(x)
        q = einshape("ns(hd)->nshd", q, d=64)
        k = einshape("ns(hd)->nshd", k, d=64)
        v = einshape("ns(hd)->nshd", v, d=64)
        q = nn.LayerNorm(feature_axes=(-2, -1), name="self_attn_query_norm")(q)
        k = nn.LayerNorm(feature_axes=(-2, -1), name="self_attn_key_norm")(k)
        attn_weights = jnp.einsum("...qhd,...khd->...hqk", q, k) / jnp.sqrt(q.shape[-1])
        attn_weights = nn.Dropout(self.dropout_rate)(attn_weights, deterministic)
        attn_weights = jax.nn.softmax(attn_weights)
        out = jnp.einsum("...hqk,...khd->...qhd", attn_weights, v)
        out = einshape("nshd->ns(hd)", out)
        x = nn.Dense(
            x_skip.shape[-1], kernel_init=init_out, dtype=self.dtype, name="out"
        )(out)
        x = nn.Dropout(self.dropout_rate)(x, deterministic)
        x = x_skip + x

        # Feedforward
        x_skip = x
        x = nn.LayerNorm(name="ff_norm")(x)
        x1 = nn.Dense(
            x.shape[-1] * 4, kernel_init=init, dtype=self.dtype, name="ff_1_1"
        )(x)
        x2 = nn.Dense(
            x.shape[-1] * 4, kernel_init=init, dtype=self.dtype, name="ff_1_2"
        )(x)
        x = x1 * nn.gelu(x2)
        x = nn.Dropout(self.dropout_rate)(x, deterministic)
        x = nn.Dense(
            x_skip.shape[-1], kernel_init=init_out, dtype=self.dtype, name="ff_2"
        )(x)
        x = nn.Dropout(self.dropout_rate)(x, deterministic)
        x = x_skip + x

        return x


class TransformerModel(nn.Module):
    width: int
    depth: int
    wavelet_levels: int
    wavelet: str = "bior4.4"
    dropout_rate: float = 0.0
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, xt, t, deterministic=True):
        n, h, w, c = xt.shape

        # Precompute wavelet transform kernels
        if not self.has_variable("kernels", "kernel"):
            filt = jw.make_kernels(jw.get_filter_bank(self.wavelet), c)
            filt = self.variable("kernels", "kernel", lambda: filt).value
        else:
            filt = self.variable("kernels", "kernel").value

        # Karras preconditioner
        c_in = 1 / jnp.sqrt(t**2 + 1)
        c_out = t / jnp.sqrt(t**2 + 1)
        c_skip = 1 / (t**2 + 1)
        x = xt * rb(c_in, xt)

        # Input patching
        x = jw.wavelet_dec(x, filt[0], self.wavelet_levels, mode="reflect")
        _, h2, w2, c2 = x.shape
        x = einshape("nhwc->n(hw)c", x)
        x = nn.Dense(self.width, dtype=self.dtype, name="proj_in")(x)

        # Timestep embedding
        z = FourierFeatures(self.width, std=1.0, name="timestep_embed")(
            jnp.log(t)[:, None]
        )
        z = nn.Dense(self.width, dtype=self.dtype, name="timestep_embed_in")(z)[:, None]

        # Positional embedding
        n_image_toks = x.shape[1]
        x = jnp.concatenate([x, z], axis=1)
        x = x + self.param(
            "pos_emb", nn.initializers.normal(1.0), (x.shape[1], self.width)
        )

        # Transformer
        x = nn.LayerNorm(name="norm_in")(x)
        for i in range(self.depth):
            x = TransformerBlock(
                dropout_rate=self.dropout_rate,
                dtype=self.dtype,
                name=f"transformer_{i}",
            )(x, z, deterministic=deterministic)
        x = x[:, :n_image_toks]
        x = nn.LayerNorm(name="norm_out")(x)

        # Output unpatching
        x = nn.Dense(c2, kernel_init=nn.initializers.zeros, name="proj_out")(x)
        x = einshape("n(hw)c->nhwc", x, h=h2, w=w2)
        x = jw.wavelet_rec(x, filt[1], self.wavelet_levels, mode="reflect")

        # Karras preconditioner, output
        x = x * rb(c_out, x) + xt * rb(c_skip, xt)

        return x

    @staticmethod
    def weight_decay_mask(params):
        return flax.core.FrozenDict(
            flax.traverse_util.path_aware_map(
                lambda p, v: len(p) >= 3 and p[-1] == "kernel", params
            )
        )
