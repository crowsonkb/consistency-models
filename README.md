# consistency-models

`consistency-models` is a [JAX](https://jax.readthedocs.io/en/latest/) implementation of the continuous time formulation of [Consistency Models](https://arxiv.org/abs/2303.01469), which allows distillation of a [diffusion model](https://arxiv.org/abs/2006.11239) into a single-step generative model.

**This code is a WORK IN PROGRESS, it is not done, it does not produce high quality results yet, I am releasing it due to general interest in consistency model implementations.**

## Requirements

```bash
pip install git+https://github.com/crowsonkb/jax-wavelets
pip install -r requirements.txt
```

## Notes

`train.py` trains a diffusion model and a consistency model at the same time, and uses L_CD to continuously distill the EMA diffusion model into the consistency model. The consistency model is then used to generate samples in one step. This seems to work better than training the consistency model directly with L_CT.
