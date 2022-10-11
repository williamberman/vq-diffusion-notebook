# VQ-Diffusion notebook

This is a python notebook for verifying [VQ-diffusion](https://github.com/microsoft/VQ-Diffusion) 
from the original repo against the port to [diffusers](https://github.com/huggingface/diffusers/pull/658).

#### Running

This repository and notebook is written with the intent to be ran on a google colab instance. 
To get started, create a new notebook from [main.ipynb](./main.ipynb) and run the notebook step by step.

#### Weights

Weights are downloaded from a re-upload to the [huggingface hub](https://huggingface.co/williamberman/vq-diffusion-orig)
instead of the links in the original repository to make downloading easier.

If you don't want to re-download weights every time you spin up a new instance, you can save them to your google drive
and mount your google drive on restart instead.
