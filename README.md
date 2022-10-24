# VQ-Diffusion notebook

This is a python notebook for verifying [VQ-diffusion](https://github.com/microsoft/VQ-Diffusion) 
from the original repo against the port to [diffusers](https://github.com/huggingface/diffusers/pull/658).

#### tl;dr
The end of [main.ipynb](./main.ipynb) shows the differences between the outputs of the
autoencoder, the transformer, and the text encoder between the original vq-diffusion model
and the diffusers port.

#### Running

This repository and notebook is written with the intent to be ran on a google colab instance. 
To get started, create a new notebook from [main.ipynb](./main.ipynb) and run the notebook step by step.
Take note of which cells are optional depending on if you want to download weights from the hub or use
pre-downloaded weights mounted from google drive.

The notebook runs on a colab instance with high-RAM and standard GPU.

#### Weights

Weights are downloaded from a re-upload to the [huggingface hub](https://huggingface.co/williamberman/vq-diffusion-orig)
instead of the links in the original repository to make downloading easier.

If you don't want to re-download weights every time you spin up a new instance, you can save them to your google drive
and mount your google drive on restart instead. 

Note there is no way to mount an isolated subdirectory of your google drive and this _will_ give this notebook full access 
to your google drive. The cells which request google drive access are marked as such and can be skipped as long as the 
weights are downloaded from the huggingface hub.

#### Changes from original VQ-Diffusion
This runs a fork of the original VQ-Diffusion repository with a few commits to assist verifying. I.e. writing intermediate
latents to disk and returning PIL objects.

#### Notebook Steps

1. Installs dependencies. 
2. The original model is converted to diffusers and written to the file system.
3. [test_vq_diffusion_orig.py](./test_vq_diffusion_orig.py) loads the
original vq diffusion model and writes outputs from the autoencoder, transformer, and text encoder to the file system.
4. [test_vq_diffusion_diffusers.py](./test_vq_diffusion_diffusers.py) does the same for the diffusers port.
5. The outputs are verified against each other. The notebook specifies
what are acceptable discrepancies in the outputs.
