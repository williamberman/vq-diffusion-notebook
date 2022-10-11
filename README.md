# VQ-Diffusion notebook

This is a python notebook for verifying [VQ-diffusion](https://github.com/microsoft/VQ-Diffusion) 
from the original repo against the port to [diffusers](https://github.com/huggingface/diffusers/pull/658).

#### Running

This repository and notebook is written with the intent to be ran on a google colab instance. 
To get started, create a new notebook from [main.ipynb](./main.ipynb) and run the notebook step by step.
Take note of which cells are optional depending on if you want to download weights from the hub or use
pre-downloaded weights mounted from google drive.

The notebook runs on a colab instance with minimum high-RAM and a standard GPU.

#### Weights

Weights are downloaded from a re-upload to the [huggingface hub](https://huggingface.co/williamberman/vq-diffusion-orig)
instead of the links in the original repository to make downloading easier.

If you don't want to re-download weights every time you spin up a new instance, you can save them to your google drive
and mount your google drive on restart instead. 

Note there is no way to mount an isolated subdirectory of your google drive and this _will_ give this notebook full access 
to your google drive. The cells which request google drive access are marked as such and can be skipped as long as the 
weights are downloaded from the huggingface hub.
