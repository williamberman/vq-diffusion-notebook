import torch
import PIL
import numpy as np
from inference_VQ_Diffusion import VQ_Diffusion
import torch.nn.functional as F
import os

torch.manual_seed(0)

# Must be run from root directory of microsoft/VQ-Diffusion

model = VQ_Diffusion(config='configs/ithq.yaml', path='OUTPUT/pretrained_model/ithq_learnable.pth')

AUTOENCODER_ENCODED_OUT = "/content/autoencoder_encoded_out_orig.pt"
AUTOENCODER_OUT = "/content/autoencoder_out_orig.pt"

TRANSFORMER_OUT = "/content/transformer_out_orig.pt"

TEXT_EMBEDDER_TOKENIZED_OUT = "/content/text_embedder_tokenized_out_orig.pt"
TEXT_EMBEDDER_OUT = "/content/text_embedder_out_orig.pt"

E2E_LATENTS_OUT = "/content/e2e_latents_out_orig"
E2E_OUT = "/content/e2e_out_orig.pt"
E2E_IMAGE_OUT = "/content/e2e_out_orig.png"

device = 'cuda'

def test_autoencoder():
    print("testing autoencoder")

    vqvae = model.model.content_codec

    input_file_name = "/content/cat.jpg"

    image = PIL.Image.open(input_file_name).convert("RGB")
    image = np.array(image)[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(device)

    with torch.no_grad():
        encoded = vqvae.get_tokens(image)['token']
        reconstructed = vqvae.decode(encoded)[0]

    reconstructed = reconstructed.round().cpu().numpy().astype("uint8").transpose(1, 2, 0)

    print(f"writing autoencoder encoded output to {AUTOENCODER_ENCODED_OUT}")
    torch.save(encoded, AUTOENCODER_ENCODED_OUT)
    print("done writing autoencoder encoded output")

    print(f"writing autoencoder output to {AUTOENCODER_OUT}")
    torch.save(reconstructed, AUTOENCODER_OUT)
    print("done writing autoencoder output")


def test_transformer():
    print("testing transformer")

    transformer = model.model.transformer.transformer

    diffusion_steps = 100
    height = 32
    width = 32
    unrolled_image_dim = height * width
    masked_value = 1408
    batch_size = 1
    condition_len = 77

    x_t = torch.full((batch_size, unrolled_image_dim), masked_value, dtype=torch.long, device=device)

    condition_embedding = torch.ones((batch_size, condition_len, 512), dtype=torch.float, device=device)

    t = torch.full((batch_size,), diffusion_steps - 1, dtype=torch.long, device=device)

    with torch.no_grad():
        transformer_out = transformer(x_t, condition_embedding, t)

        # In the diffusers port, we softmax the output w/in the transformer while
        # the original version outputs the logits. We take the softmax here so we
        # can compare the outputs
        transformer_out = F.log_softmax(transformer_out.double(), dim=1).float()

    print(f"writing transformer output to {TRANSFORMER_OUT}")
    torch.save(transformer_out, TRANSFORMER_OUT)
    print("done writing transformer output")


def test_text_embedder():
    print("testing text embedder")

    condition_codec = model.model.condition_codec
    condition_emb = model.model.transformer.condition_emb

    condition_on = "some words we wrote"

    with torch.no_grad():
        tokenized = condition_codec.get_tokens(condition_on)['token'].to(device)
        embedded = condition_emb(tokenized)

    print(f"writing tokenized output to {TEXT_EMBEDDER_TOKENIZED_OUT}")
    torch.save(tokenized, TEXT_EMBEDDER_TOKENIZED_OUT)
    print("done writing tokenized output")

    print(f"writing embedded output to {TEXT_EMBEDDER_OUT}")
    torch.save(embedded, TEXT_EMBEDDER_OUT)
    print("done writing embedded output")

def test_e2e():
    print("testing e2e")

    os.makedirs(E2E_LATENTS_OUT, exist_ok=True)

    condition_on = "horse"

    image = model.inference_generate_sample_with_condition(condition_on, truncation_rate=0.86, save_root="RESULT", batch_size=1)[0]

    image.save(E2E_IMAGE_OUT)

    torch.save(np.array(image), E2E_OUT)
    
    print("done testing e2e")

test_autoencoder()
test_transformer()
test_text_embedder()
test_e2e()
