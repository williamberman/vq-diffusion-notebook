from diffusers.pipelines import VQDiffusionPipeline
import torch
import PIL
import numpy as np
import os

torch.manual_seed(0)

device = 'cuda'

pipeline = VQDiffusionPipeline.from_pretrained("/content/vq-diffusion-diffusers-dump").to(device)


AUTOENCODER_ENCODED_OUT = "/content/autoencoder_encoded_out.pt"
AUTOENCODER_OUT = "/content/autoencoder_out.pt"
AUTOENCODER_IMAGE_OUT = "/content/cat-reconstructed.jpg"

TRANSFORMER_OUT = "/content/transformer_out.pt"

TEXT_EMBEDDER_TOKENIZED_OUT = "/content/text_embedder_tokenized_out.pt"
TEXT_EMBEDDER_OUT = "/content/text_embedder_out.pt"

E2E_LATENTS_OUT = "/content/e2e_latents_out"
E2E_OUT = "/content/e2e_out.pt"
E2E_IMAGE_OUT = "/content/e2e_out.png"

def test_autoencoder():
    print("testing autoencoder")

    vqvae = pipeline.vqvae

    input_file_name = "/content/cat.jpg"

    image = PIL.Image.open(input_file_name).convert("RGB")
    image = preprocess_image(image).to(device)

    with torch.no_grad():
        encoded = vqvae.encode(image).latents
        # Original vq-diffusion uses the min encoding indices
        _, _, (_, _, encoded_min_encoding_indices) = vqvae.quantize(encoded)
        reconstructed = postprocess_image(vqvae.decode(encoded).sample)

    print(f"writing autoencoder encoded output to {AUTOENCODER_ENCODED_OUT}")
    torch.save(encoded_min_encoding_indices, AUTOENCODER_ENCODED_OUT)
    print("done writing autoencoder encoded output")

    print(f"writing autoencoder output to {AUTOENCODER_OUT}")
    torch.save(reconstructed, AUTOENCODER_OUT)
    print("done writing autoencoder output")

    print(f"writing autoencoder reconstructed image to {AUTOENCODER_IMAGE_OUT}")
    PIL.Image.fromarray(reconstructed).save(AUTOENCODER_IMAGE_OUT)
    print("done writing autoencoder reconstructed image")


def preprocess_image(image):
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0


def postprocess_image(image):
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).round().astype('uint8')
    image = image[0]
    return image


def test_transformer():
    print("testing transformer")

    transformer = pipeline.transformer

    diffusion_steps = 100
    height = 32
    width = 32
    unrolled_image_dim = height * width
    masked_value = 1408
    batch_size = 1
    condition_len = 77

    x_t = torch.full((batch_size, unrolled_image_dim), masked_value, dtype=torch.long, device=device)

    condition_embedding = torch.ones((batch_size, condition_len, 512), dtype=torch.float, device=device)

    t = torch.tensor(diffusion_steps - 1, device=device, dtype=torch.long)

    with torch.no_grad():
        transformer_out = transformer(x_t, condition_embedding, t)

    print(f"writing transformer output to {TRANSFORMER_OUT}")
    torch.save(transformer_out, TRANSFORMER_OUT)
    print("done writing transformer output")


def test_text_embedder():
    print("testing text embedder")

    tokenizer = pipeline.tokenizer
    text_encoder = pipeline.text_encoder

    condition_on = "some words we wrote"

    with torch.no_grad():
        tokenized = tokenizer(condition_on, padding='max_length', return_tensors='pt').input_ids.to(device)
        embedded = text_encoder(tokenized).last_hidden_state
        embedded = embedded / embedded.norm(dim=-1, keepdim=True)

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

    image = pipeline(condition_on, truncation_rate=0.86, callback=write_latents).images[0]

    image.save(E2E_IMAGE_OUT)

    torch.save(np.array(image), E2E_OUT)
    
    print("done testing e2e")

def write_latents(_, t, x_t):
    torch.save(x_t, f"{E2E_LATENTS_OUT}/{t}.pt")

test_autoencoder()
test_transformer()
test_text_embedder()
test_e2e()
