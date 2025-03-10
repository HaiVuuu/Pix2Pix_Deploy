import torch
import numpy as np
import onnxruntime as ort
import cv2
from diffusers import AutoencoderKL
from transformers import CLIPTokenizer
from PIL import Image
import matplotlib.pyplot as plt

# Load Hugging Face VAE
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to("cuda").eval()

# Load ONNX models
providers = ["CUDAExecutionProvider"]
text_encoder_sess = ort.InferenceSession("./text_encoder.onnx", providers=providers)
unet_sess = ort.InferenceSession("./unet.onnx", providers=providers)

# Load tokenizer
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")


def preprocess_image(image_path: str) -> torch.Tensor:
    """ Convert image to grayscale and preprocess for VAE encoder """
    image = Image.open(image_path).convert("L").resize((512, 512))  # Convert to grayscale
    image = np.array(image).astype(np.float32) / 255.0  # Normalize to [0,1]

    # Convert to 3-channel grayscale image (R=G=B)
    image = np.expand_dims(image, axis=-1)
    image = np.repeat(image, 3, axis=-1)

    # Convert to tensor and add batch dimension
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to("cuda")  # [1,3,512,512]
    return image_tensor, image  # Also return the original image for display


def encode_latents(image_tensor: torch.Tensor) -> torch.Tensor:
    """ Encode the grayscale image to latent space using HF VAE """
    with torch.no_grad():
        latents = vae.encode(image_tensor).latent_dist.sample()

    # Duplicate latents along channel dimension to get 8 channels
    latents_8ch = torch.cat([latents, latents], dim=1)  # Duplicate to 8 channels
    latents_8ch = latents_8ch.to(torch.float32)  # Convert to float32

    return latents_8ch


def preprocess_prompt(prompt: str) -> np.ndarray:
    """ Tokenize text prompt and encode using ONNX text encoder """
    tokens = tokenizer(prompt, return_tensors="pt", padding="max_length", max_length=77, truncation=True)
    input_ids = tokens.input_ids.cpu().numpy()

    text_embeddings = text_encoder_sess.run(None, {"input_text": input_ids})[0]
    return text_embeddings


def run_inference(image_path: str, prompt: str):
    # Preprocess image
    image_tensor, original_image = preprocess_image(image_path)

    # Encode image using Hugging Face VAE
    latents = encode_latents(image_tensor)

    # Encode prompt
    text_embeddings = preprocess_prompt(prompt)

    # Run UNet
    latents_np = latents.cpu().numpy().astype(np.float16)
    output_latents = unet_sess.run(None, {
        "latents": latents_np,
        "timestep": np.array([1]).astype(np.float16),
        "text_embeddings": text_embeddings.astype(np.float16)
    })[0]

    # Convert back to tensor
    output_latents = torch.from_numpy(output_latents).to("cuda")

    # Debug: Check latents min/max values
    print("Output Latents Min/Max:", output_latents.min().item(), output_latents.max().item())

    # Scale output latents if too dark
    output_latents = output_latents * 1.5  # Adjust this value if necessary

    # Decode image using Hugging Face VAE
    with torch.no_grad():
        decoded_image = vae.decode(output_latents.to(torch.float32)).sample

    # Normalize contrast
    decoded_image = (decoded_image - decoded_image.min()) / (decoded_image.max() - decoded_image.min())
    decoded_image = torch.clamp(decoded_image, 0, 1)

    # Convert to NumPy
    decoded_image = (decoded_image * 255).byte().cpu().numpy()
    decoded_image = np.squeeze(decoded_image, axis=0)  # Remove batch dim
    decoded_image = np.transpose(decoded_image, (1, 2, 0))  # CHW -> HWC

    # ✅ Apply histogram equalization to each channel separately
    channels = cv2.split(decoded_image)  # Split into R, G, B channels
    eq_channels = [cv2.equalizeHist(ch) for ch in channels]  # Apply equalization
    decoded_image = cv2.merge(eq_channels)  # Merge back into an RGB image

    # Convert processed image to PIL
    output_pil = Image.fromarray(decoded_image, "RGB")

    # Show original and processed image side by side
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Original Image
    axes[0].imshow(original_image, cmap="gray")
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Processed Image
    axes[1].imshow(output_pil)
    axes[1].set_title("Processed Image")
    axes[1].axis("off")

    plt.show()


# Run inference
run_inference("./_NFP4016.png", "fallen night")
