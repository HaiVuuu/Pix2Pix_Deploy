import numpy as np
import torch
import cv2
import io
from PIL import Image
from matplotlib import pyplot as plt
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
import tritonclient.http as httpclient
from diffusers import AutoencoderKL
from transformers import CLIPTokenizer

# Initialize FastAPI
app = FastAPI()

# Triton Server details
TRITON_SERVER_URL = "localhost:8000"
triton_client = httpclient.InferenceServerClient(url=TRITON_SERVER_URL)

# Load HF VAE (for encoding/decoding latents)
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to("cpu").eval()

# Load tokenizer
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")



def run_triton_model(client, model_name, data_dict):
    inputs = []
    
    # Convert each input into a Triton InferInput
    for input_name, data in data_dict.items():
        if not isinstance(data, np.ndarray):
            raise ValueError(f"Expected NumPy array for {input_name}, got {type(data)}")
        
        # Convert to float16 explicitly
        data = data.astype(np.float16)
        
        inp = httpclient.InferInput(input_name, data.shape, "FP16")  # Ensure data type is FP16
        inp.set_data_from_numpy(data, binary_data=True)
        inputs.append(inp)

    # Define output
    out = httpclient.InferRequestedOutput("predicted_noise", binary_data=True)

    # Send request
    response = client.infer(model_name=model_name, inputs=inputs, outputs=[out])

    return response.as_numpy("predicted_noise")



def preprocess_image(image: Image.Image) -> torch.Tensor:
    """ Preprocess image for VAE encoding """
    image = image.convert("L").resize((512, 512))  # Convert to grayscale and resize
    image = np.array(image).astype(np.float32) / 255.0  # Normalize
    image = np.repeat(image[:, :, None], 3, axis=-1)  # Convert to 3-channel grayscale
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to("cpu")  # [1,3,512,512]
    return image_tensor


def encode_latents(image_tensor: torch.Tensor) -> torch.Tensor:
    """ Encode image to latents using VAE """
    with torch.no_grad():
        latents = vae.encode(image_tensor).latent_dist.sample()
    latents_8ch = torch.cat([latents, latents], dim=1)  # Duplicate to 8 channels
    return latents_8ch.to(torch.float32)


def preprocess_prompt(prompt):
    tokens = tokenizer(prompt, return_tensors="pt", padding="max_length", max_length=77, truncation=True)
    input_ids = tokens.input_ids.cpu().numpy().astype(np.int64)

    print("[DEBUG] Sending input to Triton:", input_ids.shape, input_ids)

    # Prepare input
    triton_input = httpclient.InferInput("input_text", input_ids.shape, "INT64")
    triton_input.set_data_from_numpy(input_ids, binary_data=True)

    # Prepare output
    triton_output = httpclient.InferRequestedOutput("text_embeddings", binary_data=True)

    # Send request to Triton
    response = triton_client.infer(
        model_name="text_encoder",
        inputs=[triton_input],
        outputs=[triton_output]
    )

    # Extract embeddings
    text_embeddings = response.as_numpy("text_embeddings")

    if text_embeddings is None:
        raise ValueError("Triton returned None for text embeddings! Check the model input and server logs.")

    print("[DEBUG] Triton Response Shape:", text_embeddings.shape)
    return text_embeddings

@app.post("/infer")
async def infer(file: UploadFile = File(...), prompt: str = ""):
    # Read image
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes))
    image_tensor = preprocess_image(img)
    latents = encode_latents(image_tensor)

    # Encode prompt
    text_embeddings = preprocess_prompt(prompt)

    # Run UNet inference
    latents_np = latents.cpu().numpy().astype(np.float16)
    output_latents = run_triton_model(
        triton_client,
        "unet",
        {
            "latents": latents_np.astype(np.float16),
            "timestep": np.array([1]).astype(np.float16),
            "text_embeddings": text_embeddings.astype(np.float16),
        },
    )

    # Convert back to tensor
    output_latents = torch.from_numpy(output_latents).to("cpu")

    # Decode image using VAE
    with torch.no_grad():
        decoded_image = vae.decode(output_latents.to(torch.float32)).sample
    decoded_image = (decoded_image - decoded_image.min()) / (decoded_image.max() - decoded_image.min())
    decoded_image = (decoded_image * 255).byte().cpu().numpy()[0]  # Remove batch dim
    decoded_image = np.transpose(decoded_image, (1, 2, 0))  # CHW -> HWC

    # Apply histogram equalization to each channel
    channels = cv2.split(decoded_image)
    eq_channels = [cv2.equalizeHist(ch) for ch in channels]
    decoded_image = cv2.merge(eq_channels)

    # Convert NumPy array to PIL image
    output_pil = Image.fromarray(decoded_image, "RGB")
    img_io = io.BytesIO()
    output_pil.save(img_io, format="PNG")
    img_io.seek(0)

    return StreamingResponse(img_io, media_type="image/png")

