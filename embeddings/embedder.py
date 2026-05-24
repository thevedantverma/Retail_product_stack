import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32"
)

processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32"
)

def get_embedding(image_crop):

    image = Image.fromarray(image_crop)

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    with torch.no_grad():

        features = model.get_image_features(
            **inputs
        )

    embedding = features[0].cpu().numpy()

    return embedding.flatten()