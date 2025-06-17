# Scripts for stylizing images (useful for getting ArtFID scores)
# TODO: Make it batch, not stylize images 1 by 1 (which is really slow)

import torch
import torchvision
import models
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import argparse

# Make sure the content and style directiories contain images (without subdirectories)
# Make sure the stylized images directory is empty
CONTENT_IMAGES_PATH = "content_images/"
STYLE_IMAGES_PATH = "style_images/"
STYLIZED_IMAGES_DEST = "stylized_images/adain/"
DECODER_WEIGHTS_PATH = "models/epoch_25_sw5.pt"
IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# Obviously, doesn't include rescaling and cropping
def inverse_vgg_transform(image: torch.Tensor) -> Image:
    mean = torch.Tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.Tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    unnorm = (std * image) + mean
    np_arr = unnorm.clamp(0, 1).detach().numpy().transpose(1, 2, 0)
    return Image.fromarray((np_arr*255).astype(np.uint8))

def vgg_transform_custom_resize(image: Image) -> torch.Tensor:
    image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
    mean = torch.Tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.Tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    t1 = torchvision.transforms.ToTensor()
    image = t1(image)
    return (image - mean) / std

def stylize_images():
    encoder = models.get_vgg19_extractor().to(DEVICE)
    adain = models.AdaIN().to(DEVICE)
    decoder = models.DecoderForVGG19Encoder().to(DEVICE)
    decoder.load_state_dict(torch.load(DECODER_WEIGHTS_PATH, map_location=torch.device("cpu")))

    content_paths = [CONTENT_IMAGES_PATH + x for x in os.listdir(CONTENT_IMAGES_PATH)]
    style_paths = [STYLE_IMAGES_PATH + x for x in os.listdir(STYLE_IMAGES_PATH)]
    for i in tqdm(range(min(len(content_paths), len(style_paths)))):
        content_image = vgg_transform_custom_resize(Image.open(content_paths[i]).convert('RGB')).unsqueeze(0).to(DEVICE)
        style_image = vgg_transform_custom_resize(Image.open(style_paths[i]).convert('RGB')).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            encoded_content = encoder(content_image)[0]
            encoded_style = encoder(style_image)[0]
            adain_output = adain(encoded_content, encoded_style)
            stylized_image = decoder(adain_output)[0] # Removes the batch dimension
        stylized_image = inverse_vgg_transform(stylized_image)
        stylized_image_path = STYLIZED_IMAGES_DEST + content_paths[i].split("/")[-1] + "_" + style_paths[i].split("/")[-1] + "_stylized.png"
        stylized_image.save(stylized_image_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--content_images_path", type=str)
    parser.add_argument("--style_images_path", type=str)
    parser.add_argument("--stylized_images_dest", type=str)
    parser.add_argument("--decoder_weights_path", type=str)
    parser.add_argument("--image_width", type=int, default=512)
    parser.add_argument("--image_height", type=int, default=512)

    args = parser.parse_args()
    CONTENT_IMAGES_PATH = args.content_images_path
    STYLE_IMAGES_PATH = args.style_images_path
    STYLIZED_IMAGES_DEST = args.stylized_images_dest
    DECODER_WEIGHTS_PATH = args.decoder_weights_path
    IMAGE_WIDTH = args.image_width
    IMAGE_HEIGHT = args.image_height

    stylize_images()