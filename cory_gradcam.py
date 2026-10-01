import torch
import numpy as np

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# Was derived from pytorch gradcam https://jacobgil.github.io/pytorch-gradcam-book/introduction.html

def denormalise(normalised_tensor):
    return (normalised_tensor * IMAGENET_STD + IMAGENET_MEAN).clamp(0, 1)

def get_resnet50_gradcam(model, normalised_image_tensor, device = "cpu"):
    model.eval()
    model = model.to(device)

    target_layers = [model.backbone.layer4[-1]]
    input_tensor = normalised_image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        predicted_class = logits.argmax(dim=1).item()

    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0]

    original_image = denormalise(normalised_image_tensor).permute(1, 2, 0).cpu().numpy()
    overlay = show_cam_on_image(original_image, grayscale_cam, use_rgb=True)

    return overlay, grayscale_cam, predicted_class