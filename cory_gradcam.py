"""
Creates the Grad-CAM heatmaps for the ResNet-50.
Grad-CAM (Selvaraju et al., 2017) uses the gradients of the predicted class flowing into the last convolutional layer
to show which parts of the image the model used.
Was derived from pytorch gradcam https://jacobgil.github.io/pytorch-gradcam-book/introduction.html (MIT licence)
"""
import torch
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
# ImageNet mean and std used to normalise the ResNet-50 inputs, shaped so it applies to each colour channel
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# Was derived from pytorch gradcam https://jacobgil.github.io/pytorch-gradcam-book/introduction.html

def denormalise(normalised_tensor):
    """Undo the ImageNet normalisation so the image shows its real colours, clamped to 0-1"""
    return (normalised_tensor * IMAGENET_STD + IMAGENET_MEAN).clamp(0, 1)

def get_resnet50_gradcam(model, normalised_image_tensor, device = "cpu"):
    """
        Create a Grad-CAM heatmap for one image, for the class the model predicts.

        @param model: a CNNModel (ResNet-50) with the fine-tuned weights loaded.
        @param normalised_image_tensor: one (3, 224, 224) test image, normalised with the ImageNet statistics.
        @param device: the device to run the model on, "cpu" or "cuda".
        @return: the heatmap overlaid on the image (H, W, 3 RGB array), the raw heatmap
                 (H, W array scaled to [0, 1]) and the predicted class index.
        """
    # Set to eval mode so batch norm uses its saved statistics
    model.eval()
    model = model.to(device)
    # Use the last residual block, as it is the last convolutional layer and still keeps where things are in the image
    target_layers = [model.backbone.layer4[-1]]

    input_tensor = normalised_image_tensor.unsqueeze(0).to(device)

    # Get the predicted class so it can be compared with the true label
    with torch.no_grad():
        logits = model(input_tensor)
        predicted_class = logits.argmax(dim=1).item()

    # Create the heatmap, targets=None uses the predicted class instead of the true label
    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0]

    # Change the image back to real colours and to (height, width, colour) for plotting
    original_image = denormalise(normalised_image_tensor).permute(1, 2, 0).cpu().numpy()

    # Put the coloured heatmap over the image, red shows the areas that raised the class score the most
    overlay = show_cam_on_image(original_image, grayscale_cam, use_rgb=True)
    return overlay, grayscale_cam, predicted_class