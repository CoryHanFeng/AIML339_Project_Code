"""
Attention rollout for determining which patches are most important to the model.

Is an implementation from Abnar and Zuidema (2020), "quantifying attention flow in transformers".
Generative AI was used to help create this file.
"""


import torch
import numpy as np
from PIL import Image

from cory_gradcam_ import denormalise, IMAGENET_MEAN, IMAGENET_STD
from pytorch_grad_cam.utils.image import show_cam_on_image

def get_vit_attention_rollout(model, normalised_image_tensor, device="cpu", discard_ratio=0.0):
    """
    Create an attention rollout heatmap overlay for a single image.
    :param model: a model instance either an augmented or non-augmented version of ViT
    """
    model.eval()
    model = model.to(device)

    for block in model.backbone.blocks:
        block.attn.fused_attn = False

    attention_maps  = []

    def hook_fn(module, input, output):
        attention_maps.append(output.detach())

    hooks = []
    for block in model.backbone.blocks:
        h = block.attn.attn_drop.register_forward_hook(hook_fn)
        hooks.append(h)

    input_tensor = normalised_image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        predicted_class = logits.argmax(dim=1).item()

    for hook in hooks:
         hook.remove()

    if len(attention_maps) == 0:
        raise RuntimeError("No attention maps captured")

    # Roll out across layers
    N = attention_maps[0].shape[-1]  # num_patches + 1 (CLS token)
    rollout = torch.eye(N).to(device)
    for i in attention_maps:
        attention_avg = i.mean(dim=1)[0]

        if discard_ratio > 0:
            flat = attention_avg.flatten()
            n_discard = int(flat.numel() * discard_ratio)
            if n_discard > 0:
                threshold = flat.kthvalue(n_discard).values
                attention_avg = torch.where(attention_avg < threshold, torch.zeros_like(attention_avg), attention_avg)

        attn_with_residual = attention_avg + torch.eye(N).to(device)
        attn_with_residual = attn_with_residual / attn_with_residual.sum(dim=-1, keepdim=True)

        rollout = attn_with_residual @ rollout

    cls_attention = rollout[0, 1:]

    num_patches = cls_attention.shape[0]
    grid_size = int(num_patches ** 0.5)
    if grid_size * grid_size != num_patches:
        raise RuntimeError(f"Number of patches ({num_patches}) is not a perfect square - "
                          f"cannot reshape into a spatial grid.")
    attention_grid = cls_attention.reshape(grid_size, grid_size).cpu().numpy()
    attention_grid = attention_grid / attention_grid.max()

    H, W = normalised_image_tensor.shape[1], normalised_image_tensor.shape[2]
    attention_grid_tensor = torch.from_numpy(attention_grid).unsqueeze(0).unsqueeze(0).float()
    heatmap = torch.nn.functional.interpolate(
        attention_grid_tensor, size=(H, W), mode="bilinear", align_corners=False
    )[0, 0].numpy()
    # Overlay on original denormalised image
    original_image = denormalise(normalised_image_tensor).permute(1, 2, 0).cpu().numpy()
    overlay = show_cam_on_image(original_image, heatmap, use_rgb=True)
    return overlay, predicted_class