"""
Attention rollout for determining which patches are most important to the model.
Is an implementation based from Abnar and Zuidema (2020), "quantifying attention flow in transformers".
and also an adaptation from https://github.com/jacobgil/vit-explain/blob/main/vit_rollout.py

"MIT License

Copyright (c) 2020 Jacob Gildenblat

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE."

Generative AI was used to help debug and adapt the implementation.
"""

import torch
import numpy as np
from cory_data_wrapper import VIT_MEAN, VIT_STD
from pytorch_grad_cam.utils.image import show_cam_on_image


VIT_MEAN_TENSOR = torch.tensor(VIT_MEAN).view(3, 1, 1)
VIT_STD_TENSOR = torch.tensor(VIT_STD).view(3, 1, 1)

def denormalise_vit(normalised_tensor):
    return (normalised_tensor.cpu() * VIT_STD_TENSOR + VIT_MEAN_TENSOR).clamp(0, 1)

def rollout(attentions):
    result = torch.eye(attentions[0].size(-1), device=attentions[0].device)
    with torch.no_grad():
        for i in attentions:
            attn_head_fused = i.mean(axis=1)

            I = torch.eye(attn_head_fused.size(-1), device=attn_head_fused.device)
            a = (attn_head_fused + 1.0 * I) /2
            a = a/a.sum(dim=-1, keepdim=True)
            result = torch.matmul(a, result)

    # look at attention between class token and the image patches
    mask = result[0, 0, 1:]
    width = int(mask.size(-1)**0.5)
    mask = mask.reshape(width, width).cpu().numpy()
    mask = mask / np.max(mask)
    return mask

def get_vit_attention_rollout(model, normalised_image_tensor, device="cpu"):
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

    attention_grid = rollout(attention_maps)

    H, W = normalised_image_tensor.shape[1], normalised_image_tensor.shape[2]
    attention_grid_tensor = torch.from_numpy(attention_grid).unsqueeze(0).unsqueeze(0).float()
    heatmap = torch.nn.functional.interpolate(
        attention_grid_tensor, size=(H, W), mode="bilinear", align_corners=False
    )[0, 0].numpy()
    # Overlay on original denormalised image
    original_image = denormalise_vit(normalised_image_tensor).permute(1, 2, 0).cpu().numpy()
    overlay = show_cam_on_image(original_image, heatmap, use_rgb=True)
    return overlay, predicted_class
