"""
Attention rollout for determining which patches are most important to the model.
Is an implementation based from Abnar and Zuidema (2020), "quantifying attention flow in transformers".
and also an adaptation from https://github.com/jacobgil/vit-explain/blob/main/vit_rollout.py and
quantifying attention flows in transformers paper.

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

# ViT mean and std (0.5) used to normalise the ViT inputs, shaped so it applies to each colour channel
VIT_MEAN_TENSOR = torch.tensor(VIT_MEAN).view(3, 1, 1)
VIT_STD_TENSOR = torch.tensor(VIT_STD).view(3, 1, 1)

def denormalise_vit(normalised_tensor):
    """Undo the ViT normalisation so the image shows its real colours, clamped to 0-1, just like in gradcam"""
    return (normalised_tensor.cpu() * VIT_STD_TENSOR + VIT_MEAN_TENSOR).clamp(0, 1)

def rollout(attentions):
    """
        Combine the attention of every layer into one map of how much each patch affects the class token.
        @param attentions: The attention matrix of each transformer block, (1, heads, tokens, tokens).
        @return: A 14x14 map for the 196 image patches, scaled so the largest value is 1.
        """
    # Start with the identity, so before any layer each token only contains itself
    result = torch.eye(attentions[0].size(-1), device=attentions[0].device)
    with torch.no_grad():
        for i in attentions:
            # Average the attention over the heads, as in the original method
            attn_head_fused = i.mean(axis=1)

            # Add the identity and halve it to account for the residual connection around each attention layer
            I = torch.eye(attn_head_fused.size(-1), device=attn_head_fused.device)
            a = (attn_head_fused + 1.0 * I) /2
            # Re-normalise so each row sums to 1 again
            a = a/a.sum(dim=-1, keepdim=True)
            # Multiply through the layers so the result tracks attention from the input up to this layer
            result = torch.matmul(a, result)

    # Take the class token's row (token 0) and drop its attention to itself, leaving the image patches
    mask = result[0, 0, 1:]
    # Reshape the 196 patches back into the 14x14 patch grid
    width = int(mask.size(-1)**0.5)
    mask = mask.reshape(width, width).cpu().numpy()
    # Scale so the strongest patch is 1
    mask = mask / np.max(mask)
    return mask

def get_vit_attention_rollout(model, normalised_image_tensor, device="cpu"):
    """
    Create an attention rollout heatmap overlay for a single image.
    :param model: a model instance either an augmented or non-augmented version of ViT
    """
    # Set to eval mode so dropout is off
    model.eval()
    model = model.to(device)

    # timm's fused attention skips the attn_drop module, so turn it off or the hooks capture nothing
    for block in model.backbone.blocks:
        block.attn.fused_attn = False

    attention_maps  = []

    # Save the attention matrix (after the softmax) each time a block runs
    def hook_fn(module, input, output):
        attention_maps.append(output.detach())

    # Attach the hook to every transformer block
    hooks = []
    for block in model.backbone.blocks:
        h = block.attn.attn_drop.register_forward_hook(hook_fn)
        hooks.append(h)

    input_tensor = normalised_image_tensor.unsqueeze(0).to(device)

    # One forward pass collects the attention maps and gives the predicted class
    with torch.no_grad():
        logits = model(input_tensor)
        predicted_class = logits.argmax(dim=1).item()

    # Remove the hooks so they do not keep collecting maps on later calls
    for hook in hooks:
         hook.remove()

    if len(attention_maps) == 0:
        raise RuntimeError("No attention maps captured")

    attention_grid = rollout(attention_maps)

    # Upsample the 14x14 map to the image size so it can overlap the image
    H, W = normalised_image_tensor.shape[1], normalised_image_tensor.shape[2]
    attention_grid_tensor = torch.from_numpy(attention_grid).unsqueeze(0).unsqueeze(0).float()
    heatmap = torch.nn.functional.interpolate(
        attention_grid_tensor, size=(H, W), mode="bilinear", align_corners=False
    )[0, 0].numpy()
    # Overlay on original denormalised image
    original_image = denormalise_vit(normalised_image_tensor).permute(1, 2, 0).cpu().numpy()
    overlay = show_cam_on_image(original_image, heatmap, use_rgb=True)
    return overlay, predicted_class
