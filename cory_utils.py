# Imports
import random

import numpy as np
import torch
import torch.nn as nn

# Setup tensorboard
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

# Create the class for the CNN model
class CNNModel(nn.Module):
    """ResNet-50 pretrained, change the size of head to the number of classes"""

    def __init__(self, arch_name, num_classes=9, pretrained=True):
        super().__init__()
        import torchvision.models as torchv_models

        # Only accept the defined models
        if arch_name == "resnet50":
            weights = torchv_models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            self.backbone = torchv_models.resnet50(weights=weights)
            self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

        else:
            raise ValueError(f"Unknown CNN architecture: {arch_name}. Supported: 'resnet50'")

    # Forward pass
    def forward(self, x):
        return self.backbone(x)

# Create the class for the Vision Transformer
class ViTModel(nn.Module):
    """ViT base 16 pretrained"""
    def __init__(self, arch_name="vit_base_patch16_224", num_classes=9, pretrained=True):
        super().__init__()
        import timm
        self.backbone = timm.create_model(arch_name, pretrained=pretrained, num_classes=num_classes)
    # forward pass
    def forward(self, x):
        return self.backbone(x)

# Create the optimiser and cosine scheduler to adapt the learning rate
def build_optimizer_and_scheduler(model, lr, weight_decay, warmup_steps, total_steps):
    # Create the optimiser with the learning rate set to the learning rate found in the parameter search
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Create a warmup to help sensitive initialisation
    warmup = torch.optim.lr_scheduler.LinearLR(optimiser, start_factor=1e-8, end_factor=1.0, total_iters=max(1, warmup_steps))
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=max(1, total_steps - warmup_steps), eta_min=0.0)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimiser, schedulers=[warmup, cosine], milestones=[warmup_steps])
    return optimiser, scheduler

# Create the training loop
# using AMP: https://docs.pytorch.org/docs/2.13/notes/amp_examples.html
def train_model(model, train_loader, val_loader, device, epochs = 10,
                lr=3e-5, weight_decay=0.01, warmup_fraction = 0.1,
                grad_clip = 1.0, use_amp = True, early_stopping_patience = 2,
                model_name = "model", verbose = True, log_dir = None):
    """Train model with AMP (to speed up training) use early stopping and gradient clipping to avoid exploding gradient and overfitting"""
    # initialise tensorboard writer
    if log_dir is not None and SummaryWriter is None:
        raise ImportError("need to install tensorboard")
    writer = SummaryWriter(log_dir) if log_dir is not None else None

    # Set up
    # move to gpu
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    amp = use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled = amp)

    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * warmup_fraction)
    optimizer, scheduler = build_optimizer_and_scheduler(model, lr, weight_decay, warmup_steps, total_steps)

    history = {"train_loss":[],
               "train_accuracy":[],
               "val_loss":[],
               "val_accuracy":[]}
    best_val_acc = -1.0
    epochs_without_improvement = 0
    best_state_dict = None

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss, train_correct, train_total = 0.0, 0, 0
        n_batches = 0
        # Iterate over the training data
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            # Use AMP to speed up training
            with torch.amp.autocast("cuda", enabled=amp):
                outputs = model(images)
                loss = criterion(outputs, labels)

            # Clip the gradients and update the parameters (backpropagation)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # calculate the loss and the accuracy and update the total loss and total accuracy
            running_loss += loss.item()
            n_batches += 1
            train_correct += (outputs.argmax(dim=1) == labels).sum().item()
            train_total += labels.size(0)
        # Callculate the training loss and accuracy
        train_loss = running_loss / max(1, n_batches)
        train_acc = train_correct / max(1, train_total)

        # Validation
        model.eval()
        val_loss_total, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss_total += criterion(outputs, labels).item()
                val_correct += (outputs.argmax(dim=1) == labels).sum().item()
                val_total += labels.size(0)

        val_loss = val_loss_total / max(1, len(val_loader))
        val_acc = val_correct / max(1, val_total)

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        if writer is not None:
            writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, epoch)
            writer.add_scalars("Accuracy", {"train": train_acc, "val": val_acc}, epoch)

        if verbose:
            print(f"[{model_name}] Epoch {epoch+1}/{epochs}: "
                  f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f} "
                  f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

        # Update the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
                if verbose:
                    print(f"[{model_name}] Early stopping at epoch {epoch+1}")
                break
    if writer is not None:
        writer.close()

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    return model.to(device), history

def hyperparameter_search(make_model, train_loader, val_loader,device,
                          learning_rates=(1e-5, 3e-5, 1e-4), search_epochs=3,
                          warmup_fraction=0.1, verbose=True, log_dir=None):
    """train the model on each learning rate for 3 epochs return the best model and use that learning rate"""
    trials = []
    best_lr =None
    best_val_acc = -1.0

    for l in learning_rates:
        if verbose:
            print(f"\n--- HP search: lr={l:.0e} ---")
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)
        # Train the model
        trial_log_dir = f"{log_dir}/lr{l:.0e}" if log_dir is not None else None
        model, history = train_model(make_model(),
                                     train_loader,
                                     val_loader,
                                     device,
                                     epochs=search_epochs,
                                     lr=l,
                                     warmup_fraction=warmup_fraction,
                                     early_stopping_patience=search_epochs,
                                     model_name=f"hpsearch_lr{l:.0e}",
                                     verbose=verbose,
                                     log_dir=trial_log_dir)
        val_acc = max(history["val_accuracy"])
        trials.append({"lr": l, "val_accuracy": val_acc})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_lr = l

    if verbose: print(f"Best lr: {best_lr:.0e} (val_acc={best_val_acc:.4f})")
    return best_lr, trials

def evaluate_accuracy(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            correct += (model(images).argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return correct / max(1, total)

def evaluate_on_variants(model, variant_loaders, device):
    """Get the accuracy of the model on each test variant"""
    return {name: evaluate_accuracy(model, loader,device) for name, loader in variant_loaders.items()}

def compute_reliance_metrics(variant_accuracies, num_classes=9):
    """Compute the reliance metrics, no foreground, mixed same background and the original accuracy"""
    result = {"chance_accuracy": 1.0 / num_classes}
    if "no_fg" in variant_accuracies:
        result["no_fg_accuracy"] = variant_accuracies["no_fg"]
    if "mixed_same" in variant_accuracies and "mixed_rand" in variant_accuracies:
        result["bg_gap"] = variant_accuracies["mixed_same"] - variant_accuracies["mixed_rand"]
    if "original" in variant_accuracies:
        result["original_accuracy"] = variant_accuracies["original"]
    return result

