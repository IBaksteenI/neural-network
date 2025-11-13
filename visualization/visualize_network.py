import os
import sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import matplotlib.animation as animation
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection
import matplotlib.patheffects as path_effects

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.neural_network import NeuralNet

def get_device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def get_test_loader(batch_size=100, normalize=False, num_workers=2):
    tfms = [transforms.ToTensor()]
    if normalize:
        tfms.append(transforms.Normalize((0.1307,), (0.3081,)))
    transform = transforms.Compose(tfms)

    testset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=True, num_workers=num_workers, persistent_workers=False
    )
    return loader

def load_model(weights_path="mnist_model.pth", device=None):
    if device is None:
        device = get_device()
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Missing weights file: {weights_path}")
    model = NeuralNet().to(device)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model

def get_activations(model, image, device):
    """Extract activation values from all layers for a given input image"""
    activations = {}
    
    def hook_fn(name):
        def _fn(module, inp, out):
            activations[name] = out.detach().cpu()
        return _fn
    
    hooks = []
    hooks.append(model.fc1.register_forward_hook(hook_fn("fc1")))
    hooks.append(model.fc2.register_forward_hook(hook_fn("fc2")))
    hooks.append(model.fc3.register_forward_hook(hook_fn("fc3")))
    
    with torch.no_grad():
        output = model(image.unsqueeze(0).to(device))
        probs = torch.softmax(output, dim=1).cpu()
    
    for h in hooks:
        h.remove()
    
    input_flat = image.view(-1).cpu().numpy()
    fc1_act = torch.relu(activations["fc1"]).squeeze().numpy()
    fc2_act = torch.relu(activations["fc2"]).squeeze().numpy()
    fc3_act = activations["fc3"].squeeze().numpy()
    probs_np = probs.squeeze().numpy()
    
    return {
        'input': input_flat,
        'fc1': fc1_act,
        'fc2': fc2_act,
        'fc3': fc3_act,
        'probs': probs_np
    }

def draw_neural_network(ax, activations, model):
    """Draw network showing most active neurons with connections"""
    ax.clear()
    ax.set_facecolor('white')
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-0.5, 10.5)
    ax.axis('off')
    
    layer_sizes = [784, 128, 64, 10]
    layer_names = ['Input\n784', 'Hidden 1\n128', 'Hidden 2\n64', 'Output\n10']
    x_positions = [0, 1.5, 3, 4.2]
    show_per_side = [15, 15, 12, 10]
    
    act_values = [
        activations['input'],
        activations['fc1'],
        activations['fc2'],
        activations['probs']
    ]
    
    def normalize(arr):
        arr = np.abs(arr)
        max_val = arr.max() if arr.max() > 1e-6 else 1.0
        return np.clip(arr / max_val, 0, 1)
    
    norm_acts = [normalize(a) for a in act_values]
    
    # Build positions for subset of neurons (most active ones)
    node_positions = []
    all_activations = []
    
    for layer_idx, (size, x, norm_act, show_n) in enumerate(zip(layer_sizes, x_positions, norm_acts, show_per_side)):
        all_activations.append(norm_act)
        
        if size <= 10:
            y_positions = np.linspace(1, 9, size)
            indices = list(range(size))
            positions = [(x, y, idx, act) for y, idx, act in zip(y_positions, indices, norm_act)]
            node_positions.append(positions)
        else:
            # Show most active neurons
            top_active = np.argsort(norm_act)[-show_n:][::-1]
            mid = len(top_active) // 2
            top_indices = top_active[:mid]
            bottom_indices = top_active[mid:]
            
            y_top = np.linspace(1, 4, len(top_indices))
            y_bottom = np.linspace(6, 9, len(bottom_indices))
            
            positions = []
            for y, idx in zip(y_top, top_indices):
                positions.append((x, y, idx, norm_act[idx]))
            
            positions.append((x, 5, -1, 0))  # "..." marker
            
            for y, idx in zip(y_bottom, bottom_indices):
                positions.append((x, y, idx, norm_act[idx]))
            
            node_positions.append(positions)
    
    weights = [
        model.fc1.weight.detach().cpu().numpy(),
        model.fc2.weight.detach().cpu().numpy(),
        model.fc3.weight.detach().cpu().numpy(),
    ]
    
    # Draw connections
    for layer_idx in range(len(node_positions) - 1):
        from_layer = [p for p in node_positions[layer_idx] if p[2] != -1]
        to_layer = [p for p in node_positions[layer_idx + 1] if p[2] != -1]
        W = weights[layer_idx]
        
        from_acts = all_activations[layer_idx]
        to_acts = all_activations[layer_idx + 1]
        
        for x2, y2, to_idx, to_act in to_layer:
            for x1, y1, from_idx, from_act in from_layer:
                w = W[to_idx, from_idx]
                strength = abs(w) * from_acts[from_idx] * to_acts[to_idx]
                
                if strength > 0.02:
                    color = '#A9A9A9'
                    alpha = min(0.9, 0.3 + strength * 3)
                    lw = 0.5 + min(2.0, strength * 10)
                else:
                    color = '#E8E8E8'
                    alpha = 0.15
                    lw = 0.25
                
                ax.plot([x1, x2], [y1, y2], color=color, linewidth=lw, 
                       alpha=alpha, zorder=1, solid_capstyle='round')
    
    # Draw neurons
    for layer_idx, (size, x, positions) in enumerate(zip(layer_sizes, x_positions, node_positions)):
        for i, (x_pos, y_pos, idx, activation) in enumerate(positions):
            if idx == -1:
                ax.text(x_pos, y_pos, '⋮', fontsize=20, ha='center', va='center', 
                       color='#666666', weight='bold')
                continue
            
            base_size = 150
            threshold = 0.01 if layer_idx == 3 else 0.1
            
            if activation > threshold:
                intensity = min(1.0, activation * 2)
                color = '#A9A9A9'
                edge_color = '#A9A9A9'
                size_mult = 1.0 + (intensity * 0.2)
                alpha = 0.8 + (intensity * 0.2)
            else:
                color = 'white'
                edge_color = '#AAAAAA'
                size_mult = 1.0
                alpha = 0.5
            
            ax.scatter(x_pos, y_pos, s=base_size * size_mult, 
                      c=color, alpha=alpha, edgecolors=edge_color, 
                      linewidths=1.5, zorder=3)
            
            # Label output neurons
            if layer_idx == len(layer_sizes) - 1:
                txt = ax.text(x_pos + 0.25, y_pos, f"{idx}", 
                            fontsize=12, va='center', ha='left',
                            color='#000000', weight='bold', zorder=4)
                txt.set_path_effects([
                    path_effects.Stroke(linewidth=3, foreground='white'),
                    path_effects.Normal()
                ])
        
        txt = ax.text(x, -0.2, layer_names[layer_idx], 
                     ha='center', va='center', fontsize=11, 
                     color='#2C3E50', weight='bold')
    
    ax.text(2, 10.3, "Neural Network Activity", 
           ha='center', fontsize=14, weight='bold', color='#2C3E50')

def compute_confusion_matrix(model, device=None, normalize=False):
    """Compute confusion matrix and accuracy on test set"""
    if device is None:
        device = get_device()
    loader = get_test_loader(batch_size=512, normalize=normalize)
    num_classes = 10
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1)
            for t, p in zip(labels.view(-1), preds.view(-1)):
                cm[t.long().item(), p.long().item()] += 1
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total if total > 0 else 0.0
    return cm, acc

def visualize_all(model=None, testloader=None, device=None):
    """Main visualization with 7 informative graphs"""
    if device is None:
        device = get_device()
    
    if model is None:
        model = load_model("mnist_model.pth", device=device)
    elif isinstance(model, str):
        model = load_model(model, device=device)
    
    if testloader is None:
        testloader = get_test_loader(batch_size=100)

    model.eval()
    images, labels = next(iter(testloader))

    fig = plt.figure(figsize=(24, 13))
    fig.patch.set_facecolor('white')
    gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.4, 
                          left=0.05, right=0.98, top=0.95, bottom=0.05)
    
    ax_img = fig.add_subplot(gs[0, 0])
    ax_prob = fig.add_subplot(gs[1, 0])
    ax_net = fig.add_subplot(gs[:, 1:3])
    ax_confusion = fig.add_subplot(gs[2, 0])
    ax_layer_act = fig.add_subplot(gs[0, 3])
    ax_weight_dist = fig.add_subplot(gs[1, 3])
    ax_confidence = fig.add_subplot(gs[2, 3])
    
    cm, acc = compute_confusion_matrix(model, device=device)
    confidence_history = []
    
    def update(frame):
        img = images[frame]
        label = labels[frame].item()
        
        activations = get_activations(model, img, device)
        pred = np.argmax(activations['probs'])
        confidence = activations['probs'][pred]
        confidence_history.append(confidence)
        if len(confidence_history) > 50:
            confidence_history.pop(0)
        
        # Input image
        ax_img.clear()
        ax_img.imshow(img.squeeze(), cmap="gray")
        ax_img.set_title(f"Input Digit: {label}\nPrediction: {pred} ({confidence*100:.1f}%)", 
                        fontsize=11, weight='bold', color='#2C3E50')
        ax_img.axis("off")
        
        # Output probabilities
        ax_prob.clear()
        colors = ['#2ECC71' if i == pred else '#3498DB' for i in range(10)]
        bars = ax_prob.bar(np.arange(10), activations['probs'], 
                          color=colors, alpha=0.8, edgecolor='white', linewidth=1.5)
        ax_prob.set_title(f"Output Probabilities", fontsize=10, weight='bold', color='#2C3E50')
        ax_prob.set_xticks(np.arange(10))
        ax_prob.set_ylim(0, 1)
        ax_prob.set_ylabel("Probability", fontsize=9, color='#2C3E50')
        ax_prob.grid(axis='y', alpha=0.2, linestyle='--')
        ax_prob.spines['top'].set_visible(False)
        ax_prob.spines['right'].set_visible(False)
        if pred < len(bars):
            bars[pred].set_linewidth(3)
        
        # Network visualization
        draw_neural_network(ax_net, activations, model)
        
        # Confusion matrix
        ax_confusion.clear()
        cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
        im = ax_confusion.imshow(cm_norm, cmap='Blues', aspect='auto', vmin=0, vmax=1)
        ax_confusion.set_title(f"Confusion Matrix\nAccuracy: {acc*100:.2f}%", 
                              fontsize=10, weight='bold', color='#2C3E50')
        ax_confusion.set_xlabel("Predicted", fontsize=9)
        ax_confusion.set_ylabel("Actual", fontsize=9)
        ax_confusion.set_xticks(np.arange(10))
        ax_confusion.set_yticks(np.arange(10))
        ax_confusion.tick_params(labelsize=8)
        
        # Layer activations
        ax_layer_act.clear()
        layer_names = ['Input', 'Hidden1', 'Hidden2', 'Output']
        layer_means = [
            np.mean(activations['input']),
            np.mean(activations['fc1']),
            np.mean(activations['fc2']),
            np.mean(activations['probs'])
        ]
        layer_maxs = [
            np.max(activations['input']),
            np.max(activations['fc1']),
            np.max(activations['fc2']),
            np.max(activations['probs'])
        ]
        
        x = np.arange(len(layer_names))
        width = 0.35
        ax_layer_act.bar(x - width/2, layer_means, width, label='Mean', color='#3498DB', alpha=0.8)
        ax_layer_act.bar(x + width/2, layer_maxs, width, label='Max', color='#E74C3C', alpha=0.8)
        ax_layer_act.set_title("Layer Activations", fontsize=10, weight='bold', color='#2C3E50')
        ax_layer_act.set_ylabel("Activation Value", fontsize=9)
        ax_layer_act.set_xticks(x)
        ax_layer_act.set_xticklabels(layer_names, fontsize=8, rotation=15)
        ax_layer_act.legend(fontsize=8)
        ax_layer_act.grid(axis='y', alpha=0.2)
        ax_layer_act.spines['top'].set_visible(False)
        ax_layer_act.spines['right'].set_visible(False)
        
        # Weight distribution for predicted neuron
        ax_weight_dist.clear()
        output_weights = model.fc3.weight.detach().cpu().numpy()[pred]
        ax_weight_dist.hist(output_weights, bins=30, color='#9B59B6', alpha=0.7, edgecolor='black')
        ax_weight_dist.axvline(x=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax_weight_dist.set_title(f"Weight Distribution\nOutput Neuron {pred}", 
                                fontsize=10, weight='bold', color='#2C3E50')
        ax_weight_dist.set_xlabel("Weight Value", fontsize=9)
        ax_weight_dist.set_ylabel("Frequency", fontsize=9)
        ax_weight_dist.grid(axis='y', alpha=0.2)
        ax_weight_dist.spines['top'].set_visible(False)
        ax_weight_dist.spines['right'].set_visible(False)
        ax_weight_dist.tick_params(labelsize=8)
        
        # Confidence over time
        ax_confidence.clear()
        if len(confidence_history) > 1:
            ax_confidence.plot(confidence_history, color='#16A085', linewidth=2)
            ax_confidence.fill_between(range(len(confidence_history)), confidence_history, 
                                      alpha=0.3, color='#16A085')
        ax_confidence.set_title("Prediction Confidence", fontsize=10, weight='bold', color='#2C3E50')
        ax_confidence.set_xlabel("Sample #", fontsize=9)
        ax_confidence.set_ylabel("Confidence", fontsize=9)
        ax_confidence.set_ylim(0, 1)
        ax_confidence.axhline(y=0.9, color='green', linestyle='--', alpha=0.3, linewidth=1)
        ax_confidence.axhline(y=0.5, color='orange', linestyle='--', alpha=0.3, linewidth=1)
        ax_confidence.grid(axis='both', alpha=0.2)
        ax_confidence.spines['top'].set_visible(False)
        ax_confidence.spines['right'].set_visible(False)
        ax_confidence.tick_params(labelsize=8)
        
    ani = animation.FuncAnimation(fig, update, frames=len(images), 
                                 interval=1000, repeat=True)
    plt.show()

if __name__ == "__main__":
    device = get_device()
    print(f"Using device: {device}")

    model = load_model(device=device)
    testloader = get_test_loader(batch_size=100)
    visualize_all(model, testloader=testloader, device=device)