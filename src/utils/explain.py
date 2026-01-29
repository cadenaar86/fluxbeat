import torch
import torch.nn.functional as F
import numpy as np
import cv2

class GradCAM:
    """
    Grad-CAM: Gradient-weighted Class Activation Mapping.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.hooks = []
        self._register_hooks()
        
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output
            
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
            
        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self.hooks.append(self.target_layer.register_full_backward_hook(backward_hook))
        
    def generate_cam(self, input_tensor, target_class_idx=None):
        """
        Generate CAM for a specific class.
        Args:
            input_tensor: (1, C, H, W)
            target_class_idx: int or None. If None, uses max prediction.
            
        Returns:
            heatmap: (H, W) numpy array, normalized 0-1
        """
        self.model.eval()
        self.model.zero_grad()
        
        # Forward pass
        output = self.model(input_tensor)
        
        if target_class_idx is None:
            target_class_idx = torch.argmax(output)
            
        # Backward pass
        # Target score for backprop
        target = output[0, target_class_idx]
        target.backward()
        
        # Gradients: (1, C, H', W')
        gradients = self.gradients
        # Activations: (1, C, H', W')
        activations = self.activations
        
        # Global Average Pooling of gradients -> Weights
        # (1, C, H', W') -> (1, C)
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        
        # Weighted sum of activations
        # (1, C, H', W') * (1, C, 1, 1) -> (1, C, H', W') -> Sum -> (1, H', W')
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        
        # ReLU (we only care about positive influence)
        cam = F.relu(cam)
        
        # Normalize 0-1
        cam = cam.squeeze().detach().cpu().numpy() # (H', W')
        if np.max(cam) > 0:
            cam = cam / np.max(cam)
            
        # Upsample to input size
        input_h, input_w = input_tensor.shape[2], input_tensor.shape[3]
        cam = cv2.resize(cam, (input_w, input_h))
        
        return cam

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
