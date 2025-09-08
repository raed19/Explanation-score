import numpy as np
import torch
from lime import lime_image
from skimage.segmentation import mark_boundaries

explainer = lime_image.LimeImageExplainer()

def preprocess_for_lime(image_tensor: np.ndarray) -> np.ndarray:
    img = image_tensor.copy().transpose(1, 2, 0)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img

def explain_prediction(image_chw: np.ndarray, label: int, model, device):
    def batch_predict(images):
        images = torch.tensor(images.transpose(0,3,1,2)).float().to(device)
        with torch.no_grad():
            outputs = model(images)
        return outputs.detach().cpu().numpy()

    explanation = explainer.explain_instance(
        preprocess_for_lime(image_chw),
        batch_predict,
        top_labels=5,
        hide_color=0,
        num_samples=1000
    )

    if label not in explanation.top_labels:
        label = int(explanation.top_labels[0])

    temp, mask = explanation.get_image_and_mask(
        label, positive_only=True, num_features=5, hide_rest=False
    )

    mask = (mask > 0).astype(np.uint8)
    temp = (temp * 255).astype(np.uint8)
    gray = np.full_like(temp, 128)
    gray[mask == 1] = temp[mask == 1]
    return mark_boundaries(gray, mask), mask
