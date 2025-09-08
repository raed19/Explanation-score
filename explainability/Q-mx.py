import numpy as np
import cv2

def compute_explainability_score(explanation_mask, object_mask, grid_size=(3,3), penalty_factor=0.1):
    if explanation_mask.ndim == 3:
        explanation_mask = cv2.cvtColor(explanation_mask, cv2.COLOR_BGR2GRAY)

    explanation_mask = (explanation_mask > explanation_mask.mean()).astype(np.uint8)
    object_mask = (object_mask > 0).astype(np.uint8)

    if explanation_mask.shape != object_mask.shape:
        explanation_mask = cv2.resize(explanation_mask, (object_mask.shape[1], object_mask.shape[0]))

    h, w = object_mask.shape
    gh, gw = h // grid_size[0], w // grid_size[1]

    weights = np.zeros_like(object_mask, dtype=float)
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            region_weight = ((grid_size[0] - abs(i - grid_size[0]//2)) *
                             (grid_size[1] - abs(j - grid_size[1]//2)))
            weights[i*gh:(i+1)*gh, j*gw:(j+1)*gw] = region_weight
    weights /= weights.max() if weights.max() > 0 else 1.0

    valid = weights * object_mask * explanation_mask
    denom = (weights * object_mask).sum()
    s_align = valid.sum() / denom if denom != 0 else 0.0

    irrelevant = weights * (1 - object_mask) * explanation_mask
    penalty = penalty_factor * (irrelevant.sum() / (weights.sum() if weights.sum() > 0 else 1.0))

    return max(s_align - penalty, 0.0)
