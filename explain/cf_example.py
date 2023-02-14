from typing import Dict, List, Tuple

import torch


class DeepCounterfactualExplainer:
    def __init__(self,
                 encoder: torch.nn.Module,
                 decoder: torch.nn.Module,
                 classifier: torch.nn.Module,
                 target_feature: str):
        self.encoder = encoder
        self.decoder = decoder
        self.classifier = classifier
        self.target_feature = target_feature

    def explain(self, x: torch.Tensor,
                attrs: Dict[str, torch.Tensor],
                target_class: int,
                sample_points=100) -> Tuple[torch.Tensor, torch.Tensor]:
        codes = self.encoder(x, attrs)
        codes = codes.repeat(sample_points, *[1 for _ in codes.shape[1:]])

        with torch.no_grad():
            original_class = self.classifier(x).argmax(1).cpu().item()

        cf_attrs = {
            k: attrs[k].repeat(sample_points, *[1 for _ in attrs[k].shape[1:]])
            for k in attrs
            if k != self.target_feature
        }

        eye = torch.eye(attrs[self.target_feature].shape[1])
        eye_original = eye[original_class].reshape((1, eye.shape[1])).repeat(sample_points, 1)
        eye_target = eye[target_class].reshape((1, eye.shape[1])).repeat(sample_points, 1)
        probs = torch.linspace(0, 1, sample_points).reshape((sample_points, 1))
        cf_attrs[self.target_feature] = (1 - probs) * eye_original + probs * eye_target

        with torch.no_grad():
            samples = self.decoder(codes, cf_attrs)
            preds = self.classifier(samples).argmax(1)

            if (preds != target_class).sum() == sample_points:
                raise ValueError(f"Failed to flip the class label from class {original_class}"
                                 f" to {target_class}")

            probs = probs[preds == target_class]
            samples = samples[preds == target_class]
            sorted_inds = probs.argsort()
            return samples[sorted_inds], probs[sorted_inds]
