import numpy as np
import torch

class ActiveLearningSelector:
    def __init__(self, model, mc_passes=10, device="cpu"):
        self.model = model
        self.mc_passes = mc_passes
        self.device = device

    def uncertainty_sampling(self, unlabeled_list, n_samples=10):
        self.model.train()  # Enable dropout at inference time for MC dropout
        scores = []
        with torch.no_grad():
            for data in unlabeled_list:
                data = data.to(self.device)
                preds = []
                for _ in range(self.mc_passes):
                    y, _ = self.model(data)
                    preds.append(y.cpu().numpy())
                preds = np.array(preds)
                uncertainty = float(np.var(preds))
                scores.append(uncertainty)
        idxs = np.argsort(scores)[-n_samples:]
        return idxs.tolist(), [scores[i] for i in idxs]
