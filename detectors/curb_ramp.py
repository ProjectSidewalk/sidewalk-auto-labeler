import threading

import torch
from transformers import AutoModel
import numpy as np
from torchvision import transforms
from skimage.feature import peak_local_max


class CurbRampDetector:
    def __init__(self):
        # detect() is called from many download threads; concurrent full-resolution
        # forward passes would exhaust GPU memory, so device work is serialized.
        self._inference_lock = threading.Lock()
        if torch.cuda.is_available():
            self.DEVICE = torch.device("cuda")
        elif torch.backends.mps.is_available():  # Apple Silicon
            self.DEVICE = torch.device("mps")
        else:
            self.DEVICE = torch.device("cpu")

        self.model = AutoModel.from_pretrained("projectsidewalk/rampnet-model", trust_remote_code=True).to(self.DEVICE).eval()

    def detect(self, pil_image):
        preprocess = transforms.Compose([
            transforms.Resize((2048, 4096), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        img_tensor = preprocess(pil_image).unsqueeze(0)

        with self._inference_lock, torch.no_grad():
            heatmap = self.model(img_tensor.to(self.DEVICE)).squeeze().cpu().numpy()

        peaks = peak_local_max(np.clip(heatmap, 0, 1), min_distance=10, threshold_abs=0.55)

        detections = [(float(c / heatmap.shape[1]), float(r / heatmap.shape[0]), float(heatmap[r][c])) for r, c in peaks]

        return detections