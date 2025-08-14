import torch
from transformers import AutoModel
import numpy as np
from torchvision import transforms
from skimage.feature import peak_local_max


class CurbRampDetector:
    def __init__(self):
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = AutoModel.from_pretrained("projectsidewalk/rampnet-model", trust_remote_code=True).to(self.DEVICE).eval()

    def detect(self, pil_image):
        preprocess = transforms.Compose([
            transforms.Resize((2048, 4096), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        img_tensor = preprocess(pil_image).unsqueeze(0).to(self.DEVICE)

        with torch.no_grad():
            heatmap = self.model(img_tensor).squeeze().cpu().numpy()

        peaks = peak_local_max(np.clip(heatmap, 0, 1), min_distance=10, threshold_abs=0.5)
        scale_w = pil_image.width / heatmap.shape[1]
        scale_h = pil_image.height / heatmap.shape[0]
        # coordinates = [(int(c * scale_w), int(r * scale_h)) for r, c in peaks]
        coordinates = [(float(c), float(r)) for r, c in peaks]

        return coordinates