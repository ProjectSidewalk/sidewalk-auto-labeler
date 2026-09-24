import threading

import torch
from transformers import AutoModel
import numpy as np
from torchvision import transforms
from skimage.feature import peak_local_max

from detectors import (DETECTION_STORAGE_FLOOR, MAX_PEAKS_PER_PANO, MODEL_REPO,
                       load_with_offline_fallback, provenance_for_loaded_model)


def _cached_snapshot_file():
    """Path of config.json in the hub-cache snapshot `main` resolves to, or None.

    Reads only the local cache (no network), so it works on an offline compute node; it is
    the snapshot from_pretrained just loaded, because both resolve the same ref.
    """
    from huggingface_hub import try_to_load_from_cache
    path = try_to_load_from_cache(MODEL_REPO, "config.json")
    return path if isinstance(path, str) else None


class CurbRampDetector:
    """RampNet over one equirectangular pano, plus the provenance of the weights it runs.

    ``provenance`` is resolved from the snapshot actually loaded (issue #39), never
    declared: construction raises detectors.ModelProvenanceError when the revision cannot be
    resolved, or is missing from detectors.KNOWN_REVISIONS and ``allow_unknown_revision`` is
    False, so no record is ever written under a guessed identity.
    """

    def __init__(self, allow_unknown_revision=False):
        # detect() is called from many download threads; concurrent full-resolution
        # forward passes would exhaust GPU memory, so device work is serialized.
        self._inference_lock = threading.Lock()
        if torch.cuda.is_available():
            self.DEVICE = torch.device("cuda")
        elif torch.backends.mps.is_available():  # Apple Silicon
            self.DEVICE = torch.device("mps")
        else:
            self.DEVICE = torch.device("cpu")

        model = load_with_offline_fallback(AutoModel.from_pretrained, MODEL_REPO,
                                           trust_remote_code=True)
        # transformers records the commit it resolved in config._commit_hash; the hub
        # cache's snapshots/<sha> directory is the fallback for a load that did not set it.
        commit_hash = getattr(model.config, '_commit_hash', None)
        self.provenance = provenance_for_loaded_model(
            commit_hash, _cached_snapshot_file, allow_unknown=allow_unknown_revision)
        self.model = model.to(self.DEVICE).eval()

    def detect(self, pil_image):
        preprocess = transforms.Compose([
            transforms.Resize((2048, 4096), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        img_tensor = preprocess(pil_image).unsqueeze(0)

        with self._inference_lock, torch.no_grad():
            heatmap = self.model(img_tensor.to(self.DEVICE)).squeeze().cpu().numpy()

        # Peaks are stored down to the storage floor; the operational threshold is
        # applied by consumers, not here (see detectors/__init__.py). num_peaks keeps
        # the highest-intensity peaks, so the >= OPERATIONAL_CONFIDENCE set is
        # unaffected by the lower floor.
        peaks = peak_local_max(np.clip(heatmap, 0, 1), min_distance=10,
                               threshold_abs=DETECTION_STORAGE_FLOOR,
                               num_peaks=MAX_PEAKS_PER_PANO)

        detections = [(float(c / heatmap.shape[1]), float(r / heatmap.shape[0]), float(heatmap[r][c])) for r, c in peaks]

        return detections