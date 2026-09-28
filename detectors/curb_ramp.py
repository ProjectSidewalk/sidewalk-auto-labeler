import threading
import time

import torch
from transformers import AutoModel
import numpy as np
from torchvision import transforms
from skimage.feature import peak_local_max

from detectors import (DETECTION_STORAGE_FLOOR, MAX_PEAKS_PER_PANO, MODEL_REPO,
                       load_with_offline_fallback, provenance_for_loaded_model)
from detectors.batching import DEFAULT_BATCH_WAIT_S, Batcher, ForwardStats

# The model's fixed input size (rampnet_model.PANO_INPUT_SIZE) and ImageNet normalization.
INPUT_SIZE = (2048, 4096)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def detections_from_heatmap(heatmap):
    """Heatmap peaks -> normalized ``[(x, y, confidence), ...]``, highest first.

    Peaks are stored down to the storage floor; the operational threshold is applied by
    consumers, not here (see detectors/__init__.py). num_peaks keeps the highest-intensity
    peaks, so the >= OPERATIONAL_CONFIDENCE set is unaffected by the lower floor. This is
    per image whether or not the forward pass was batched.
    """
    peaks = peak_local_max(np.clip(heatmap, 0, 1), min_distance=10,
                           threshold_abs=DETECTION_STORAGE_FLOOR,
                           num_peaks=MAX_PEAKS_PER_PANO)
    return [(float(c / heatmap.shape[1]), float(r / heatmap.shape[0]), float(heatmap[r][c]))
            for r, c in peaks]


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

    Concurrency (issue #2). ``detect()`` is called from many fetch threads. Preprocessing
    (resize to 2048x4096, ToTensor, Normalize) runs in the calling thread, so those threads
    preprocess in parallel; only the forward pass is serialized, because concurrent
    full-resolution passes would exhaust GPU memory.

    - ``batch_size=1`` (default): each call takes the inference lock and runs its own
      forward pass -- the pre-#2 path, bit-identical detections.
    - ``batch_size=N>1``: calls hand their tensors to a ``detectors.batching.Batcher``; one
      consumer thread runs a single forward over up to N stacked images, waiting at most
      ``batch_wait_s`` for a batch to fill, and each caller gets its own image's heatmap
      back. Peak extraction stays per image and unchanged. An exception in a batched
      forward (e.g. CUDA OOM) is raised in every caller of that batch, which main._process
      records as a retryable failure. Memory: each queued image is one 3x2048x4096 float32
      tensor (~100 MB), and the stacked batch is a second copy on the device.

    ``stats()`` reports forward passes, images, mean batch fill and seconds in the forward,
    so a batching gain is measured, not assumed. ``close()`` (or ``with``) stops the
    consumer after draining; it is a daemon thread, so forgetting to close never blocks
    process exit.

    Example::

        det = CurbRampDetector(batch_size=4)
        with ThreadPoolExecutor(8) as pool:
            results = list(pool.map(det.detect, images))
        print(det.stats())
        det.close()
    """

    def __init__(self, allow_unknown_revision=False, batch_size=1,
                 batch_wait_s=DEFAULT_BATCH_WAIT_S):
        if batch_size < 1:
            raise ValueError('batch_size must be >= 1')
        # Serializes device work: concurrent full-resolution forward passes would exhaust
        # GPU memory. With batching, only the consumer thread takes it.
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

        # Built once (it was rebuilt on every call before #2); stateless, so thread-safe.
        self._preprocess = transforms.Compose([
            transforms.Resize(INPUT_SIZE, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        self.batch_size = batch_size
        # batch_size 1 starts no consumer thread: detect() stays on the pre-#2 path.
        self._batcher = (Batcher(self._forward_batch, batch_size, batch_wait_s)
                         if batch_size > 1 else None)
        self._stats = ForwardStats(1) if self._batcher is None else None

    def heatmap(self, pil_image):
        """The model's 512x1024 heatmap for one image, as a float32 numpy array."""
        img_tensor = self._preprocess(pil_image)  # CPU, in the calling thread
        if self._batcher is not None:
            return self._batcher.submit(img_tensor)
        img_tensor = img_tensor.unsqueeze(0)
        with self._inference_lock, torch.no_grad():
            t0 = time.perf_counter()
            heatmap = self.model(img_tensor.to(self.DEVICE)).squeeze().cpu().numpy()
            self._stats.record(1, time.perf_counter() - t0)
        return heatmap

    def detect(self, pil_image):
        """Normalized ``[(x, y, confidence), ...]`` for one pano, down to the storage floor."""
        return detections_from_heatmap(self.heatmap(pil_image))

    def _forward_batch(self, tensors):
        """One forward over stacked tensors -> one 512x1024 heatmap per input, in order.
        Runs on the batcher's consumer thread."""
        with self._inference_lock, torch.no_grad():
            batch = torch.stack(tensors).to(self.DEVICE)
            out = self.model(batch).cpu().numpy()
        if out.ndim != 4 or out.shape[:2] != (len(tensors), 1):
            raise RuntimeError(f'expected a (B, 1, H, W) heatmap batch, got {out.shape}')
        return [out[i, 0] for i in range(out.shape[0])]

    def stats(self):
        """Forward-pass tally: see detectors.batching.ForwardStats.snapshot."""
        return (self._batcher.stats() if self._batcher is not None
                else self._stats.snapshot())

    def close(self):
        """Drain and stop the batch consumer (no-op at batch_size 1). Idempotent."""
        if self._batcher is not None:
            self._batcher.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
