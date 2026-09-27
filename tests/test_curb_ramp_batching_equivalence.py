"""Batched inference gives the detections unbatched inference gives (issue #2).

OPT-IN: it runs real forward passes on CPU, several minutes even downscaled, so it is
skipped unless RAMPNET_EQUIVALENCE is set, and it also skips without torch (CI's
requirements-test.txt has none), without the cached RampNet weights, or without the local
RampNet bundle panos. It never touches the network. CPU, so the numbers do not depend on
whichever GPU happens to be free.

    RAMPNET_EQUIVALENCE=1     pytest tests/test_curb_ramp_batching_equivalence.py -s
    RAMPNET_EQUIVALENCE=full  pytest tests/test_curb_ramp_batching_equivalence.py -s

``1`` feeds 512x1024 inputs (~25 s per CPU forward, ~4 min in all); the model is fully
convolutional up to a fixed-size upsample, so this exercises the same stacking, splitting
and peak code. ``full`` feeds the production 2048x4096 input (~95 s per CPU forward), where
the unbatched comparison is literally the pre-#2 detect() code, inlined below.
"""
import os
import threading
from pathlib import Path

import pytest

MODE = os.environ.get('RAMPNET_EQUIVALENCE')
if MODE not in ('1', 'full'):
    pytest.skip('opt-in: set RAMPNET_EQUIVALENCE=1 (512x1024) or =full (2048x4096)',
                allow_module_level=True)

torch = pytest.importorskip('torch')
pytest.importorskip('transformers')
pytest.importorskip('torchvision')
pytest.importorskip('skimage')

from PIL import Image  # noqa: E402

from detectors import MODEL_REPO  # noqa: E402

BUNDLE = Path(__file__).resolve().parents[2] / 'RampNet' / 'benchmark' / 'paterson' / 'panos'
# Three bundle panos with 8-10 stored detections each at 0.55 in the run's own records.
PANO_IDS = ['AhV3GKVFnENcHwhG45WN2g', 'tekhQ4HQ9pcOqs_kGGpGaw', '0Drku25sOlOlWGiVf7uetw']
SIZE = (2048, 4096) if MODE == 'full' else (512, 1024)


def _model_cached():
    from huggingface_hub import try_to_load_from_cache
    return isinstance(try_to_load_from_cache(MODEL_REPO, 'model.safetensors'), str)


def _panos():
    paths = [BUNDLE / f'{pid}.jpg' for pid in PANO_IDS]
    if not all(p.exists() for p in paths):
        pytest.skip(f'bundle panos not found under {BUNDLE}')
    Image.MAX_IMAGE_PIXELS = None  # native 16384x8192 bundle JPEGs
    # Production feeds 4096x2048 (panorama.py); the detector resizes to SIZE from there.
    return [Image.open(p).convert('RGB').resize((4096, 2048), Image.BILINEAR) for p in paths]


def _old_detect(model, device, pil_image, size):
    """The pre-#2 CurbRampDetector.detect, verbatim except for the input size.
    Returns (detections, heatmap)."""
    import numpy as np
    from skimage.feature import peak_local_max
    from torchvision import transforms
    from detectors import DETECTION_STORAGE_FLOOR, MAX_PEAKS_PER_PANO
    preprocess = transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = preprocess(pil_image).unsqueeze(0)
    with torch.no_grad():
        heatmap = model(img_tensor.to(device)).squeeze().cpu().numpy()
    peaks = peak_local_max(np.clip(heatmap, 0, 1), min_distance=10,
                           threshold_abs=DETECTION_STORAGE_FLOOR,
                           num_peaks=MAX_PEAKS_PER_PANO)
    return ([(float(c / heatmap.shape[1]), float(r / heatmap.shape[0]), float(heatmap[r][c]))
             for r, c in peaks], heatmap)


@pytest.fixture(scope='module')
def cpu_detectors():
    if not _model_cached():
        pytest.skip('RampNet weights are not in the local Hugging Face cache')
    from torchvision import transforms
    from detectors import curb_ramp
    mp = pytest.MonkeyPatch()
    mp.setattr(torch.cuda, 'is_available', lambda: False)
    mp.setattr(torch.backends.mps, 'is_available', lambda: False)
    try:
        single = curb_ramp.CurbRampDetector(batch_size=1)
        # A long wait bound so three concurrent callers are certain to share one batch.
        batched = curb_ramp.CurbRampDetector(batch_size=3, batch_wait_s=30.0)
    finally:
        mp.undo()
    if SIZE != curb_ramp.INPUT_SIZE:
        small = transforms.Compose([
            transforms.Resize(SIZE, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=curb_ramp.IMAGENET_MEAN, std=curb_ramp.IMAGENET_STD)])
        single._preprocess = batched._preprocess = small
    yield single, batched
    batched.close()


def test_batched_equals_unbatched(cpu_detectors):
    import numpy as np
    from detectors.curb_ramp import detections_from_heatmap
    single, batched = cpu_detectors
    images = _panos()

    # (a) batch_size=1 reproduces the pre-#2 algorithm exactly.
    old = [_old_detect(single.model, single.DEVICE, im, SIZE) for im in images]
    new = [single.detect(im) for im in images]
    assert [o[0] for o in old] == new
    assert sum(len(d) for d in new) > 0, 'no detections: the comparison would be vacuous'
    s1 = single.stats()
    assert s1['forward_passes'] == s1['images'] == len(images)

    # (b) three concurrent callers share ONE forward and each gets its own heatmap.
    heatmaps = [None] * len(images)
    errors = []

    def call(i):
        try:
            heatmaps[i] = batched.heatmap(images[i])
        except Exception as e:  # pragma: no cover - surfaced by the assert below
            errors.append(e)
    threads = [threading.Thread(target=call, args=(i,)) for i in range(len(images))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    s3 = batched.stats()
    assert s3['forward_passes'] == 1 and s3['images'] == len(images)

    max_diff = max(float(np.abs(hb - o[1]).max()) for hb, o in zip(heatmaps, old))
    print(f'\nsize {SIZE}: max |heatmap batched - single| = {max_diff:.3g}; '
          f'single forward {s1["forward_seconds"] / s1["images"]:.2f} s/image, '
          f'batch-of-{len(images)} forward {s3["forward_seconds"]:.2f} s; '
          f'detections per image {[len(d) for d in new]}')
    assert max_diff < 1e-4
    for hb, (old_dets, _) in zip(heatmaps, old):
        dets = detections_from_heatmap(hb)
        assert [(x, y) for x, y, _ in dets] == [(x, y) for x, y, _ in old_dets]
        assert all(abs(c - oc) < 1e-4 for (_, _, c), (_, _, oc) in zip(dets, old_dets))
