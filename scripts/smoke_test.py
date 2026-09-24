from pathlib import Path

import torch
import mmcv
import mmcv._ext
import mmengine
import mmdet

from mmcv.ops import nms
from mmengine.config import Config
from mmdet.registry import MODELS
from mmdet.utils import register_all_modules


MMDET_ROOT = Path("/opt/mmdetection")
CONFIG = (
    MMDET_ROOT
    / "configs"
    / "tood"
    / "tood_r101-dconv-c3-c5_fpn_ms-2x_coco.py"
)


def test_environment() -> None:
    print("=== Environment ===")
    print("PyTorch     :", torch.__version__)
    print("ROCm        :", torch.version.hip)
    print("GPU         :", torch.cuda.get_device_name(0))
    print("MMCV        :", mmcv.__version__)
    print("MMEngine    :", mmengine.__version__)
    print("MMDetection :", mmdet.__version__)
    print("mmcv._ext   : OK")

    assert torch.cuda.is_available()
    assert mmcv.__version__ == "2.1.0"
    assert mmengine.__version__ == "0.10.7"
    assert mmdet.__version__ == "3.3.0"


def test_nms() -> None:
    print("\n=== MMCV NMS GPU ===")

    boxes = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0],
            [1.0, 1.0, 11.0, 11.0],
            [30.0, 30.0, 40.0, 40.0],
        ],
        device="cuda",
    )

    scores = torch.tensor(
        [0.9, 0.8, 0.7],
        device="cuda",
    )

    dets, inds = nms(boxes, scores, 0.5)
    torch.cuda.synchronize()

    indices = inds.cpu().tolist()

    print("Indices :", indices)
    print("Device  :", dets.device)

    assert indices == [0, 2]
    assert dets.is_cuda

    print("MMCV_GPU_OK")


def test_tood() -> None:
    print("\n=== TOOD R101 + DCNv2 GPU ===")

    register_all_modules()

    if not CONFIG.is_file():
        raise FileNotFoundError(CONFIG)

    cfg = Config.fromfile(str(CONFIG))
    cfg.model.backbone.init_cfg = None

    model = MODELS.build(cfg.model)
    model = model.cuda()
    model.eval()

    params = sum(p.numel() for p in model.parameters())

    print("Config     :", CONFIG)
    print("Parameters :", f"{params:,}")
    print("Device     :", next(model.parameters()).device)

    x = torch.randn(
        1,
        3,
        512,
        512,
        device="cuda",
        dtype=torch.float32,
    )

    with torch.inference_mode():
        model(x, mode="tensor")
        torch.cuda.synchronize()

    print("TOOD_R101_DCN_GPU_OK")


if __name__ == "__main__":
    test_environment()
    test_nms()
    test_tood()
