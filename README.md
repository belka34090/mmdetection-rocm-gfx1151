# MMDetection + ROCm on gfx1151

Reproducible Docker stack for running MMDetection and TOOD with native MMCV HIP operations on AMD ROCm `gfx1151`.

## Validated hardware

- AMD Radeon 8060S
- AMD Ryzen AI Max+ 395
- GPU architecture: `gfx1151`

## Validated stack

| Component | Version |
|---|---|
| Ubuntu | 24.04 |
| Python | 3.12.3 |
| ROCm SDK | 7.14.1 |
| PyTorch | 2.12.0+rocm7.14.1 |
| MMCV | 2.1.0 |
| MMEngine | 0.10.7 |
| MMDetection | 3.3.0 |
| OpenCV | 5.0.0 headless |
| TOOD | R101 + DCNv2 |

MMCV is compiled from source with HIP operations for `gfx1151`.

MMCV 2.1.0 is patched from C++17 to C++20 for the validated PyTorch 2.12 extension toolchain.

The ROCm development package is installed explicitly to provide the development headers and libraries required by the MMCV build, including rocThrust and `libamdhip64`.

Python dependencies are explicitly pinned and installed without transitive dependency resolution. This prevents MMEngine from installing the GUI `opencv-python` package alongside `opencv-python-headless`, so the container has a single OpenCV provider suitable for headless GPU workloads.

## Build

    docker build -t mmdetection-rocm-gfx1151 .

## Run

    docker run --rm \
      --device=/dev/kfd \
      --device=/dev/dri \
      --group-add "$(getent group video | cut -d: -f3)" \
      --group-add "$(getent group render | cut -d: -f3)" \
      --ipc=host \
      --shm-size=16g \
      mmdetection-rocm-gfx1151

Expected final validation markers:

    MMCV_GPU_OK
    TOOD_R101_DCN_GPU_OK

The smoke test validates:

1. PyTorch visibility of the AMD GPU through ROCm.
2. Import of the compiled `mmcv._ext` extension.
3. Native MMCV NMS execution on the GPU.
4. Construction of TOOD R101 with DCNv2.
5. A real 512x512 TOOD forward pass on the GPU.

## Why PyTorch shows cuda:0

PyTorch keeps the `torch.cuda` API namespace when using its ROCm backend.

Therefore:

    Device : cuda:0

is normal with an AMD GPU using ROCm. It does not mean NVIDIA CUDA is being used.

## MMCV patch

The repository contains:

    patches/mmcv-cxx20.patch

It changes MMCV 2.1.0 extension compilation from C++17 to C++20.

## Scope

This repository reproduces the configuration validated on AMD `gfx1151`, specifically the Radeon 8060S.

It does not claim validation on every AMD GPU architecture.

## Third-party projects

This stack builds on:

- AMD ROCm
- PyTorch
- OpenMMLab MMCV
- OpenMMLab MMEngine
- OpenMMLab MMDetection

Their respective licenses remain applicable.

## Reproducibility pins

The validated environment is pinned to the following exact upstream revisions:

- ROCm/PyTorch image:
  `sha256:cc9b00f90b85c97b015b040fa55c8d1b404b7cacc6ad57d74ee3451c97508da1`
- MMEngine:
  `390ba2fbb272816adfd2883642326d0fd0ca6049`
- MMCV:
  `57c4e25e06e2d4f8a9357c84bcd24089a284dc88`
- MMDetection:
  `44ebd17b145c2372c4b700bfb9cb20dbd28ab64a`

The Docker image uses a multi-stage build. ROCm development headers and build-time source trees are used only in the builder stage and are not kept in the final runtime image.

## Known non-blocking ROCm messages

On the validated Radeon 8060S / `gfx1151` environment, ROCm may print messages such as:

    MIOpen(HIP): Warning [ParseAndLoadDb] File is unreadable: ".../gfx1151_20.HIP.fdb.txt"
    warning: xnack 'Off' was requested for a processor that does not support it!

Apex may also report:

    Using the native apex kernel for RoPE.

These messages were present during the validated GPU smoke tests and did not prevent:

- MMCV native GPU NMS
- DCNv2 execution
- TOOD R101 model construction
- a complete 512x512 GPU forward pass

They are therefore documented rather than suppressed.
