FROM rocm/pytorch:rocm7.14.1_ubuntu24.04_py3.12_pytorch_release_2.12.0@sha256:cc9b00f90b85c97b015b040fa55c8d1b404b7cacc6ad57d74ee3451c97508da1 AS builder

ENV MMCV_WITH_OPS=1
ENV PYTORCH_ROCM_ARCH=gfx1151
ENV MAX_JOBS=8

RUN python -m pip install --no-cache-dir \
        --index-url https://repo.amd.com/rocm/whl-multi-arch/ \
        "rocm[devel]==7.14.1" \
    && rocm-sdk init

ENV ROCM_DEVEL=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_devel
ENV CPATH="${ROCM_DEVEL}/include"
ENV CPLUS_INCLUDE_PATH="${ROCM_DEVEL}/include"
ENV LIBRARY_PATH="${ROCM_DEVEL}/lib:/opt/venv/lib/python3.12/site-packages/_rocm_sdk_core/lib:/opt/venv/lib/python3.12/site-packages/_rocm_sdk_libraries/lib"
ENV LD_LIBRARY_PATH="${ROCM_DEVEL}/lib:/opt/venv/lib/python3.12/site-packages/_rocm_sdk_core/lib:/opt/venv/lib/python3.12/site-packages/_rocm_sdk_libraries/lib"
ENV LDFLAGS="-L${ROCM_DEVEL}/lib"

WORKDIR /build

COPY patches/mmcv-cxx20.patch /build/mmcv-cxx20.patch

RUN python -m pip install --no-cache-dir \
        opencv-python-headless==5.0.0.93

RUN git init /build/mmengine \
    && cd /build/mmengine \
    && git remote add origin https://github.com/open-mmlab/mmengine.git \
    && git fetch --depth 1 origin 390ba2fbb272816adfd2883642326d0fd0ca6049 \
    && git checkout --detach FETCH_HEAD \
    && python -m pip wheel \
        --no-build-isolation \
        --no-deps \
        --wheel-dir /wheels \
        .

RUN python -m pip install --no-cache-dir \
        /wheels/mmengine-0.10.7-py3-none-any.whl

RUN git init /build/mmcv \
    && cd /build/mmcv \
    && git remote add origin https://github.com/open-mmlab/mmcv.git \
    && git fetch --depth 1 origin 57c4e25e06e2d4f8a9357c84bcd24089a284dc88 \
    && git checkout --detach FETCH_HEAD \
    && git apply /build/mmcv-cxx20.patch \
    && python -m pip wheel \
        --no-build-isolation \
        --no-deps \
        --wheel-dir /wheels \
        .

RUN git init /build/mmdetection \
    && cd /build/mmdetection \
    && git remote add origin https://github.com/open-mmlab/mmdetection.git \
    && git fetch --depth 1 origin 44ebd17b145c2372c4b700bfb9cb20dbd28ab64a \
    && git checkout --detach FETCH_HEAD \
    && python -m pip wheel \
        --no-build-isolation \
        --no-deps \
        --wheel-dir /wheels \
        .


FROM rocm/pytorch:rocm7.14.1_ubuntu24.04_py3.12_pytorch_release_2.12.0@sha256:cc9b00f90b85c97b015b040fa55c8d1b404b7cacc6ad57d74ee3451c97508da1 AS runtime

ENV LD_LIBRARY_PATH="/opt/venv/lib/python3.12/site-packages/_rocm_sdk_core/lib:/opt/venv/lib/python3.12/site-packages/_rocm_sdk_libraries/lib"

WORKDIR /opt/stack

COPY requirements.txt /opt/stack/requirements.txt
COPY scripts/smoke_test.py /opt/stack/scripts/smoke_test.py

RUN python -m pip install --no-cache-dir \
        -r /opt/stack/requirements.txt

COPY --from=builder /wheels /tmp/wheels

RUN python -m pip install --no-cache-dir \
        /tmp/wheels/mmengine-0.10.7-py3-none-any.whl \
        /tmp/wheels/mmcv-2.1.0-cp312-cp312-linux_x86_64.whl \
        /tmp/wheels/mmdet-3.3.0-py3-none-any.whl \
    && rm -rf /tmp/wheels

COPY --from=builder /build/mmdetection/configs /opt/mmdetection/configs
COPY --from=builder /build/mmdetection/tools /opt/mmdetection/tools

WORKDIR /opt/mmdetection

CMD ["python", "/opt/stack/scripts/smoke_test.py"]
