FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/venv/bin:${PATH}

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        libglib2.0-0 \
        libgl1 \
        python3.12 \
        python3.12-venv \
    && rm -rf /var/lib/apt/lists/* \
    && python3.12 -m venv /opt/venv

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        torch==2.10.0 \
        torchvision \
        --index-url https://download.pytorch.org/whl/cu128

ARG SAM3_REF=main
RUN git clone https://github.com/facebookresearch/sam3.git /opt/sam3 \
    && cd /opt/sam3 \
    && git checkout ${SAM3_REF} \
    && pip install --no-cache-dir -e .

WORKDIR /service
COPY object_editing_inference/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir --no-deps simple-lama-inpainting==0.1.2

COPY object_editing_inference/app ./app
COPY object_editing_inference/tests ./tests
COPY src/core/networks/fba_matting ./core/networks/fba_matting

EXPOSE 80
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80", "--workers", "1"]
