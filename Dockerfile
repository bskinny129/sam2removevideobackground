#syntax=docker/dockerfile:1.4
FROM r8.im/cog-base:cuda11.7-python3.10
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked apt-get update -qq && apt-get install -qqy libgl1-mesa-glx nvidia-cuda-toolkit && rm -rf /var/lib/apt/lists/*
COPY .cog/tmp/build20250610210934.080331/cog-0.15.4-py3-none-any.whl /tmp/cog-0.15.4-py3-none-any.whl
ENV CFLAGS="-O3 -funroll-loops -fno-strict-aliasing -flto -S"
RUN --mount=type=cache,target=/root/.cache/pip pip install --no-cache-dir /tmp/cog-0.15.4-py3-none-any.whl 'pydantic>=1.9,<3'
ENV CFLAGS=
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir torch==2.0.1+cu117 torchvision==0.15.2+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
RUN pip install --no-cache-dir git+https://github.com/facebookresearch/segment-anything-2.git@sam2.1
RUN pip install --no-cache-dir opencv-python==4.7.0.72 numpy==1.24.4 Pillow==9.4.0 matplotlib==3.7.1 pycocotools==2.0.6 scipy==1.10.1 timm==0.6.13
RUN wget -q https://huggingface.co/facebook/sam2.1-hiera-large/resolve/main/sam2.1_hiera_large.pt -O /sam2_hiera_large.pt
RUN wget -q https://huggingface.co/facebook/sam2.1-hiera-large/resolve/main/sam2.1_hiera_l.yaml -O sam2.1_hiera_l.yaml
RUN bash -lc 'SAM2_PKG=$(python3 -c "import sam2,os; print(os.path.dirname(sam2.__file__))") && mv sam2.1_hiera_l.yaml "${SAM2_PKG}/sam2.1_hiera_l.yaml"'
WORKDIR /src
EXPOSE 5000
CMD ["python", "-m", "cog.server.http"]
COPY . /src
