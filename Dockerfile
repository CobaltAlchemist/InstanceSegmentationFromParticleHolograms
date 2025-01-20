FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel AS base

ARG DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install git gcc g++ ffmpeg libsm6 libxext6 -y

# Required because docker build doesn't see the gpu
# Set this to your gpu arch: "Pascal", "Volta", "Turing", "Ampere", etc.
ARG ARCHITECTURE="Ampere"
ENV FORCE_CUDA="1"
ENV TORCH_CUDA_ARCH_LIST="${ARCHITECTURE}"
ENV PATH=/usr/local/cuda/bin:$PATH

RUN python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

RUN git clone https://github.com/IDEA-Research/MaskDINO.git
RUN cd MaskDINO && pip install -r requirements.txt && cd maskdino/modeling/pixel_decoder/ops && sh make.sh
COPY scripts/maskdino_setup.py MaskDINO/setup.py
RUN cd MaskDINO && pip install -e .

FROM base AS development

# Leverage caching for deps
WORKDIR /HoloDino
COPY pyproject.toml .
RUN pip install .[dev]
RUN pip uninstall -y holodino fakeholo

FROM base AS deployment

COPY config config
COPY datasets datasets
COPY fakeholo fakeholo
COPY holodino holodino
COPY pyproject.toml .
RUN pip install .
ENTRYPOINT ["python", "-m"]
