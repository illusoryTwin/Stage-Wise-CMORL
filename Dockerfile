
# ===========================
# Ubuntu 20.04 + CUDA 11.6 base
# ===========================
FROM ubuntu:20.04

# Set working directory
WORKDIR /workspace
ENV DEBIAN_FRONTEND=noninteractive

# Install basic tools
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    curl \
    gnupg \
    ca-certificates \
    sudo \
    git \
    lsb-release \
    libgl1-mesa-glx \
    libglib2.0-0 \
    vim \
    && rm -rf /var/lib/apt/lists/*

# # =========================
# # Install Vulkan SDK
# # =========================
# ENV VULKAN_VERSION=1.3.261.1
# ENV VULKAN_SDK=/opt/vulkan-sdk

#RUN wget https://sdk.lunarg.com/sdk/download/$VULKAN_VERSION/linux/vulkan-sdk-$VULKAN_VERSION.tar.gz -O /tmp/vulkan-sdk.tar.gz && \
#    mkdir -p /opt && \
#    tar -xzf /tmp/vulkan-sdk.tar.gz -C /opt/ && \
#    mv /opt/$VULKAN_VERSION $VULKAN_SDK && \
#    rm /tmp/vulkan-sdk.tar.gz

#ENV PATH=$VULKAN_SDK/bin:$PATH
#ENV LD_LIBRARY_PATH=$VULKAN_SDK/lib:$LD_LIBRARY_PATH


# Add NVIDIA CUDA repository (network repo version)
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin \
    && mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600 \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub \
    && echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /" > /etc/apt/sources.list.d/cuda.list \
    && apt-get update

# Install CUDA runtime 11.6 and cuDNN
RUN apt-get install -y cuda-runtime-11-6 libcudnn8 && rm -rf /var/lib/apt/lists/*

# Install Mambaforge (conda)
RUN wget https://github.com/conda-forge/miniforge/releases/download/23.11.0-0/Mambaforge-Linux-x86_64.sh -O mambaforge.sh \
    && bash mambaforge.sh -b -p /opt/conda \
    && rm mambaforge.sh

# Set environment variables
ENV PATH=/opt/conda/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/conda/envs/py38/lib:$LD_LIBRARY_PATH

# Switch to bash shell
SHELL ["/bin/bash", "-c"]

# Create Python 3.8 environment
RUN conda create -y -n py38 python=3.8 && conda clean -afy

# Activate environment and install PyTorch 1.13 (CUDA 11.6)
RUN source /opt/conda/bin/activate py38 && \
    pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116

# Install other Python dependencies
RUN source /opt/conda/bin/activate py38 && \
    pip install numpy==1.21.5 ruamel.yaml requests pandas scipy wandb

# Clone Stage-Wise CMORL repo
RUN git clone https://github.com/rllab-snu/Stage-Wise-CMORL.git /workspace/Stage-Wise-CMORL

# Set working directory to repo
WORKDIR /workspace/Stage-Wise-CMORL

# Copy Isaac Gym tar.gz to container and extract it
COPY IsaacGym_Preview_4_Package.tar.gz /workspace/
RUN cd /workspace && tar -xzf IsaacGym_Preview_4_Package.tar.gz

# Install Isaac Gym
RUN source /opt/conda/bin/activate py38 && \
    cd /workspace/isaacgym/python && \
    pip install -e . && \
    pip show isaacgym

# Install Isaac Gym Envs
RUN git clone https://github.com/NVIDIA-Omniverse/IsaacGymEnvs.git /workspace/isaacgymenvs && \
    source /opt/conda/bin/activate py38 && \
    pip install -e /workspace/isaacgymenvs


# Copy current code into container (overwrite cloned repo with local changes)
COPY . /workspace/Stage-Wise-CMORL

RUN source /opt/conda/bin/activate py38

# # Copy the wrapper script
# COPY run_task.sh /workspace/run_task.sh
# RUN chmod +x /workspace/run_task.sh


# # Copy current code into container
# COPY . /workspace

# # Copy the wrapper script
# COPY run_task.sh /workspace/run_task.sh
# RUN chmod +x /workspace/run_task.sh

# Optional: Set as entrypoint
# ENTRYPOINT ["/workspace/run_task.sh"]
