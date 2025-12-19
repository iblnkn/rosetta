FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04

# Metadata labels
LABEL maintainer="charlielito"
LABEL description="Development container with CUDA 12.6.3, Python, and PyTorch (RTX 5090 compatible)"

# Or your actual UID, GID on Linux if not the default 1000
ARG USERNAME=ada
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Set environment variables
# Suppress TensorFlow warnings for cleaner output (TF_CPP_MIN_LOG_LEVEL: 0=all, 1=no INFO, 2=no INFO/WARN, 3=no INFO/WARN/ERROR)
ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_VERSION=12.6.3 \
    CUDNN_VERSION=9 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TF_CPP_MIN_LOG_LEVEL=2
    
ENV DEBIAN_FRONTEND=noninteractive

# Python package versions (using ARG for build-time, ENV for runtime)
ARG PYTHON_VERSION=3.10

# Install basic utilities and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    wget \
    git \
    vim \
    nano \
    sudo \
    git-lfs \
    python${PYTHON_VERSION} \
    python3-pip \
    python3-dev \
    && ln -s /usr/bin/python3 /usr/bin/python \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# -----------------------------------------------------------------------------
# Define user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME \
    # ------------------------------
    # [Optional] Add sudo support for non-root user
    && apt-get update && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    # ------------------------------
    # Clean up
    && apt-get autoremove && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# setting up ROS2 stuff
ENV ROS_DISTRO=humble

# Configure locale
RUN apt update && apt install -y locales \
    && locale-gen en_US en_US.UTF-8 \
    && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 \
    &&  apt-get autoremove && apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

ENV LANG=en_US.UTF-8

# Install necessary packages
RUN apt update && apt install -y \
    software-properties-common \
    curl \
    gnupg2 \
    lsb-release \
    && apt-get autoremove && apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

# Add ROS 2 apt repository
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" \
    > /etc/apt/sources.list.d/ros2.list

# Install ROS 2 {ROS_DISTRO} Desktop
RUN apt update && apt install -y \
    ros-${ROS_DISTRO}-desktop \
    ros-${ROS_DISTRO}-image-transport \
    ros-${ROS_DISTRO}-image-transport-plugins \
    ros-${ROS_DISTRO}-sensor-msgs-py \
    ros-${ROS_DISTRO}-rviz2 \
    ros-${ROS_DISTRO}-geographic-msgs \
    ros-${ROS_DISTRO}-rmw-cyclonedds-cpp \
    ros-${ROS_DISTRO}-rosbag2-storage-mcap \
    ros-${ROS_DISTRO}-foxglove-msgs\
    && echo "export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp" >> ~/.bashrc \
    && apt-get autoremove && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

# Install colcon build tools
RUN apt update && apt install -y python3-colcon-common-extensions rsync \
    && pip install rosdep \
    && apt-get autoremove && apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

# ENV PATH=/home/ada/.local/bin:$PATH
# export PATH=/home/ada/.local/bin:$PATH

# Add ROS2 sourcing to user's .bashrc (not root's, since container runs as $USERNAME)
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /home/$USERNAME/.bashrc


# -----------------------------------------------------------------------------
# Install PyTorch
RUN pip install --upgrade pip && \
    # --ignore-installed to avoid error with sympy package
    pip install torch==2.7.1 torchvision==0.22.1 --ignore-installed --index-url https://download.pytorch.org/whl/cu126  && \
    # Clean up
    apt-get autoremove && apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# Install rosboard_client dependencies
RUN pip install \
    autobahn==23.6.2 \
    icmplib~=3.0.3 \
    opencv-python~=4.6.0 \
    psutil==5.8.0 \
    PyQt5~=5.15.6 \
    simplejpeg~=1.6.3 \
    twisted~=23.10.0 \
    "setuptools>=61.0, <70.0.0"

# Without the constraints of torch, lerobot tries to download another torch version -.- no cuda support
COPY constraints.txt constraints.txt
RUN pip install -c constraints.txt lerobot==0.4.2

# For some reason numpy2 is not compatible with ros cv_bridge
RUN pip uninstall numpy
# To be able to colcon build --symlink-install our packages we need setuptools 71.0.0 since with 80.0.0 we had problems
RUN pip install 'numpy>=1.22.4,<2' setuptools==71.0.0

# apt install ros-{ROS_DISTRO}-rclpy-message-converter

# -----------------------------------------------------------------------------
# Set working directory
WORKDIR /workspace

# Switch to user
USER $USERNAME

# Default command
CMD ["/bin/bash"]

