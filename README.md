# Stage-Wise CMORL

This is a fork from official GitHub Repository for paper ["Stage-Wise Reward Shaping for Acrobatic Robots: A Constrained Multi-Objective Reinforcement Learning Approach"](https://arxiv.org/abs/2409.15755).

## Requirements

- python==3.8
- torch==1.13.1
- numpy==1.21.5
- isaacgym (https://developer.nvidia.com/isaac-gym)
- IsaacGymEnvs (https://github.com/isaac-sim/IsaacGymEnvs)
- ruamel.yaml
- requests
- pandas
- scipy
- wandb

## Docker Setup

### Prerequisites

1. **Download IsaacGym**:
   - Go to https://developer.nvidia.com/isaac-gym
   - Sign up/login with NVIDIA Developer account
   - Download `IsaacGym_Preview_4_Package.tar.gz`
   - Place the downloaded file in the root directory of this repository

2. **Build Docker image**:
   ```bash
   docker build -t stage-wise-cmorl .
   ```

3. **Run Docker container**:
   ```
   xhost +local:docker
   ```

   ```bash
   docker run --gpus all -it stage-wise-cmorl
   ```

    Or, 
    ```
    docker run -it --rm --gpus all \
        --env="DISPLAY" \
        --env="NVIDIA_DRIVER_CAPABILITIES=all" \
        --env="NVIDIA_VISIBLE_DEVICES=all" \
        -e DISPLAY=$DISPLAY \
        -e XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        --device /dev/dri \
        stage-wise-cmorl
    ```

**Note**: The `IsaacGym_Preview_4_Package.tar.gz` file is NOT included in this repository due to its size (200MB+) and NVIDIA's licensing requirements. You must download it manually before building the Docker image.

## Organization
```
Stage-Wise-CMORL/
    └── algos/
    │     └── common/
    │     └── comoppo/
    │     └── student/
    └── assets/
    │     └── go1/
    │     └── go2/
    │     └── h1/
    └── tasks/
    └── utils/
    └── main_student.py
    └── main_teacher.py
```
- `algos/`: contains the implementation of the proposed algorithm
- `assets/`: contains the assets of the robots
- `tasks/`: contains the implementation of the tasks
- `utils/`: contains the utility functions

## Tasks

- GO1 Robot (Quadruped from Unitree)
    - Back-Flip
    - Side-Flip
    - Side-Roll
    - Two-Hand Walk
- GO2 Robot (Quadruped from Unitree)
    - Back-Flip
- H1 Robot (Humanoid from Unitree)
    - Back-Flip
    - Two-Hand Walk

## Training and Evaluation

It is required to train a teacher poicy first, and then train a student policy using the teacher policy.

### Teacher Learning

- training: `python main_teacher.py --task_cfg_path tasks/{task_name}.yaml --algo_cfg_path algos/comoppo/{task_name}.yaml --wandb --seed 1`
- test: `python main_teacher.py --task_cfg_path tasks/{task_name}.yaml --algo_cfg_path algos/comoppo/{task_name}.yaml --test --render --seed 1 --model_num {saved_model_num}`

### Student Learning

- training: `python main_student.py --task_cfg_path tasks/{task_name}.yaml --algo_cfg_path algos/student/{task_name}.yaml --wandb --seed 1`
- test: `python main_student.py --task_cfg_path tasks/{task_name}.yaml --algo_cfg_path algos/student/{task_name}.yaml --test --render --seed 1 --model_num {saved_model_num}`

**For example, launch Go2 training this way:**
```
    # TEACHER

    python main_teacher.py --task_cfg_path tasks/go2_backflip.yaml --algo_cfg_path algos/comoppo/go2_backflip.yaml --wandb --seed 1
    python main_teacher.py --task_cfg_path tasks/go2_backflip.yaml --algo_cfg_path algos/comoppo/go2_backflip.yaml --test --render --seed 1 --model_num 90000000

    # STUDENT 

    python main_student.py --task_cfg_path tasks/go2_backflip.yaml --algo_cfg_path algos/student/go2_backflip.yaml --wandb --seed 1
    python main_student.py --task_cfg_path tasks/go2_backflip.yaml --algo_cfg_path algos/student/go2_backflip.yaml --test --render --seed 1 --model_num 90000000
```

## Deploy

The `utils/export_policy.py` script loads a trained reinforcement learning policy checkpoint, reconstructs the actor neural network, and exports it as a TorchScript model for deployment without requiring IsaacGym.
