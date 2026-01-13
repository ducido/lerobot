"""
Minimal example script for converting a dataset to LeRobot format.

We use the Libero dataset (stored in RLDS) for this example, but it can be easily
modified for any other data you have saved in a custom format.

Usage:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data

If you want to push your dataset to the Hugging Face Hub, you can use the following command:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data --push_to_hub

Note: to run the script, you need to install tensorflow_datasets:
`uv pip install tensorflow tensorflow_datasets`

You can download the raw Libero datasets from https://huggingface.co/datasets/openvla/modified_libero_rlds
The resulting dataset will get saved to the $HF_LEROBOT_HOME directory.
Running this conversion script will take approximately 30 minutes.
"""

import shutil

from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import tyro


import numpy as np
from pathlib import Path
import re

from lerobot.envs.factory import make_env
from lerobot.configs.eval import EvalPipelineConfig


import abc
from dataclasses import dataclass, field
from typing import Any

import draccus

from lerobot.configs.types import FeatureType, PolicyFeature

from lerobot.utils.constants import (
    ACTION,
    LIBERO_KEY_EEF_MAT,
    LIBERO_KEY_EEF_POS,
    LIBERO_KEY_EEF_QUAT,
    LIBERO_KEY_GRIPPER_QPOS,
    LIBERO_KEY_GRIPPER_QVEL,
    LIBERO_KEY_JOINTS_POS,
    LIBERO_KEY_JOINTS_VEL,
    LIBERO_KEY_PIXELS_AGENTVIEW,
    LIBERO_KEY_PIXELS_EYE_IN_HAND,
    OBS_ENV_STATE,
    OBS_IMAGE,
    OBS_IMAGES,
    OBS_STATE,
)

import os
os.environ["MUJOCO_GL"] = "egl"
import matplotlib.pyplot as plt
import numpy as np

import metaworld.policies as policies
import gym


FEATURES = {
            "observation.state": {
                "dtype": "float32",
                "shape": (4,),
                "names": None,
                "fps": 80,
            },
            "action": {
                "dtype": "float32",
                "shape": (4,),
                "names": ["x", "y", "z", "gripper"],
                "fps": 80,
            },
            "next.reward": {
                "dtype": "float32",
                "shape": (1,),
                "names": None,
                "fps": 80,
            },
            "next.success": {
                "dtype": "bool",
                "shape": (1,),
                "names": None,
                "fps": 80,
            },
            "observation.environment_state": {
                "dtype": "float32",
                "shape": (39,),
                "names": ["keypoints"],
                "fps": 80,
            },
            "observation.image": {
                "dtype": "image",
                "shape": (3, 480, 480),
                "names": ["channels", "height", "width"],
                "fps": 80,
            },
            "observation.image_gripper_mask": {
                "dtype": "image",
                "shape": (3, 480, 480),
                "names": ["channels", "height", "width"],
                "fps": 80,
            },
            "observation.image_instance_mask": {
                "dtype": "image",
                "shape": (3, 480, 480),
                "names": ["channels", "height", "width"],
                "fps": 80,
            },
            "ooi_mapping": {
                "dtype": "string",
                "shape": (1,),
                "names": {"ooi_mapping": ['ooi_mapping']},
            },
        }


TASK_POLICY_MAPPING = {
    "assembly-v3": "SawyerAssemblyV3Policy", "basketball-v3": "SawyerBasketballV3Policy",
    "bin-picking-v3": "SawyerBinPickingV3Policy", "box-close-v3": "SawyerBoxCloseV3Policy",
    "button-press-topdown-v3": "SawyerButtonPressTopdownV3Policy",
    "button-press-topdown-wall-v3": "SawyerButtonPressTopdownWallV3Policy",
    "button-press-v3": "SawyerButtonPressV3Policy", "button-press-wall-v3": "SawyerButtonPressWallV3Policy",
    "coffee-button-v3": "SawyerCoffeeButtonV3Policy", "coffee-pull-v3": "SawyerCoffeePullV3Policy",
    "coffee-push-v3": "SawyerCoffeePushV3Policy", "dial-turn-v3": "SawyerDialTurnV3Policy",
    "disassemble-v3": "SawyerDisassembleV3Policy", "door-close-v3": "SawyerDoorCloseV3Policy",
    "door-lock-v3": "SawyerDoorLockV3Policy", "door-open-v3": "SawyerDoorOpenV3Policy",
    "door-unlock-v3": "SawyerDoorUnlockV3Policy", "drawer-close-v3": "SawyerDrawerCloseV3Policy",
    "drawer-open-v3": "SawyerDrawerOpenV3Policy", "faucet-close-v3": "SawyerFaucetCloseV3Policy",
    "faucet-open-v3": "SawyerFaucetOpenV3Policy", "hammer-v3": "SawyerHammerV3Policy",
    "hand-insert-v3": "SawyerHandInsertV3Policy", "handle-press-side-v3": "SawyerHandlePressSideV3Policy",
    "handle-press-v3": "SawyerHandlePressV3Policy", "handle-pull-side-v3": "SawyerHandlePullSideV3Policy",
    "handle-pull-v3": "SawyerHandlePullV3Policy", "lever-pull-v3": "SawyerLeverPullV3Policy",
    "peg-insert-side-v3": "SawyerPegInsertionSideV3Policy", "peg-unplug-side-v3": "SawyerPegUnplugSideV3Policy",
    "pick-out-of-hole-v3": "SawyerPickOutOfHoleV3Policy", "pick-place-v3": "SawyerPickPlaceV3Policy",
    "pick-place-wall-v3": "SawyerPickPlaceWallV3Policy",
    "plate-slide-back-side-v3": "SawyerPlateSlideBackSideV3Policy",
    "plate-slide-back-v3": "SawyerPlateSlideBackV3Policy",
    "plate-slide-side-v3": "SawyerPlateSlideSideV3Policy", "plate-slide-v3": "SawyerPlateSlideV3Policy",
    "push-back-v3": "SawyerPushBackV3Policy", "push-v3": "SawyerPushV3Policy",
    "push-wall-v3": "SawyerPushWallV3Policy", "reach-v3": "SawyerReachV3Policy",
    "reach-wall-v3": "SawyerReachWallV3Policy", "shelf-place-v3": "SawyerShelfPlaceV3Policy",
    "soccer-v3": "SawyerSoccerV3Policy", "stick-pull-v3": "SawyerStickPullV3Policy",
    "stick-push-v3": "SawyerStickPushV3Policy", "sweep-into-v3": "SawyerSweepIntoV3Policy",
    "sweep-v3": "SawyerSweepV3Policy", "window-open-v3": "SawyerWindowOpenV3Policy",
    "window-close-v3": "SawyerWindowCloseV3Policy"}


@dataclass
class EnvConfig(draccus.ChoiceRegistry, abc.ABC):
    task: str | None = None
    fps: int = 30
    features: dict[str, PolicyFeature] = field(default_factory=dict)
    features_map: dict[str, str] = field(default_factory=dict)
    max_parallel_tasks: int = 1
    disable_env_checker: bool = True

    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)

    @property
    def package_name(self) -> str:
        """Package name to import if environment not found in gym registry"""
        return f"gym_{self.type}"

    @property
    def gym_id(self) -> str:
        """ID string used in gym.make() to instantiate the environment"""
        return f"{self.package_name}/{self.task}"

    @property
    @abc.abstractmethod
    def gym_kwargs(self) -> dict:
        raise NotImplementedError()


@EnvConfig.register_subclass("metaworld")
@dataclass
class MetaworldEnv(EnvConfig):
    task: str = "metaworld-push-v2"  # add all tasks
    fps: int = 80
    episode_length: int = 400
    obs_type: str = "pixels_agent_pos"
    render_mode: str = "segmentation"
    multitask_eval: bool = True
    features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(4,)),
        }
    )
    features_map: dict[str, str] = field(
        default_factory=lambda: {
            "action": ACTION,
            "agent_pos": OBS_STATE,
            "top": f"{OBS_IMAGE}",
            "pixels/top": f"{OBS_IMAGE}",
        }
    )
    render_segmentation: bool = True

    def __post_init__(self):
        if self.obs_type == "pixels":
            self.features["top"] = PolicyFeature(type=FeatureType.VISUAL, shape=(480, 480, 3))

        elif self.obs_type == "pixels_agent_pos":
            self.features["agent_pos"] = PolicyFeature(type=FeatureType.STATE, shape=(4,))
            self.features["pixels/top"] = PolicyFeature(type=FeatureType.VISUAL, shape=(480, 480, 3))

        else:
            raise ValueError(f"Unsupported obs_type: {self.obs_type}")

    @property
    def gym_kwargs(self) -> dict:
        return {
            "obs_type": self.obs_type,
            "render_mode": self.render_mode,
            "render_segmentation": self.render_segmentation,
        }


from scipy import ndimage as ndi
import numpy as np

def clean_and_smooth_mask(mask,
                          sigma=3.6,
                          thresh=0.45,
                          keep_largest=True):
    mask = mask.astype(np.float32)

    # 1. Smooth
    blurred = ndi.gaussian_filter(mask, sigma=sigma)
    binary = blurred > thresh

    # 2. Fill holes
    filled = ndi.binary_fill_holes(binary)

    # # 3. Keep largest component
    # if keep_largest:
    #     labeled, num = ndi.label(filled)
    #     if num > 1:
    #         sizes = ndi.sum(filled, labeled, range(1, num + 1))
    #         largest = np.argmax(sizes) + 1
    #         filled = (labeled == largest)

    return filled.astype(np.uint8)


def to_3ch_chw_uint8(mask: np.ndarray, binary: bool = False) -> np.ndarray:
    mask = mask.astype(np.uint8)
    if binary and mask.max() == 1:
        mask *= 255
    return np.repeat(mask[None, :, :], 3, axis=0)

def make_policy(task_name):
    policy_cls_name = TASK_POLICY_MAPPING[task_name]
    policy_cls = getattr(policies, policy_cls_name)
    return policy_cls()




















'''
export HF_LEROBOT_HOME=/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/duc
python convert_mw_to_lerobot.py 
'''

EPISODES_PER_TASK = 1
MAX_STEPS = 200
REPO_NAME = 'ducido/meta-world_MT50'

def main():
    # Clean up any existing dataset in the output directory
    output_path = HF_LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        shutil.rmtree(output_path)

    # Create LeRobot dataset, define features to store
    # OpenPi assumes that proprio is stored in `state` and actions in `action`
    # LeRobot assumes that dtype of image data is `image`
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="panda",
        fps=80,   # fps chung, từng feature vẫn có fps riêng nếu cần
        features=FEATURES,
        image_writer_threads=10,
        image_writer_processes=5,
    )

    for task_name in list(TASK_POLICY_MAPPING.keys())[:1]:
    # for task_name in TASK_POLICY_MAPPING.keys():
        policy = make_policy(task_name)

        mt_env_pixel = MetaworldEnv(
            task=task_name,
            render_mode="rgb_array",
            render_segmentation=True,
            obs_type="pixels_agent_pos"   # giữ như bạn đang dùng
        )

        envs_pixel = make_env(
            mt_env_pixel,
            n_envs=1,
            use_async_envs=False
        )

        # =========================
        # Collect loop
        # =========================
        for task_name, task_envs in envs_pixel.items():
            print(f"\nCollecting data for task: {task_name}")

            # Pixel vector env
            vec_env_pixel = task_envs[0]  # SyncVectorEnv(num_envs=1)

            # State env (single env, NOT vector)

            episodes_collected = 0

            while episodes_collected < EPISODES_PER_TASK:
                # ---- Reset BOTH envs with same seed ----
                seed = np.random.randint(0, 1_000_000)
                obs_pixel, info = vec_env_pixel.reset(seed=seed)
                

                done = False
                step = 0

                while not done and step < MAX_STEPS:
                    # ---- ACTION from EXPERT (STATE env) ----
                    state_obs = obs_pixel['environment_state'][0]
                    action = policy.get_action(state_obs)   # (act_dim,)
                    action_vec = action[None, :]             # (1, act_dim)

                    # ---- Step BOTH envs ----
                    next_pixel_obs, reward, terminated_p, truncated_p, info = vec_env_pixel.step(action_vec)

                    done = bool(terminated_p or truncated_p)

                    ### Clean OOI mapping
                    ooi_mapping = {k: v for k, v in info["ooi_mapping"].items() if isinstance(k, int)}
                    ooi_mapping[99] = np.array(['wall'])

                    ### Delete -1 value in original segmentation
                    original_instance_mask = obs_pixel["segmentation"][0]
                    original_instance_mask[original_instance_mask==-1] = 99

                    #### Extract only gripper mask
                    arm_parts = ['leftpad', 'rightpad', 'hand', 'right_hand', 'right_l6', 'right_l5', 'right_l4', 'right_l3', 'right_l2', 'right_l1', 'head', 'right_l0', 'right_arm_base_link']
                    pixel_to_arm = {k: v for k, v in ooi_mapping.items() if v.item() not in arm_parts} 
                    gripper_mask = np.zeros_like(original_instance_mask)
                    for i in pixel_to_arm:
                        gripper_mask[original_instance_mask==i] = 1
                    gripper_mask[original_instance_mask==99] = 1
                    cleaned_gripper_mask = clean_and_smooth_mask(gripper_mask)
                    # print(np.max(gripper_mask))
                    
                    dataset.add_frame(
                        {
                            "observation.state": obs_pixel["agent_pos"][0].astype('float32'),
                            "action": action.astype('float32'),
                            "next.reward": np.array(reward, dtype=np.float32),
                            "next.success": np.array([done], dtype=np.bool_),
                            "observation.environment_state": obs_pixel["environment_state"][0].astype('float32'),
                            "observation.image": obs_pixel["pixels"][0],
                            "observation.image_instance_mask": to_3ch_chw_uint8(original_instance_mask),
                            "observation.image_gripper_mask": to_3ch_chw_uint8(cleaned_gripper_mask),
                            "ooi_mapping": str(ooi_mapping),
                            "task": task_name
                        },
                    )
                    #### Update counter
                    obs_pixel = next_pixel_obs
                    step += 1
                
                episodes_collected += 1
                dataset.save_episode()


if __name__ == "__main__":
    tyro.cli(main)