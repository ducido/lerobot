import torch

IMAGENET_STATS = {
    "mean": [[[0.485]], [[0.456]], [[0.406]]],  # (c,1,1)
    "std": [[[0.229]], [[0.224]], [[0.225]]],  # (c,1,1)
}

from lerobot.configs.eval import EvalPipelineConfig
from lerobot.configs.default import EvalConfig
from lerobot.envs.configs import LiberoEnv
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.factory import make_policy, make_pre_post_processors


class Pi0TorchInference:
    def __init__(self, model_dir: str, device):
        """
        Wrap an existing, loaded PI0Policy for single‐step inference.

        Args:
            policy: an instance of PI0Policy (or any PreTrainedPolicy) with weights already loaded
            device: e.g. "cpu" or "cuda"
        """
        cfg = EvalPipelineConfig(
            env=LiberoEnv(),
            eval=EvalConfig(
                batch_size=1,
                n_episodes=1
            ),
            policy=PI05Config(n_action_steps=10),
            output_dir='./libero_eval_logs',
            job_name='test'
        )
        self.policy = make_policy(
            cfg=cfg.policy,
            env_cfg=cfg.env,
            rename_map=cfg.rename_map,
        )

        self.policy.eval()

        preprocessor_overrides = {
            "device_processor": {"device": str(self.policy.config.device)},
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        }

        self.preprocessor, _ = make_pre_post_processors(
            policy_cfg=cfg.policy,
            pretrained_path=cfg.policy.pretrained_path,
            preprocessor_overrides=preprocessor_overrides,
        )
        self.device = torch.device(device)
        # self.policy = PI05Policy.from_pretrained(model_dir, device=self.device)

    def libero_step(self, data: dict) -> torch.Tensor:
        """
        Run one policy step. 

        Args:
            data = {
                "observation.images.image": ,
                "observation.images.wrist_image":,
                "observation.state":
                "task":,
            }

        Returns:
            action: np.ndarray of shape (action_dim,)
        """

        batch: dict[str, torch.Tensor] = {}

        # state
        state = torch.as_tensor(data["observation.state"], dtype=torch.float32, device=self.device)
        batch["observation.state"] = state.unsqueeze(0)

        # normalize images using ImageNet stats
        for key in ("observation.images.image", "observation.images.wrist_image"):
            img = torch.as_tensor(data[key], dtype=torch.float32, device=self.device)
            if img.ndim == 3 and img.shape[2] in (1, 3, 4):  # HWC → CHW
                img = img.permute(2, 0, 1)
            img = img[:3, :, :]  # keep only RGB if 4-channel (e.g. RGBA)
            img = img / 255.0    # scale to [0,1]
            # img = normalize_imagenet(img)
            batch[key] = img.unsqueeze(0)

        batch["task"] = [data["task"]]

        batch = self.preprocessor(batch)

        with torch.no_grad():
            action = self.policy.predict_action_chunk(batch)
            action = action.squeeze().cpu().numpy()
        return action

    def calvin_step(self, data: dict) -> torch.Tensor:
        """
        Run one policy step. 

        Args:
            data = {
                "observation.images.image": ,
                "observation.images.wrist_image":,
                "observation.state":
                "task":,
            }

        Returns:
            action: np.ndarray of shape (action_dim,)
        """

        batch: dict[str, torch.Tensor] = {}

        # state
        state = torch.as_tensor(data["state"], dtype=torch.float32, device=self.device)
        batch["state"] = state.unsqueeze(0)

        # normalize images using ImageNet stats
        for key in ('image', 'wrist_image'):
            img = torch.as_tensor(data[key], dtype=torch.float32, device=self.device)
            if img.ndim == 3 and img.shape[2] in (1, 3, 4):  # HWC → CHW
                img = img.permute(2, 0, 1)
            img = img[:3, :, :]  # keep only RGB if 4-channel (e.g. RGBA)
            img = img / 255.0   # scale to [0,1]
            batch[key] = img.unsqueeze(0)

        batch["task"] = [data["task"]]

        with torch.no_grad():
            action = self.policy.select_action_chunk(batch)
            action = action.squeeze().cpu().numpy()
        return action



import numpy as np
def normalize_gripper_action(action, binarize=True):
    """
    Changes gripper action (last dimension of action vector) from [0,1] to [-1,+1].
    Necessary for some environments (not Bridge) because the dataset wrapper standardizes gripper actions to [0,1].
    Note that unlike the other action dimensions, the gripper action is not normalized to [-1,+1] by default by
    the dataset wrapper.

    Normalization formula: y = 2 * (x - orig_low) / (orig_high - orig_low) - 1
    """
    # Just normalize the last action to [-1,+1].
    orig_low, orig_high = 0.0, 1.0
    action[..., -1] = 2 * (action[..., -1] - orig_low) / (orig_high - orig_low) - 1

    if binarize:
        # Binarize to -1 or +1.
        action[..., -1] = np.sign(action[..., -1])

    return action

def invert_gripper_action(action):
    """
    Flips the sign of the gripper action (last dimension of action vector).
    This is necessary for some environments where -1 = open, +1 = close, since
    the RLDS dataloader aligns gripper actions such that 0 = close, 1 = open.
    """
    action[..., -1] = action[..., -1] * -1.0
    return action