# ############ Path for hessian
export PROJECT_DIR=/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge
export TORCHINDUCTOR_CACHE_DIR=$PROJECT_DIR/VLA/cache
export HF_HOME=$PROJECT_DIR/VLA/hf_cache
export TMPDIR=$PROJECT_DIR/VLA/cache
export PYTHONPATH=$PROJECT_DIR/VLA/duc/lerobot:$PYTHONPATH
CHECKPOINT_DIR=$PROJECT_DIR/VLA/LIBERO/pi05_base
DATA_DIR=$PROJECT_DIR/VLA/LIBERO/merged_libero_scale_100_mask_depth_noops_lerobot_v30
EXP_NAME=pi05_libero_100%_merged
SAVE_CHECKPOINT_DIR=$PROJECT_DIR/VLA/duc/lerobot
POLICY_CONFIG_PATH=configs/policy_config/default_decay300k.json
OTHER_CONFIG_PATH=configs/libero_config/default.json


# accelerate launch --multi-gpu --num_processes=2 src/lerobot/scripts/lerobot_train.py \
#     --dataset.repo_id=None \
#     --dataset.root=$DATA_DIR \
#     --policy.type=pi05 \
#     --output_dir=outputs/train/$(date +%Y-%m-%d)/$(date +%H-%M-%S)_$EXP_NAME \
#     --job_name=$EXP_NAME \
#     --policy.repo_id=None \
#     --policy.pretrained_path=$CHECKPOINT_DIR \
#     --policy.gradient_checkpointing=true \
#     --wandb.enable=true \
#     --policy.dtype=bfloat16 \
#     --steps=3000 \
#     --policy.freeze_vision_encoder=false \
#     --policy.train_expert_only=false \
#     --batch_size=64 \
#     --policy.normalization_mapping='{"ACTION": "MEAN_STD", "STATE": "MEAN_STD", "VISUAL": "IDENTITY"}' \


CUDA_VISIBLE_DEVICES=0  accelerate launch --num_processes=1 --main_process_port 29521 src/lerobot/scripts/lerobot_train.py \
    --dataset.repo_id=None \
    --dataset.root=$DATA_DIR \
    --policy.type=pi05 \
    --output_dir=outputs/train/$(date +%Y-%m-%d)/$(date +%H-%M-%S)_$EXP_NAME \
    --job_name=$EXP_NAME \
    --policy.repo_id=None \
    --policy.pretrained_path=$CHECKPOINT_DIR \
    --policy.dtype=bfloat16 \
    --steps=60000 \
    --log_freq=10 \
    --policy.scheduler_decay_steps=3000 \
    --policy.freeze_vision_encoder=true \
    --policy.train_expert_only=false \
    --policy.normalization_mapping='{"ACTION": "MEAN_STD", "STATE": "MEAN_STD", "VISUAL": "IDENTITY"}' \
    --batch_size=3 \
    --policy.device=cuda


# CUDA_VISIBLE_DEVICES=0,1,2,3  accelerate launch --num_processes=4 --main_process_port 29520 --mixed_precision=bf16 src/lerobot/scripts/lerobot_train.py \
#     --dataset.repo_id=None \
#     --dataset.root=$DATA_DIR \
#     --policy.type=pi05 \
#     --output_dir=outputs/train/$(date +%Y-%m-%d)/$(date +%H-%M-%S)_$EXP_NAME \
#     --job_name=$EXP_NAME \
#     --policy.repo_id=None \
#     --policy.pretrained_path=$CHECKPOINT_DIR \
#     --policy.compile_model=true \
#     --policy.gradient_checkpointing=true \
#     --wandb.enable=true \
#     --policy.dtype=bfloat16 \
#     --steps=60000 \
#     --policy.freeze_vision_encoder=true \
#     --policy.train_expert_only=false \
#     --policy.normalization_mapping='{"ACTION": "MEAN_STD", "STATE": "MEAN_STD", "VISUAL": "IDENTITY"}' \
#     --batch_size=32 \
#     --policy.scheduler_warmup_steps=10_000 \
#     --policy.optimizer_lr=5e-5 \
#     --policy.scheduler_decay_steps=1_000_000 \
#     --policy.scheduler_decay_lr=5e-5 \
#     --policy.device=cuda
