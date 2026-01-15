

CHECKPOINT_DIR=/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/duc/lerobot/outputs/train/2026-01-13/11-52-49_pi05_libero_100%_baseline/checkpoints/020000/pretrained_model

lerobot-eval \
  --env.type=libero \
  --env.task=libero_spatial \
  --eval.batch_size=2 \
  --eval.n_episodes=10 \
  --policy.path=$CHECKPOINT_DIR \
  --policy.n_action_steps=10 \
  --output_dir=./libero_eval_logs/ \
  --env.max_parallel_tasks=1