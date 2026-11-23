#!/usr/bin/env bash
# MiniMax-H3 text-to-audio-video DMD2 distillation on a tiny random H3: a GPU smoke of the distribution_matching
# trainer, not a quality recipe.
#
# MODEL_PATH: the output directory of tests/special_e2e/build_minimax_h3_tiny_random.py (holds FL2VA/ and
#   transformer/).
# CACHE_DIR: a prompt cache in the PDMD layout (rank-*.jsonl manifests and safetensors shards; see
#   verl_omni/utils/dataset/prompt_cache_dataset.py) whose embeddings match the tiny DiT's text width.
set -euo pipefail

MODEL_PATH=${MODEL_PATH:?set MODEL_PATH to the tiny MiniMax-H3 build}
CACHE_DIR=${CACHE_DIR:?set CACHE_DIR to the prompt cache}
NUM_GPUS=${NUM_GPUS:-1}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-20}
# video_rows, audio_rows, latent_T, latent_H, latent_W, audio_T: 2 x 4 x 4 video latents in (1, 2, 2) patches
LATENT_META=${LATENT_META:-[8,10,2,4,4,5]}

if (( NUM_GPUS <= 0 )); then
    echo "NUM_GPUS must be positive, got $NUM_GPUS" >&2
    exit 1
fi
if [[ ! -d "$MODEL_PATH/FL2VA" || ! -d "$MODEL_PATH/transformer" ]]; then
    echo "MODEL_PATH must hold FL2VA/ and transformer/ (got: '$MODEL_PATH')" >&2
    exit 1
fi

h3_lora_targets="['to_q','to_k','to_v','to_out.0','ff.net.0.proj','ff.net.2']"

python3 -m verl_omni.trainer.main_diffusion_v1 \
    trainer.use_v1=true \
    trainer.v1.trainer_mode=distribution_matching \
    "data.train_files=$CACHE_DIR" \
    "data.val_files=$CACHE_DIR" \
    data.train_batch_size="$NUM_GPUS" \
    data.seed=42 \
    data.dataloader_num_workers=0 \
    "data.latent_meta_buckets=[$LATENT_META]" \
    "actor_rollout_ref.model.path=$MODEL_PATH/FL2VA" \
    "actor_rollout_ref.model.config_path=$MODEL_PATH/transformer" \
    +actor_rollout_ref.model.architecture=MiniMaxH3Pipeline \
    actor_rollout_ref.model.algorithm=dmd2 \
    actor_rollout_ref.model.model_type=diffusion_dmd_model \
    actor_rollout_ref.model.attn_backend=native \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.lora_rank=8 \
    actor_rollout_ref.model.lora_alpha=8 \
    actor_rollout_ref.model.policy_state_adapters='["default","fake_score"]' \
    actor_rollout_ref.model.target_modules="$h3_lora_targets" \
    actor_rollout_ref.model.fsdp_layer_prefixes="['transformer_blocks.','token_refiner.refiner_blocks.']" \
    '+actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[MiniMaxH3TransformerBlock,MiniMaxH3TokenRefinerBlock]' \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.actor.optim.clip_grad=1.0 \
    actor_rollout_ref.actor.optim.lr_scheduler_type=constant \
    actor_rollout_ref.actor.ppo_mini_batch_size="$NUM_GPUS" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.diffusion_loss.loss_mode=dmd2 \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.rollout_attn_backend=TORCH_SDPA \
    dmd.num_inference_steps=4 \
    dmd.fake_update_ratio=5 \
    dmd.student_micro_batch_size_per_gpu=1 \
    dmd.fake_score_micro_batch_size_per_gpu=1 \
    dmd.fake_score_optim.lr=1e-5 \
    +dmd.modality_shifts.video=12.0 \
    +dmd.modality_shifts.audio=3.0 \
    +dmd.modality_weights.video=1.0 \
    +dmd.modality_weights.audio=1.0 \
    dmd.critic_time_lead_modality=video \
    dmd.score_discrete_steps=0 \
    dmd.teacher_guidance_scale=1.0 \
    dmd.cfg_norm=none \
    dmd.surrogate_scale=1.0 \
    dmd.loss_clamp=5.0 \
    dmd.projected=true \
    trainer.logger='["console"]' \
    trainer.project_name=dmd \
    trainer.experiment_name=minimax_h3_t2va_dmd_tiny \
    trainer.n_gpus_per_node="$NUM_GPUS" \
    trainer.nnodes=1 \
    trainer.total_training_steps="$TOTAL_TRAINING_STEPS" "$@"
