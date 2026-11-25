#!/usr/bin/env bash
# MiniMax-H3 text-to-audio-video Projected DMD on the public H3, with the recipe of the PDMD reference run
# (https://github.com/ZeamoxWang/pdmd, training/configs/pdmd_4nfe_544p.json): 4-step deterministic Euler student
# on the unshifted grid [1, .75, .5, .25, 0], two rank-128 LoRAs (student and fake score) on one frozen base,
# 5 fake-score updates per student update, global batch 16, constant learning rates.
#
# MODEL_PATH: MiniMaxAI/MiniMax-H3 at 42ed227ee7df40d41602854ae760620d6eb651fe, holding FL2VA/ and the Diffusers
#   transformer/ (the same weights as FL2VA/transformer, which PDMD trains, under Diffusers names).
# CACHE_DIR: the PDMD prompt cache (pdmd2026/rcm-vidprom-h3-qwenvl-cache at 5abcdbbe08ae594da1fe07128fac80e1a3c1ae0a,
#   its cache/ directory): 248,221 VidProM prompts with precomputed H3 text features.
# CKPT_DIR: checkpoint directory; the trainer resumes from it when it holds a checkpoint.
#
# One verl-omni step is one update (fake score or student), as one PDMD iteration is. NUM_GPUS x 1 sample x
# 16/NUM_GPUS microbatches (2 at the default 8) give the global batch of 16. The 20 latent-geometry entries are
# PDMD's buckets in its order (12x 544x960, 4x 960x544, one each of 640x832, 832x640, 736x736, 480x1120), 124 frames:
# 37 video latent frames and 207 audio latent steps (audio T). latent_meta is (video rows, audio rows, latent T,
# latent H, latent W, audio T), with video rows = T x (H/2) x (W/2) and audio rows = 2 x audio T.
# Every checkpoint is kept: PDMD keeps one rolling checkpoint plus LoRA milestones, which are not mapped here.
set -euo pipefail

MODEL_PATH=${MODEL_PATH:?set MODEL_PATH to the MiniMax-H3 checkout}
CACHE_DIR=${CACHE_DIR:?set CACHE_DIR to the PDMD prompt cache}
CKPT_DIR=${CKPT_DIR:?set CKPT_DIR to the checkpoint directory}
NUM_GPUS=${NUM_GPUS:-8}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-2500}
SAVE_FREQ=${SAVE_FREQ:-100}
GLOBAL_BATCH=16
SEED=42  # the DMD trainer requires the data seed to equal the engine seed

if (( NUM_GPUS <= 0 || GLOBAL_BATCH % NUM_GPUS )); then
    echo "NUM_GPUS must divide the global batch $GLOBAL_BATCH, got $NUM_GPUS" >&2
    exit 1
fi
if [[ ! -d "$MODEL_PATH/FL2VA" || ! -d "$MODEL_PATH/transformer" ]]; then
    echo "MODEL_PATH must hold FL2VA/ and transformer/ (got: '$MODEL_PATH')" >&2
    exit 1
fi

b544x960="[18870,414,37,34,60,207]"
b960x544="[18870,414,37,60,34,207]"
b640x832="[19240,414,37,40,52,207]"
b832x640="[19240,414,37,52,40,207]"
b736x736="[19573,414,37,46,46,207]"
b480x1120="[19425,414,37,30,70,207]"
buckets=""
for _ in $(seq 12); do buckets+="$b544x960,"; done
for _ in $(seq 4); do buckets+="$b960x544,"; done
buckets+="$b640x832,$b832x640,$b736x736,$b480x1120"

h3_lora_targets="['to_q','to_k','to_v','to_out.0','ff.net.0.proj','ff.net.2']"

python3 -m verl_omni.trainer.main_diffusion_v1 \
    trainer.use_v1=true \
    trainer.v1.trainer_mode=distribution_matching \
    "data.train_files=$CACHE_DIR" \
    data.train_batch_size="$GLOBAL_BATCH" \
    data.seed="$SEED" \
    "data.latent_meta_buckets=[$buckets]" \
    "actor_rollout_ref.model.path=$MODEL_PATH/FL2VA" \
    "actor_rollout_ref.model.config_path=$MODEL_PATH/transformer" \
    +actor_rollout_ref.model.architecture=MiniMaxH3Pipeline \
    actor_rollout_ref.model.algorithm=dmd2 \
    actor_rollout_ref.model.model_type=diffusion_dmd_model \
    actor_rollout_ref.model.attn_backend=native \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.lora_rank=128 \
    actor_rollout_ref.model.lora_alpha=128 \
    actor_rollout_ref.model.lora_init_weights=true \
    actor_rollout_ref.model.policy_state_adapters='["default","fake_score"]' \
    actor_rollout_ref.model.target_modules="$h3_lora_targets" \
    actor_rollout_ref.model.fsdp_layer_prefixes="['transformer_blocks.','token_refiner.refiner_blocks.']" \
    '+actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[MiniMaxH3TransformerBlock,MiniMaxH3TokenRefinerBlock]' \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=5e-5 \
    actor_rollout_ref.actor.optim.betas='[0.0,0.9]' \
    actor_rollout_ref.actor.optim.weight_decay=0.0 \
    actor_rollout_ref.actor.optim.clip_grad=1.0 \
    actor_rollout_ref.actor.optim.lr_scheduler_type=constant \
    actor_rollout_ref.actor.ppo_mini_batch_size="$GLOBAL_BATCH" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.diffusion_loss.loss_mode=dmd2 \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.seed="$SEED" \
    actor_rollout_ref.rollout.rollout_attn_backend=TORCH_SDPA \
    dmd.num_inference_steps=4 \
    dmd.sampler=euler \
    dmd.fake_update_ratio=5 \
    dmd.student_micro_batch_size_per_gpu=1 \
    dmd.fake_score_micro_batch_size_per_gpu=1 \
    dmd.fake_score_optim.lr=1e-5 \
    dmd.fake_score_optim.betas='[0.0,0.9]' \
    dmd.fake_score_optim.weight_decay=0.0 \
    dmd.fake_score_optim.clip_grad=1.0 \
    +dmd.modality_shifts.video=1.0 \
    +dmd.modality_shifts.audio=1.0 \
    +dmd.modality_weights.video=0.8 \
    +dmd.modality_weights.audio=0.8 \
    dmd.critic_time=uniform \
    dmd.critic_time_lead_modality=video \
    dmd.critic_time_lead_threshold=0.95 \
    dmd.critic_time_follower_floor=0.85 \
    dmd.score_query_noise=student_estimate \
    dmd.critic_noise=fresh \
    dmd.score_discrete_steps=0 \
    dmd.teacher_guidance_scale=1.0 \
    dmd.surrogate_scale=1.0 \
    dmd.normalization_epsilon=1e-5 \
    dmd.projected=true \
    dmd.projection_epsilon=1e-8 \
    dmd.loss_clamp=5.0 \
    trainer.logger='["console"]' \
    trainer.project_name=dmd \
    trainer.experiment_name=minimax_h3_t2va_dmd_s1 \
    "trainer.default_local_dir=$CKPT_DIR" \
    trainer.save_freq="$SAVE_FREQ" \
    trainer.n_gpus_per_node="$NUM_GPUS" \
    trainer.nnodes=1 \
    trainer.total_training_steps="$TOTAL_TRAINING_STEPS" "$@"
