set -x
export WANDB_API_KEY="your wandb key here"
export HF_TOKEN="your HF token here"
export TORCH_NCCL_AVOID_RECORD_STREAMS="1"
export VLLM_USE_V1=0
##########################################################################################
PROJECT_NAME=think_genrm_grpo_naive-multiclass
BASE_MODEL=ilgee/hs2-naive-multiclass-max-ep5-lr5e-6
EPOCH=2
KL_COEF=1e-4
LR=2e-6
ROLLOUT_SIZE=512
MODEL_NAME=$(basename "$BASE_MODEL")
RUN_NAME=${MODEL_NAME}-grpo-ep${EPOCH}-lr${LR}-kl${KL_COEF}-rollout${ROLLOUT_SIZE}-v0
##########################################################################################

### Create GRPO Dataset (modify here if naive)
python3 VERL/examples/data_preprocess/hs2-naive-reasoning-multiclass.py --local_dir ~/data/hs2-naive-reasoning-multiclass
### Get Ready for GRPO Run
cd VERL
### Run GRPO (modify data.train_files, data.val_files, custom_reward_function.path)
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$RUN_NAME \
    trainer.default_local_dir=/workspace/ckpt/$PROJECT_NAME/$RUN_NAME \
    data.train_files=$HOME/data/hs2-naive-reasoning-multiclass/train.parquet \
    data.val_files=$HOME/data/hs2-naive-reasoning-multiclass/test.parquet \
    data.train_batch_size=$ROLLOUT_SIZE \
    data.max_prompt_length=4096 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    custom_reward_function.path=verl/utils/reward_score/naive-reasoning-multiclass.py \
    custom_reward_function.name=compute_score_batched \
    reward_model.reward_manager=batch \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$((ROLLOUT_SIZE / 4)) \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=$KL_COEF \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=10 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=10 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=10000 \
    trainer.test_freq=4 \
    trainer.total_epochs=$EPOCH $@

### Check Last Iteration Number
LAST_ITER="$(cat /workspace/ckpt/$PROJECT_NAME/$RUN_NAME/latest_checkpointed_iteration.txt)"

### Prepare HuggingFace Model
python3 VERL/scripts/model_merger.py \
                        --backend fsdp \
                        --hf_model_path $BASE_MODEL \
                        --local_dir /workspace/ckpt/$PROJECT_NAME/$RUN_NAME/global_step_${LAST_ITER}/actor \
                        --target_dir /workspace/ckpt/$PROJECT_NAME/$RUN_NAME/global_step_${LAST_ITER}/actor/huggingface