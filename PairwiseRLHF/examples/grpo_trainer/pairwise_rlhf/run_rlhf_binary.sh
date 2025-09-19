set -x
export WANDB_API_KEY="your wandb key here"
export HF_TOKEN="your HF token here"
export TORCH_NCCL_AVOID_RECORD_STREAMS="1"
export VLLM_USE_V1=0
##########################################################################################
PROJECT_NAME=verl_rlhf_grpo
BASE_MODEL=meta-llama/Llama-3.1-8B-Instruct
EPOCH=2
LR=2e-6
KL_COEF=1e-4
ROLLOUT_SIZE=512
FRAC=0.03
MODEL_NAME=$(basename "$BASE_MODEL")
RUN_NAME=${MODEL_NAME}-grpo-ep${EPOCH}-lr${LR}-kl${KL_COEF}-rollout${ROLLOUT_SIZE}-${FRAC}-binary
##########################################################################################

### Create GRPO Dataset
python3 PairwiseRLHF/examples/data_preprocess/sampled_hh_rlhf.py --local_dir ~/data/sampled_hh_rlhf --frac $FRAC

### Get Ready for GRPO Run
cd PairwiseRLHF

### Run GRPO (modify data.train_files, data.val_files, custom_reward_function.path)
python3 -m verl.trainer.main_ppo \
    trainer.val_before_train=False \
    algorithm.adv_estimator=grpo \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$RUN_NAME \
    trainer.default_local_dir=/workspace/ckpt/$PROJECT_NAME/$RUN_NAME \
    data.train_files=$HOME/data/sampled_hh_rlhf/train.parquet \
    data.val_files=$HOME/data/sampled_hh_rlhf/test.parquet \
    data.train_batch_size=$ROLLOUT_SIZE \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    custom_reward_function.path=verl/utils/reward_score/call_gen_prm_binary.py \
    custom_reward_function.name=compute_pref_batched \
    reward_model.reward_manager=batch_pref \
    custom_reward_function.reward_kwargs.model_name=ilgee/Binary-Think-RM-8B \
    custom_reward_function.reward_kwargs.vllm_server_ip="vllm_server_ip" \
    custom_reward_function.reward_kwargs.n_port=8 \
    custom_reward_function.reward_kwargs.max_tokens=2048 \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$((ROLLOUT_SIZE / 4)) \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=$KL_COEF \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=20 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=20 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=10000 \
    trainer.test_freq=10000 \
    trainer.total_epochs=$EPOCH $@

## Check Last Iteration Number
LAST_ITER="$(cat /workspace/ckpt/$PROJECT_NAME/$RUN_NAME/latest_checkpointed_iteration.txt)"

### Prepare HuggingFace Model
python3 PairwiseRLHF/scripts/model_merger.py \
                        --backend fsdp \
                        --hf_model_path $BASE_MODEL \
                        --local_dir /workspace/ckpt/$PROJECT_NAME/$RUN_NAME/global_step_${LAST_ITER}/actor \
                        --target_dir /workspace/ckpt/$PROJECT_NAME/$RUN_NAME/global_step_${LAST_ITER}/actor/huggingface