set -e

export HF_TOKEN="your HF token here"

cd OpenRLHF

TEMPLATE="naive-reasoning-binary"
TYPE="max"
EPOCH=5
LR=1e-5

if [[ "$TEMPLATE" == "naive-reasoning-binary" ]]; then
   TEMP="naive-binary"
elif [[ "$TEMPLATE" == "naive-reasoning-multiclass" ]]; then
   TEMP="naive-multiclass"
fi

RUN_NAME=hs2-${TEMP}-${TYPE}-ep${EPOCH}-lr${LR}
deepspeed --module openrlhf.cli.train_sft \
   --save_path /workspace/$RUN_NAME \
   --save_steps -1 \
   --logging_steps 5 \
   --eval_steps -1 \
   --micro_train_batch_size 2 \
   --train_batch_size 256 \
   --gradient_checkpointing \
   --zero_stage 3 \
   --bf16 \
   --flash_attn \
   --max_epochs $EPOCH \
   --pretrain meta-llama/Llama-3.1-8B-Instruct \
   --learning_rate $LR \
   --lr_warmup_ratio 0.03 \
   --l2 0.0 \
   --adam_betas 0.9 0.95 \
   --dataset ilgee/hs2-${TEMPLATE}-${TYPE} \
   --train_split train \
   --input_key chosen \
   --apply_chat_template \
   --tokenizer_chat_template "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}" \
   --max_len 16384 \
   --use_wandb "your wandb key here" \
   --wandb_project think_genrm_sft_${TEMP} \
   --wandb_run_name $RUN_NAME