# CE-LessonPlan

## Installation

```
cd CE-LessonPlan
pip install -r requirements.txt
```

## Lesson Plan Generator

### Base models

Download the [Qwen series](https://huggingface.co/Qwen)

### Model finetuning

```
cd run_model

#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
DIR=$( dirname -- "$( readlink -f -- "$0"; )"; )
# pip install transformers==4.37.2 accelerate==0.27.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
# pip install transformers==4.37.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
# pip install transformers==4.32.0 accelerate tiktoken einops scipy transformers_stream_generator==0.0.4 peft deepspeed -i https://pypi.tuna.tsinghua.edu.cn/simple


MODEL="base model path"

DATA="train data path"

SAVEDIR="save model path"

GPUS_PER_NODE=8
# WORLD_SIZE=1
# MASTER_PORT=6000
# RANK=0
# MASTER_ADDR="localhost"
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
is_master=${MASTER-"0"}
if [[ $is_master -eq 1 ]];then
    DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $WORLD_SIZE --node_rank $RANK --master_addr $HOSTNAME --master_port $MASTER_PORT"
else
    DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $WORLD_SIZE --node_rank $RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
fi

torchrun $DISTRIBUTED_ARGS ./qwen2_finetune.py \
    --model_name_or_path $MODEL \
    --data_path $DATA \
    --bf16 True \
    --output_dir $SAVEDIR \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 200 \
    --learning_rate 5e-6 \
    --weight_decay 0.0 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 4096 \
    --lazy_preprocess True \
    --gradient_checkpointing True \
    --deepspeed ./ds_config_zero3.json

```

## Ensemble Decoding

```
#!/bin/bash

DATASET="lesson_plan"

MODEL="lesson_plan_generator_path" # 压缩模型
TARGET_MODEL="lesson_plan_generator_path" # 解压模型

python decoding.py \
--compression_model $MODEL \
--expansion_model $TARGET_MODEL \
--alpha 0.6 \
--decoding_len 2048 \
--batch_size 6 \
--dataset_name $DATASET
```


## Lesson Plan Generation with Ensemble Decoding Results

```
#!/bin/bash

set -e
#set -e ensures that if any command (like cd, bash, python3, etc.) fails (returns a non-zero status), the script will immediately stop executing and exit. This avoids continuing to execute subsequent commands in the event of an error.

model_path="model_path" #test_data/lesson_plan_with_decoding_outputs.json
data_file="test data path"
output_file="output data path"
port=8000
n_jobs=5



# Start the inference service
python3 -m vllm.entrypoints.openai.api_server \
    --model $model_path --tensor-parallel-size=4 --trust-remote-code \
    --gpu-memory-utilization=0.8 \
    --enforce-eager --max-context-len-to-capture=4096 \
    --served-model-name=Qwen-72B  >> output.log &

# Wait for the inference service to start, set a timeout of 20 minutes
timeout_limit_seconds=1200  # 20 mins
remaining_timeout_seconds=$timeout_limit_seconds
print_interval=30  # Print the remaining time every 30 seconds
while ! nc -z localhost $port; do
  sleep 10
  ((remaining_timeout_seconds-=10))
  if [ $remaining_timeout_seconds -gt 0 ]; then
    if [ $((remaining_timeout_seconds % print_interval)) -eq 0 ]; then
      echo "Waiting for the inference service to start, the remaining waiting time is $remaining_timeout_seconds seconds"
    fi
  else
    echo "Inference service startup timeout"
    exit 1
  fi
done

echo "The inference service started successfully, taking $(($timeout_limit_seconds - $remaining_timeout_seconds)) seconds"


# Start the prediction task
python3 ./infer_by_api4lp.py --data_file ${data_file} --output_file ${output_file} --n_jobs ${n_jobs} --port ${port}

```
