import openai
import argparse
import os
import json
import concurrent.futures
from tqdm.auto import tqdm
import openai
import traceback


def save_json(file_path, data):
    with open(file_path, "w") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def load_json(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data


def load_jsonl(file_path):
    all_data = []
    with open(file_path, "r") as f:
        file_lines = f.read().strip().split("\n")
    for line in file_lines:
        line = json.loads(line)
        all_data.append(line)
    return all_data


chat_prompt = """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>human
{}<|im_end|>
<|im_start|>gpt
"""


def process_item(one_item):
    try:
        model = "Qwen-72B-tq"  # 写死的模型
        prompt = one_item["conversations"][0]["value"]
        prompt = chat_prompt.format(prompt)
        completion = openai.Completion.create(
            model=model,
            prompt=prompt,
            echo=False,
            stop=["<|im_end|>"],
            stream=False,
            temperature=0,
            do_sample=False,
            logprobs=10000,
            max_tokens=10000,
            skip_special_tokens=False,
        )
        return completion.to_dict_recursive()
    except:
        return {"result": "处理失败", "details": traceback.format_exc()}


def run_one(one_item, args):
    data_file = one_item["data_file"]

    num_run = one_item["num_run"]
    if ".jsonl" in data_file:
        test_data = load_jsonl(data_file)
    else:
        test_data = load_json(data_file)

    save_dir = os.path.join(args.save_dir, f"{one_item['task_name']}")
    os.makedirs(save_dir, exist_ok=True)

    output_file = os.path.join(
        save_dir, f"{one_item['task_name']}_{args.step_num}.jsonl"
    )

    test_data = test_data[:num_run]
    print(f"Total data num is {len(test_data)}, but we just run {num_run}")

    n_jobs = min(args.n_jobs, len(test_data))
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_jobs) as executor:
        result_list = list(
            tqdm(
                executor.map(process_item, test_data), total=len(test_data), miniters=10
            )
        )

    save_json(output_file, result_list)


def run(args):
    task_config_list = load_json(args.task_config)
    for one_item in task_config_list:
        run_one(one_item, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task_config",
        type=str,
        help="test data configs",
    )
    parser.add_argument("--save_dir", type=str, help="保存的文件夹")
    parser.add_argument("--step_num", type=int, default=1)
    parser.add_argument(
        "--n_jobs",
        type=int,
        help="Number of threads to use for parallel processing",
        default=50,
    )
    parser.add_argument("--num_run", type=int, default=100000)

    parser.add_argument(
        "--port",
        type=int,
        help="Port number for the API server",
        default=8002,
    )
    parser.add_argument(
        "--host", type=str, help="Host for the API server", default="localhost"
    )
    args = parser.parse_args()
    # Modify OpenAI's API key and API base to use vLLM's API server.
    openai.api_key = "EMPTY"
    openai.api_base = f"http://{args.host}:{args.port}/v1"
    print(args)
    run(args)
