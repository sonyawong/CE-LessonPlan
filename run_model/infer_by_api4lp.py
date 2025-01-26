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
            logprobs=0,
            max_tokens=10000,
            skip_special_tokens=False,
        )
        print(f"completion:{completion}")
        return completion.to_dict_recursive()
    except:
        return {"result": "处理失败", "details": traceback.format_exc()}


def run(args):
    if ".jsonl" in args.data_file:
        test_data = load_jsonl(args.data_file)
    else:
        test_data = load_json(args.data_file)

    save_dir = os.path.dirname(args.output_file)
    os.makedirs(save_dir, exist_ok=True)

    print(f"Total data num is {len(test_data)}, but we just run {args.num_run}")

    test_data = test_data[: args.num_run]

    n_jobs = min(args.n_jobs, len(test_data))
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_jobs) as executor:
        result_list = list(
            tqdm(
                executor.map(process_item, test_data), total=len(test_data), miniters=10
            )
        )
        for i, one_result in enumerate(result_list):
            test_data[i]["conversations"][1]["value"] = one_result["choices"][0]["text"]

    save_json(args.output_file, test_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_file",
        type=str,
        help="Path to the input data file",
        default="",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="Path to the output file",
        default="",
    )
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
