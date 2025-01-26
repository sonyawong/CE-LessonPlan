import os

os.environ["WORLD_SIZE"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import argparse
import json
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from prompts import *
from generate_prompt import generate_question
import random
import string


def main(args):

    modelname = args.compression_model.split('/')[-1]
    print(f"model_name:{modelname}")
    output_file_name = (
        f"{modelname}_k{args.topk}_a{args.alpha}_{args.dataset_name}"
        + ".json"
    )

    print("dataset:", args.dataset_name)
    print("alpha: ", args.alpha)
    # print("top-k: ", args.topk)

    # load dataset
    print("Loading Dataset...")
    with open(os.path.join(args.data_dir, args.dataset_name + ".jsonl"), "r") as f:
        dataset = f.readlines()
        print(len(dataset), "Samples")

    with open(os.path.join(args.output_dir, output_file_name), "w") as f_write:
        # start inference
        print("Start Inference...")
        model.eval()
        expansion_model.eval()

        # generate batch
        lp_comp_messages = []
        tb_gen_prompts = []
        modules = []
        questions = []
        lp_ids = []
        textbooks = []
        refs = []
        for idx, data in enumerate(dataset):
            data = json.loads(data)
            question = "_".join(
                [
                    data["subject"],
                    data["level"],
                    data["version"],
                    data["grade"],
                    data["topic"],
                ]
            )
            lp_id = data["lp_id"]

            if len(data["ctxs"]) != 0:
                lp_comp_prompt = generate_question(
                    data, example_source="ctxs", select_nums=args.topk
                )  
            else:
                lp_comp_prompt = generate_question(data, example_source="textbooks")                 

            tb_gen_prompt = generate_question(data, example_source="textbooks") 

            lp_comp_message = [
                {"role": "system", "content": ""},
                {"role": "user", "content": lp_comp_prompt},
            ]
            tb_gen_prompt = [
                {"role": "system", "content": ""},
                {"role": "user", "content": tb_gen_prompt},
            ]

            lp_comp_messages.append(lp_comp_message)
            tb_gen_prompts.append(tb_gen_prompt)
            questions.append(question)  
            modules.append(data["module"])
            lp_ids.append(lp_id)  
            textbooks.append(data["textbook"])
            refs.append("\n".join(data["ctxs"]))

        nbatch = (len(lp_comp_messages) - 1) // args.batch_size + 1
        for k in tqdm(range(nbatch)):
            print(f"Start Inference Batch {k+1}...")

            start_idx = k * args.batch_size
            end_idx = min((k + 1) * args.batch_size, len(lp_comp_messages))
            batch_size = end_idx - start_idx
            lp_comp_messages_batch = lp_comp_messages[start_idx:end_idx]
            tb_gen_prompts_batch = tb_gen_prompts[start_idx:end_idx]
            module_batch = modules[start_idx:end_idx]
            question_batch = questions[start_idx:end_idx]
            lp_id_batch = lp_ids[start_idx:end_idx]
            textbooks_batch = textbooks[start_idx:end_idx]
            refs_batch = refs[start_idx:end_idx]

            lp_comp_outputs = tokenizer.apply_chat_template(
                lp_comp_messages_batch,
                add_generation_prompt=False,
                padding="longest",
                return_dict=True,
                return_tensors="pt",
            )

            tb_gen_outputs = tokenizer.apply_chat_template(
                tb_gen_prompts_batch,
                add_generation_prompt=False,
                padding="longest",
                return_dict=True,
                return_tensors="pt",
            )

            if args.compression_model == "meta-llama/Meta-Llama-3-8B-Instruct":
                gen_tokens = torch.tensor(
                    [128006, 78191, 128007, 271]
                )  # <|start_header_id|>assistant<|end_header_id|>\n\n
                gen_tokens = gen_tokens.repeat(batch_size, 1)
                gen_att = torch.tensor([1, 1, 1, 1])
                gen_att = gen_att.repeat(batch_size, 1)
                lp_comp_ids = torch.cat(
                    [lp_comp_outputs.input_ids, gen_tokens], dim=-1
                ).to(
                    model.device
                ) 
                tb_gen_ids = torch.cat([tb_gen_outputs.input_ids, gen_tokens], dim=-1).to(
                    expansion_model.device
                )  
                attention_mask = torch.cat(
                    [lp_comp_outputs.attention_mask, gen_att], dim=-1
                ).to(
                    model.device
                )  
                attention_mask_expansion = torch.cat(
                    [tb_gen_outputs.attention_mask, gen_att], dim=-1
                ).to(
                    expansion_model.device
                ) 
            else:
                lp_comp_ids = lp_comp_outputs.input_ids
                tb_gen_ids = tb_gen_outputs.input_ids
                attention_mask = lp_comp_outputs.attention_mask
                attention_mask_expansion = tb_gen_outputs.attention_mask

            print(f"Batch {k+1} ensemble decoding...")

            # ensemble decoding
            gen_ids = None
            past_key_values = None
            past_key_values_expansion = None
            for step in range(args.decoding_len):
                with torch.no_grad():

                    outputs = model(
                        input_ids=lp_comp_ids,
                        past_key_values=past_key_values,
                        use_cache=True,
                        attention_mask=attention_mask,
                    )
                    lm_logits = outputs.logits
                    past_key_values = outputs.past_key_values
                    logit_for_next_step = lm_logits[:, -1:]
                    # print(f"logit_for_next_step:{logit_for_next_step.shape}")
                    next_id = torch.argmax(logit_for_next_step, axis=-1)

                    if args.alpha != 0.0:
                        expansion_outputs = expansion_model(
                            input_ids=tb_gen_ids,
                            past_key_values=past_key_values_expansion,
                            use_cache=True,
                            attention_mask=attention_mask_expansion,
                        )
                        lm_logits = expansion_outputs.logits
                        past_key_values_expansion = expansion_outputs.past_key_values
                        logit_for_next_step_expansion = lm_logits[:, -1:]
                        # print(f"logit_for_next_step_expansion:{logit_for_next_step_expansion.shape}")
                        ensembled_logits = (
                            logit_for_next_step * (1.0 - args.alpha)
                            + logit_for_next_step_expansion * args.alpha
                        )
                        next_id = torch.argmax(ensembled_logits, axis=-1)

                    if gen_ids == None:
                        gen_ids = next_id
                    else:
                        gen_ids = torch.cat([gen_ids, next_id], dim=-1)

                    complete = True
                    for i in range(len(gen_ids)):
                        if (
                            tokenizer.eos_token_id not in gen_ids[i]
                            and tokenizer.pad_token_id not in gen_ids[i]
                        ):
                            complete = False
                            break
                    if complete:
                        break

                    lp_comp_ids = tb_gen_ids = next_id
                    next_id_mask = next_id != tokenizer.eos_token_id
                    attention_mask = torch.cat([attention_mask, next_id_mask], dim=-1)
                    attention_mask_expansion = torch.cat(
                        [attention_mask_expansion, next_id_mask], dim=-1
                    )

            # decode
            print(f"Batch {k+1} finally decoding...")

            for i in range(len(gen_ids)):
                if tokenizer.eos_token_id in gen_ids[i]:
                    end = gen_ids[i].tolist().index(tokenizer.eos_token_id)
                    for j in range(end, len(gen_ids[i])):
                        gen_ids[i][j] = tokenizer.eos_token_id
                if tokenizer.pad_token_id in gen_ids[i]:
                    end = gen_ids[i].tolist().index(tokenizer.pad_token_id)
                    for j in range(end, len(gen_ids[i])):
                        gen_ids[i][j] = tokenizer.pad_token_id
            gen_ctx_batch = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

            # save
            for i in range(batch_size):
                json_output = {
                    "id": lp_id_batch[i],
                    "module": module_batch[i],
                    "question": question_batch[i],
                    "gen_ctx": gen_ctx_batch[i],
                    "refs": refs_batch[i],
                    "textbook": textbooks_batch[i],
                }
                f_write.write(json.dumps(json_output, ensure_ascii=False) + "\n")
            print(f"Batch {k+1} finished ...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--compression_model",
        "-m",
        type=str,
        help="compression_model"
    )
    parser.add_argument(
        "--expansion_model",
        "-tm",
        type=str,
        help="expansion_model"
    )
    parser.add_argument("--alpha", "-alpha", type=float, default=0.5)
    parser.add_argument(
        "--decoding_len", "-len", type=int, help="decoding length", default=2048
    )
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--data_dir", type=str, default="./data/test_data/")
    parser.add_argument("--output_dir", type=str, default="outputs/")
    parser.add_argument("--dataset_name", "--dataset", type=str, default="lesson_plan")
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()

    args.output_dir = os.path.join(
        args.output_dir, args.expansion_model.split("/")[-1]
    )
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # load model
    tokenizer = AutoTokenizer.from_pretrained(args.compression_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.compression_model,
        device_map="auto",
        load_in_8bit=False,
        torch_dtype=torch.float16,
    )
    print("compression Model:", args.compression_model)
    if args.compression_model == args.expansion_model:
        print("Expansion Model:", args.compression_model)
        expansion_model = model
    else:
        print("Expansion Model:", args.expansion_model)
        expansion_model = AutoModelForCausalLM.from_pretrained(
            args.expansion_model,
            device_map="auto",
            load_in_8bit=False,
            torch_dtype=torch.float16,
        )

    main(args)
