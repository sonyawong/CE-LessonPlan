import pandas as pd
import json
import random
from prompts4lp import *

def load_prompt(
    prompt_name,
    prompt_path="./prompts/",
):
    task_prompt = open(f"{prompt_path}{prompt_name}.md", "r").read()
    return task_prompt


def generate_question(
    data,
    example_source="",
    select_nums=5,
):

    with open("./data/teaching_models.json","r") as f:
        teaching_mode_info = json.load(f)

    if len(data["topic"].strip().split("@@@@")) == 1:
        first_lesson, second_lesson, kcs = data["topic"].split("@@@@")[0], "", ""
    elif len(data["topic"].strip().split("@@@@")) == 2:
        first_lesson, second_lesson, kcs = (
            data["topic"].split("@@@@")[0],
            data["topic"].split("@@@@")[1],
            "",
        )
    else:
        first_lesson, second_lesson = (
            data["topic"].split("@@@@")[0],
            data["topic"].split("@@@@")[1],
        )
        kcs = ",".join(data["topic"].split("@@@@")[2:])

    if example_source == "textbooks":
        example = data["textbook"]

        task_prompt_system = load_prompt("plan_generator_first_system")
        task_prompt_template = load_prompt("plan_generator_first_template")

        task_prompt_template = task_prompt_template.format(
            data["subject"],
            data["module"],
            data["level"],
            data["version"],
            data["grade"],
            first_lesson,
            second_lesson,
            kcs,
            data["tch_mode"],
            teaching_mode_info["模式简介"],
            teaching_mode_info["教学环节"],
            example,
        )

        question = f"<Action> generate_module </Action>\n<system start>\n{task_prompt_system}<system end>\n<input start>\n{task_prompt_template}<input end>"

    else:
        # 新的列表
        new_example_list = []
        example_list = random.sample(data["ctxs"], min(len(data["ctxs"]), select_nums))

        # 遍历列表并添加前缀
        for i, content in enumerate(example_list):
            prefix = f"教学设计{i+1}: "
            new_content = f"{prefix}{content}"
            new_example_list.append(new_content)

        example = "\n".join(new_example_list)
        question = compress_lps_template_upgrade.format(
            level=data["level"],
            grade=data["grade"],
            subject=data["subject"],
            version=data["version"],
            topic=data["topic"].replace("@@@@", ","),
            lesson_plans="\n".join(new_example_list),
        )

    return question
