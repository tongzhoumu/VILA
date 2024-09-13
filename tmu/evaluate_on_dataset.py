import argparse
import base64
import datetime
import json
import os
import time
from distutils.util import strtobool
from io import BytesIO
from pathlib import Path

import numpy as np
import requests
from PIL import Image


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=None,
        help="the name of this experiment")
    parser.add_argument("--model", type=str, default='gpt-4o',
        help="the name of the VLM model") # "claude-3-5-sonnet-20240620", "gemini-1.5-flash"
    parser.add_argument("--data-dir", type=str, default='tmu/data/test_v3/bottle_to_human')
    parser.add_argument("--support-root-dir", type=str, default='tmu/data/support')
    parser.add_argument("--one-sample", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default='tmu/vlm_prompting_exp')
    parser.add_argument("--prompt-version", type=int, required=True)
    parser.add_argument("--verbose", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
    parser.add_argument("--detail", type=str, default='high', choices=['high', 'low'])
    parser.add_argument("--crop", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
    parser.add_argument("--repeat", type=int, default=1)

    args = parser.parse_args()
    # fmt: on
    return args

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def get_sorted_image_paths_from_dir(root_dir):
    image_paths = list(Path(root_dir).rglob("*.png"))

    def extract_index(x):
        x = x.stem.split("-")[0]
        if x.isdigit():
            return int(x)
        return x

    return sorted(image_paths, key=extract_index)


def get_response(image_path, text_prompt, model="gpt-4o", max_tokens=1024, logger=None, **kwargs):
    if isinstance(text_prompt, list):
        raise NotImplementedError
    base64_image = encode_image(image_path, crop_area=(256, 512, 768, 1024) if crop else None)
    image_format = image_path.split(".")[-1]
    if image_format == "jpg":
        image_format = "jpeg"

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/{image_format};base64,{base64_image}"}},
                ],
            }
        ],
        "max_tokens": max_tokens,
    }
    try:
        response = requests.post("http://127.0.0.1:5000/v1/chat/completions", json=payload)
        response = response.json()
        response = response["choices"][0]["message"]["content"]
    except:
        if logger is not None:
            logger.error(f"Error in response: {str(response)}")
        response = "Error in response"
    return response


if __name__ == "__main__":
    args = parse_args()
    # if args.one_sample is not None:
    #     args.exp_name = "debug"

    # Set up logger
    tag = datetime.datetime.now().strftime("%y%m%d-%H%M%S") + "_" + args.model
    if args.exp_name:
        tag += "_" + args.exp_name
    log_dir_name = tag
    test_set_name = args.data_dir.split("/")[-1]
    log_dir_path = os.path.join(args.output_dir, test_set_name, f"prompt_v{args.prompt_version}", log_dir_name)
    os.makedirs(log_dir_path, exist_ok=True)
    logger = setup_logger(
        os.path.join(log_dir_path, "output.log"),
        console_level=logging.DEBUG if args.exp_name == "debug" or args.verbose else logging.INFO,
        console_only=args.exp_name == "debug",
    )

    model_kwargs = {}
    model_kwargs["detail"] = args.detail
    args.model_kwargs = model_kwargs
    if args.exp_name != "debug":
        with open(f"{log_dir_path}/args.json", "w") as f:
            json.dump(vars(args), f, indent=4)
        save_script(f"{log_dir_path}/code.py")

    #################################
    # Test
    #################################
    success_cnt = 0
    total_cnt = 0
    if args.one_sample is None:
        image_paths = get_sorted_image_paths_from_dir(args.data_dir)
    else:
        image_paths = [args.one_sample]

    all_time_cost = []
    for repeat_idx in range(args.repeat):
        logger.info(f"############### Round {repeat_idx} Eval ###############")

        for img_idx, image_path in enumerate(image_paths):
            total_cnt += 1
            image_path = str(image_path)
            base_name = image_path.split("/")[-1].split(".")[0]
            sample_idx, overall_task, current_skill, label = base_name.split("-")
            # if int(sample_idx) < 44:
            #     continue
            overall_task_description = TASK_NAME_TO_DESCRIPTION[overall_task]
            skill_seq = TASK_NAME_TO_SKILL_SEQ[overall_task]
            current_skill = current_skill.replace("_", " ")
            label = label.replace("_", " ")

            pred, time_cost = predict_one_sample(
                image_path=image_path,
                prompt_version=args.prompt_version,
                overall_task=overall_task,
                goal_description=overall_task_description,
                skill_seq=skill_seq,
                current_skill=current_skill,
                model=args.model,
                model_kwargs=model_kwargs,
                logger=logger,
                crop=args.crop,
            )
            all_time_cost.append(time_cost)

            # TODO: save an example text prompt
            pred = post_process_response(pred, logger=logger)
            success_cnt += int(pred == label)
            logger.info(f"{sample_idx}: {pred == label},\tlabel: {label},\tpred: {pred},\tTime Cost: {time_cost:.2f},\tAcc: {success_cnt / total_cnt:.2f}")
            # break

            if current_skill == skill_seq[-2] and label == skill_seq[-1]:
                break
            if args.exp_name == "debug":
                import pdb

                pdb.set_trace()

    print(f'Average query time: {np.mean(all_time_cost):.2f} s')
    print("@@ Eval is done.")
