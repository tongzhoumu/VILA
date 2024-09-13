import os
from pathlib import Path
from pprint import pprint
import json
from collections import defaultdict
import numpy as np

from llava.constants import DEFAULT_IMAGE_TOKEN

def generate_qa(task, options, label):
    # Generate question
    prompts = [ 
        f'{DEFAULT_IMAGE_TOKEN}\nYou are a robot in a warehouse environment. Your goal is to complete the task "{task}". You are asked to select an action based on your image observation:\n',
        'Your options are:\n' + '\n'.join(options),
        'Please select one of the options mentioned before.', 
    ]
    qs = '\n'.join(prompts)
    # print(qs)

    # Get answers
    # do nothing

    return qs, label


def generate_qa_cot_old(task, options, reasons, label):
    # Generate question
    prompts = [ 
        f'{DEFAULT_IMAGE_TOKEN}\nYou are a robot in a warehouse environment. Your goal is to complete the task "{task}". You are asked to select an action based on your image observation:\n',
        'Your options are:\n' + '\n'.join(options),
    ] + [
        f'You should select "{option}" if {reason}.' for option, reason in zip(options, reasons)
    ] + [
        'Please select one of the options mentioned before. Do not include quotes.', 
    ]
    qs = '\n'.join(prompts)
    # print(qs)

    # Get answers
    # do nothing

    return qs, label


def generate_qa_cot(task, options, reasons, label):
    # Generate question
    prompts = [ 
        f'{DEFAULT_IMAGE_TOKEN}\nYou are a robot in a warehouse environment. Your goal is to complete the task "{task}". You are asked to select an action based on your image observation:\n',
        'Your options are:\n' + '\n'.join(options),
        'Please think step by step, then select one of the options in a new line.', 
    ]
    qs = '\n'.join(prompts)
    # print(qs)

    # Get answers
    i = options.index(label)
    answers = [
        f'Based on the image, {reasons[i]}, so the correct option is "{label}".',
        label,
    ]
    answer = '\n'.join(answers)

    return qs, answer


if __name__ == '__main__':
    source_image_folder = '/mnt/nvme_bulk/home/tmu/data/sbr/paper'
    output_dir = '/mnt/nvme_bulk/home/tmu/data/sbr'
    dataset_name = 'paper_v4_0'
    task = 'Put paper into the box'
    options = [
        'Pick up paper',
        'Take paper from machine',
    ]
    reasons = [
        'there is a paper outside but near the box',
        'there are no papers outside but near the box',
    ]
    val_ratio = 0.1

    json_group_by_label = defaultdict(list)

    output_file = os.path.join(output_dir, f'{dataset_name}_all.jsonl')
    with open(output_file, 'w') as f:
        image_paths = list(Path(source_image_folder).rglob("*.png"))
        if dataset_name in ['paper_v0_2', 'paper_v3_0', 'paper_v4_0']:
            # only use the data from SBR
            image_paths = [p for p in image_paths if 'sunny' not in str(p)]
        if dataset_name.startswith('paper_v4'):
            image_paths = [p for p in image_paths if 'Take_paper_from_machine-VR' not in str(p)]
        for (data_cnt, image_path) in enumerate(image_paths):
            infos = image_path.stem.split('-')
            if 'sunny' in str(image_path):
                label = infos[-1].replace('_', ' ')
            else:
                label = infos[3].replace('_', ' ')
            assert label in options, f'label not in options: {label} not in {options}'
            # print(label)
            data_id = str(data_cnt)

            qs, ans = generate_qa(task, options, label)
            # qs, ans = generate_qa_cot(task, options, reasons, label)

            json_dict = {
                "id": data_id,
                "image": os.path.relpath(image_path, source_image_folder),
                "conversations": [
                    {"from": "human", "value": qs},
                    {"from": "gpt", "value": ans},
                ],
            }
            # pprint(json_dict)
            json_group_by_label[label].append(json_dict)

            json.dump(json_dict, f)
            f.write('\n')

    ####################################
    # Train/val split
    ####################################
    np.random.seed(0)
    train_jsons = []
    val_jsons_by_label = {}
    train_jsons_by_label = {}
    for label, jsons in json_group_by_label.items():
        n_val = int(len(jsons) * val_ratio + 0.5)
        jsons = np.random.permutation(jsons) # random shuffle jsons
        # only select sample w/o augmentation into val set
        val_jsons_this_label = []
        train_jsons_this_label = []

        if label == 'Pick up paper' and ( dataset_name.startswith('paper_v3') or dataset_name.startswith('paper_v4') ):
            val_jsons_this_label = [j for j in jsons if 'Auto' in j['image']]
            train_jsons_this_label = [j for j in jsons if 'Auto' not in j['image']]
        else:
            for json_dict in jsons:
                filename = Path(json_dict['image']).stem
                with_aug = filename.split('-')[-1].replace('_', ' ') not in options
                if len(val_jsons_this_label) < n_val and not with_aug:
                    val_jsons_this_label.append(json_dict)
                else:
                    train_jsons_this_label.append(json_dict)

        val_jsons_this_label = sorted(val_jsons_this_label, key=lambda x: int(x['id']))
        train_jsons_this_label = sorted(train_jsons_this_label, key=lambda x: int(x['id']))
        val_jsons_by_label[label] = val_jsons_this_label
        train_jsons_by_label[label] = train_jsons_this_label
        train_jsons.extend(train_jsons_this_label)

    with open(os.path.join(output_dir, f'{dataset_name}_train.jsonl'), 'w') as f:
        for json_dict in train_jsons:
            json.dump(json_dict, f)
            f.write('\n')

    for label, jsons in val_jsons_by_label.items():
        label = label.replace(' ', '_')
        with open(os.path.join(output_dir, f'{dataset_name}_val_{label}.jsonl'), 'w') as f:
            for json_dict in jsons:
                json.dump(json_dict, f)
                f.write('\n')

    for label, jsons in train_jsons_by_label.items():
        label = label.replace(' ', '_')
        with open(os.path.join(output_dir, f'{dataset_name}_train_{label}.jsonl'), 'w') as f:
            for json_dict in jsons:
                json.dump(json_dict, f)
                f.write('\n')
    
    # breakpoint()