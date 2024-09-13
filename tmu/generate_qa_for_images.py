import os
from pathlib import Path
from pprint import pprint
import json
from collections import defaultdict

# example:
# {"id": 2, "image": "images/8.png", "conversations": [{"from": "human", "value": "<image>\nWhich belongs to grass family?None\nStipagrostis plumosa\nWoody undershrub\nWoody perennial Please answer the question based on the options mentioned before."}, {"from": "gpt", "value": "Stipagrostis plumosa"}]}

# query = [
#     'You are a robot in an office environment. You are asked to select an action based on your image observation. Below, we have provided a list of actions along with a few image examples associated with each action.', 

#     '\nYou should choose action "A" if the image resembles following image examples:',
#     PosixPath('/home/tmu/tmu_projects/ros-visual-nav/tmu/data/retrial_test_v0/bottle_to_human/1-bottle_to_human-Pick_up_bottle-Push_bottle_right_hand.png'),
#     PosixPath('/home/tmu/tmu_projects/ros-visual-nav/tmu/data/retrial_test_v0/bottle_to_human/3-bottle_to_human-Pick_up_bottle-Push_bottle_right_hand.png'),
#     PosixPath('/home/tmu/tmu_projects/ros-visual-nav/tmu/data/retrial_test_v0/bottle_to_human/5-bottle_to_human-Pick_up_bottle-Push_bottle_right_hand.png'),

#     '\nYou should choose action "B" if the image resembles following image examples:',
#     PosixPath('/home/tmu/tmu_projects/ros-visual-nav/tmu/data/retrial_test_v0/bottle_to_human/2-bottle_to_human-Push_bottle_right_hand-Pick_up_bottle.png'), 

#     '\nYour image observation is described by the image below. Please examine it and determine which action you should choose. You should select the action where the associated image examples are most similar to your image observation below. Think step by step, pay attention to tiny details, explaining your reasoning, then reply with the your selected action on a new line. You must select an action. Your options are: "A", "B".\n',
# ]

from llava.constants import DEFAULT_IMAGE_TOKEN

BETTER_SKILL_NAME = {
    'Center from right': 'Move to the left',
    'Center from left': 'Move to the right',
    'Move back': 'Move backward',
    'Left straighten': 'Move to the left',
    'Push bottle right hand': 'Push bottle to the center',
}

SKILL_TYPE_TO_RECOVERY_SKILLS = {
    'general': [
        'Move to the left',
        'Move to the right',
        'Move forward',
        'Move backward',
    ],
    'go to door': [
        'Move to the left',
        'Move to the right',
    ]
}

RAW_SKILL_TO_RECOVERY_SKILLS = {
    'Go to restroom': SKILL_TYPE_TO_RECOVERY_SKILLS['go to door'],
    'Go to toilet': SKILL_TYPE_TO_RECOVERY_SKILLS['general'],
    'Drive to charger': SKILL_TYPE_TO_RECOVERY_SKILLS['go to door'],
    'Pick up bottle': ['Push bottle to the center'],
}


def generate_qa(current_skill, next_skill):
    current_skill = current_skill.replace('_', ' ')
    next_skill = next_skill.replace('_', ' ')

    # Generate question
    recovery_skills = RAW_SKILL_TO_RECOVERY_SKILLS[current_skill] + ['Do nothing']
    prompts = [ 
        f'{DEFAULT_IMAGE_TOKEN}\nYou are a robot in an office environment. Your goal is to complete the task "{current_skill}". You are asked to select an action based on your image observation:\n',
        'Your options are:\n' + '\n'.join(recovery_skills),
        'Please answer the question based on the options mentioned before.', 
    ]
    qs = '\n'.join(prompts)
    # print(qs)

    # Get answers
    label = BETTER_SKILL_NAME.get(next_skill, next_skill)
    if label not in recovery_skills:
        label = 'Do nothing'

    return qs, label

if __name__ == '__main__':
    source_image_folder = '/mnt/nvme_bulk/home/tmu/data/retrial/finetune_retrial_0808/restroom_door_manual_filter'
    output_dir = '/mnt/nvme_bulk/home/tmu/data/retrial'
    dataset_name = 'restroom_v1_3'
    current_skill = 'Go_to_restroom'

    json_group_by_label = defaultdict(list)

    output_file = os.path.join(output_dir, f'{dataset_name}_all.jsonl')
    with open(output_file, 'w') as f:
        image_paths = list(Path(source_image_folder).rglob("*.png"))
        for (data_cnt, image_path) in enumerate(image_paths):
            # image_id, overall_task, current_skill, next_skill = image_path.stem.split('-')
            # data_id = overall_task + '-' + image_id

            _, _, _, next_skill = image_path.stem.split('-')
            data_id = current_skill + '-' + str(data_cnt)

            qs, ans = generate_qa(current_skill, next_skill)

            json_dict = {
                "id": data_id,
                "image": os.path.relpath(image_path, source_image_folder),
                "conversations": [
                    {"from": "human", "value": qs},
                    {"from": "gpt", "value": ans},
                ],
            }
            pprint(json_dict)
            json_group_by_label[ans].append(json_dict)

            json.dump(json_dict, f)
            f.write('\n')

    ####################################
    # Train/val split
    ####################################
    train_jsons = []
    val_jsons_by_label = {}
    for label, jsons in json_group_by_label.items():
        n_val = int(len(jsons) * 0.2 + 0.5)
        val_jsons_by_label[label] = jsons[:n_val]
        train_jsons.extend(jsons[n_val:])

    for label, jsons in val_jsons_by_label.items():
        label = label.replace(' ', '_')
        with open(os.path.join(output_dir, f'{dataset_name}_val_{label}.jsonl'), 'w') as f:
            for json_dict in jsons:
                json.dump(json_dict, f)
                f.write('\n')

    with open(os.path.join(output_dir, f'{dataset_name}_train.jsonl'), 'w') as f:
        for json_dict in train_jsons:
            json.dump(json_dict, f)
            f.write('\n')

    # breakpoint()