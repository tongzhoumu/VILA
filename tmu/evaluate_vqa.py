import argparse
import json
import os
from typing import Optional
import math
import re

from tqdm import tqdm

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path

from PIL import Image


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, required=True)
    parser.add_argument("--annotation-file", type=str, required=True)
    # parser.add_argument("--num-chunks", type=int, default=1)
    # parser.add_argument("--chunk-idx", type=int, default=0)
    args = parser.parse_args()

    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    # tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_name, args.model_base,)
    import torch
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_name, args.model_base, load_4bit=True, torch_dtype=torch.float16)

    questions = [json.loads(q) for q in open(args.annotation_file, "r")]
    # questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    outputs = []
    cnts = [0, 0]
    # for line in tqdm(questions):
    for line in questions:
        assert line['conversations'][0]['from'] == 'human'
        qs = line['conversations'][0]['value']
        assert line['conversations'][1]['from'] == 'gpt'
        annotation = line['conversations'][1]['value']

        qs_id = line['id']
        image_file = line['image']
        images = [image_file]
        # image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        # if IMAGE_PLACEHOLDER in qs:
        #     if model.config.mm_use_im_start_end:
        #         qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        #     else:
        #         qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
        # else:
        #     if DEFAULT_IMAGE_TOKEN not in qs:
        #         print("no <image> tag found in input. Automatically append one at the beginning of text.")
        #         # do not repeatively append the prompt.
        #         if model.config.mm_use_im_start_end:
        #             qs = (image_token_se + "\n") * len(images) + qs
        #         else:
        #             qs = (DEFAULT_IMAGE_TOKEN + "\n") * len(images) + qs

        # qs = (DEFAULT_IMAGE_TOKEN + "\n") * len(images) + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        image = Image.open(os.path.join(args.image_folder, image_file))
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]


        # from llava.data.dataset import preprocess_v1

        # sources = [line['conversations']]
        # inputs = preprocess_v1(sources, tokenizer, has_image=True, no_system_prompt=False)
        # breakpoint()

        # inputs = {
        #     'input_ids': input_ids,
        #     'labels': None,
        #     'attention_mask': input_ids.ne(tokenizer.pad_token_id).long().cuda(),
        #     'images': image_tensor.unsqueeze(0).half().cuda(),
        # }
        # outputs = model(**inputs)
        # breakpoint()

        pred = model.generate(
            input_ids=input_ids.cuda(),
            images=image_tensor.unsqueeze(0).half().cuda(),
            do_sample=False,
            num_beams=1,
            max_new_tokens=512,
            num_return_sequences=1,
            use_cache=True
        )

        answer = tokenizer.batch_decode(pred, skip_special_tokens=True)[0]
        answer = answer.strip()

        answer = answer.split('\n')[-1]
        annotation = annotation.split('\n')[-1]

        correct = (answer == annotation)
        cnts[correct] += 1
        print(f'{qs_id}: {annotation == answer},\tlabel: {annotation},\tpred: {answer},\tacc: {cnts[1]*1.0/sum(cnts)*100:.1f}%')
