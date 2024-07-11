from llava.mm_utils import get_model_name_from_path

import torch

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    KeywordsStoppingCriteria,
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

import re

class LargeModel:
    def __init__(self, model_path, conv_mode=None):
        self.conv_mode = conv_mode

        disable_torch_init()

        self.model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, context_len = load_pretrained_model(model_path, model_name=self.model_name, model_base=None, 
        load_4bit=True, torch_dtype=torch.float16)

    def predict(self, text_prompt, pil_image):
        images = [pil_image]
        model = self.model
        model_name = self.model_name
        image_processor = self.image_processor
        tokenizer = self.tokenizer


        qs = text_prompt
        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        if IMAGE_PLACEHOLDER in qs:
            if model.config.mm_use_im_start_end:
                qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
            else:
                qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
        else:
            if DEFAULT_IMAGE_TOKEN not in qs:
                print("no <image> tag found in input. Automatically append one at the beginning of text.")
                # do not repeatively append the prompt.
                if model.config.mm_use_im_start_end:
                    qs = (image_token_se + "\n") * len(images) + qs
                else:
                    qs = (DEFAULT_IMAGE_TOKEN + "\n") * len(images) + qs

        if "llama-2" in model_name.lower():
            conv_mode = "llava_llama_2"
        elif "v1" in model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"

        if self.conv_mode is not None and conv_mode != self.conv_mode:
            print(
                "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                    conv_mode, self.conv_mode, self.conv_mode
                )
            )
        else:
            self.conv_mode = conv_mode

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # image_sizes = [x.size for x in images]
        images_tensor = process_images(
            images,
            image_processor,
            model.config
        ).to(model.device, dtype=torch.float16)

        input_ids = (
            tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        temperature = 0

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=[
                    images_tensor,
                ],
                do_sample=True if temperature > 0 else False,
                temperature=temperature,
                top_p=None,
                num_beams=1,
                max_new_tokens=512,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)]
        outputs = outputs.strip()
        return outputs

MODEL_NAME = "VILA1.5-3b"
conv_mode = "vicuna_v1"
# MODEL_NAME = "Llama-3-VILA1.5-8b"
# conv_mode = "llama_3"
# MODEL_NAME = "VILA1.5-13b-AWQ"
# MODEL_NAME = "VILA1.5-13b"
# conv_mode = "vicuna_v1"
model = LargeModel(f"Efficient-Large-Model/{MODEL_NAME}", conv_mode)

########################################################

from flask import Flask, request, jsonify

from PIL import Image
import io
import base64

def decode_image(base64_string):
    image_data = base64.b64decode(base64_string)
    image_buffer = io.BytesIO(image_data)
    return Image.open(image_buffer)


app = Flask(__name__)

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    payload = request.json
    print(f"Received message")

    assert payload["model"] == MODEL_NAME
    msg = payload["messages"][0]
    assert msg['role'] == 'user'

    text_prompt = None
    pil_image = None
    for content in msg['content']:
        if content['type'] == 'text':
            assert text_prompt is None, 'Text has already been set'
            text_prompt = content['text']
        if content['type'] == 'image_url':
            assert pil_image is None, 'Image has already been set'
            meta, data = content['image_url']['url'].split(',')
            pil_image = decode_image(data)

    outputs = model.predict(
        text_prompt=text_prompt,
        pil_image=pil_image,
    )

    # Simple mock response
    response = {
        "choices": [
            {
                "message": {
                    "content": outputs
                },
            }
        ],
    }

    return jsonify(response)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)