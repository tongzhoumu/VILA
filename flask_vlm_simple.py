from llava.mm_utils import get_model_name_from_path


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
        self.tokenizer, self.model, self.image_processor, context_len = load_pretrained_model(model_path, model_name=self.model_name, model_base=None)
        # import torch
        # self.tokenizer, self.model, self.image_processor, context_len = load_pretrained_model(model_path, model_name=self.model_name, model_base=None, 
        # load_4bit=True, torch_dtype=torch.float16)

    def predict(self, text_prompt, pil_image, max_tokens=512):
        images = [pil_image]
        model = self.model
        model_name = self.model_name
        image_processor = self.image_processor
        tokenizer = self.tokenizer


        qs = text_prompt

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

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        image = pil_image
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

        pred = model.generate(
            input_ids=input_ids.cuda(),
            images=image_tensor.unsqueeze(0).half().cuda(),
            do_sample=False,
            num_beams=1,
            max_new_tokens=max_tokens,
            num_return_sequences=1,
            use_cache=True
        )

        answer = tokenizer.batch_decode(pred, skip_special_tokens=True)[0]
        answer = answer.strip()

        return answer

# MODEL_NAME = "VILA1.5-3b"
conv_mode = "vicuna_v1"
# model_path = 'checkpoints/v1_2_3b_lr_1e-5_bs80_ep_100_tune_all'
model_path = 'checkpoints/v1_2_13b_lr_3e-5_bs80_ep_100_tune_vision'
# model_path = 'Efficient-Large-Model/VILA1.5-3b'
model = LargeModel(model_path, conv_mode)

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

    # assert payload["model"] == MODEL_NAME
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
        max_tokens=payload['max_tokens'],
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