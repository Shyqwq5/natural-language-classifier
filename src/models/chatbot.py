from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from copy import deepcopy
class Chatbot:
    def __init__(self,model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", dtype=torch.bfloat16)
        self.history_message = [{"role": "system", "content": "",}]
        self.chat_history_ids = None
        self.arguments = {'do_sample':False,'max_new_tokens':32}

    def set_arguments(self,arguments):
        self.arguments = arguments

    def modify_system_prompt(self,prompt):
        self.history_message[0]["content"] = prompt

    def encode_prompt(self,prompt):
        return self.tokenizer(prompt,return_tensors="pt")

    def decode_reply(self,reply_ids):
        return self.tokenizer.decode(reply_ids,skip_special_tokens=True)

    def generate_reply(self,prompt):
        messages = deepcopy(self.history_message)
        messages.append({"role": "user", "content": prompt})

        tokenized_chat = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")

        device = self.model.get_input_embeddings().weight.device
        tokenized_chat = tokenized_chat.to(device)

        outputs = self.model.generate(input_ids=tokenized_chat,**self.arguments)
        new_tokens = outputs[0, tokenized_chat.shape[1]:]
        decoded_outputs = self.decode_reply(new_tokens)
        return decoded_outputs


