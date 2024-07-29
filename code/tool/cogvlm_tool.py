from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
from transformers.generation import GenerationConfig
import torch
from PIL import Image

class CogvlmTool():
    def __init__(self,model_name_or_path,device_map="auto"):
        print(f"Loading model: {model_name_or_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        ).eval()
        self.tokenizer = LlamaTokenizer.from_pretrained('lmsys/vicuna-7b-v1.5')

    def prepare_input(self,img_path,prompt):
        query = f"Only answer what is asked. {prompt}"
        image = Image.open(img_path).convert("RGB")
        inputs = self.model.build_conversation_input_ids(self.tokenizer, query=query, history=[], images=[image])  # chat mode
        inputs = {
            'input_ids': inputs['input_ids'].unsqueeze(0).to('cuda'),
            'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to('cuda'),
            'attention_mask': inputs['attention_mask'].unsqueeze(0).to('cuda'),
            'images': [[inputs['images'][0].to('cuda').to(torch.bfloat16)]],
        }
        return inputs

    def generate(self, img_path,prompt, max_new_tokens, do_sample):
        inputs = self.prepare_input(img_path,prompt)

        gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": do_sample}
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            response = self.tokenizer.decode(outputs[0],skip_special_tokens=True)
            torch.cuda.empty_cache()

        return response