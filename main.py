# https://medium.com/@thakermadhav/build-your-own-rag-with-mistral-7b-and-langchain-97d0c92fa146
import torch

# get available gpu devices
devices = torch.cuda.device_count()


import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

use_4bit = True
compute_dtype = getattr(torch, "float16")

# specify how to quantize the model
bnb_config = BitsAndBytesConfig(
        load_in_4bit = True, # Activate 4-bit precision base model loading
        bnb_4bit_quant_type = "nf4", # Quantization type (fp4 or nf4)
        bnb_4bit_compute_dtype = getattr(torch, "float16"), # Compute dtype for 4-bit base models
        bnb_4bit_use_double_quant = False # Activate nested quantization for 4-bit base models (double quantization)
)

# Check GPU compatibility with bfloat16
if compute_dtype == torch.float16 and use_4bit:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16: accelerate training with bf16=True")
        print("=" * 80)

model_name_full="mistralai/Mistral-7B-Instruct-v0.2"
model_name="Mistral-7B-Instruct-v0.2"

# model_config = transformers.AutoConfig.from_pretrained(model_name, trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, attn_implementation="flash_attention_2", device_map="auto")
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config)
# save the model for future use
# model.save_pretrained("Mistral-7B-Instruct-v0.2")
tokenizer = AutoTokenizer.from_pretrained(model_name_full)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

message = "Pacituok lietuvių liaudies eilėraštį 'Du gaideliai'"

inputs_not_chat = tokenizer.encode_plus(f"[INST] {message} [/INST]", return_tensors="pt")['input_ids'].to('cuda')
generated_ids = model.generate(inputs_not_chat, max_new_tokens=1000, do_sample=True)
decoded = tokenizer.batch_decode(generated_ids)