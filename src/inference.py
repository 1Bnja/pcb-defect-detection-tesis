from unsloth import FastVisionModel
from transformers import AutoProcessor
from PIL import Image
import torch

# 1. Cargar modelo base + adaptador LoRA
model, tokenizer = FastVisionModel.from_pretrained(
    "microsoft/Phi-3.5-vision-instruct",
    load_in_4bit = True,
    trust_remote_code = True,
    attn_implementation = "eager",
)

from peft import PeftModel
model = PeftModel.from_pretrained(model, "models/phi35-vision-pcb")

processor = AutoProcessor.from_pretrained(
    "microsoft/Phi-3.5-vision-instruct",
    trust_remote_code=True,
    num_crops=4,
)

FastVisionModel.for_inference(model)

# 2. Imagen de prueba — cambia por cualquier imagen de tu dataset
image_path = "data/raw/DeepPCB/PCBData/group50600/50600/50600048_test.jpg"
image = Image.open(image_path).convert("RGB")

messages = [
    {"role": "user", "content": "<|image_1|>\nExamina esta placa de circuito e identifica fallas técnicas."},
]

prompt = processor.tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

inputs = processor(prompt, [image], return_tensors="pt").to("cuda")

with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False,
        use_cache=False,
        eos_token_id=tokenizer.eos_token_id,
    )

output_text = processor.tokenizer.decode(
    output_ids[0][inputs["input_ids"].shape[1]:],
    skip_special_tokens=True,
)
# Quedarse solo con la primera oración
output_text = output_text.split(".")[0].strip() + "."
print("=== RESPUESTA DEL MODELO ===")
print(output_text)
