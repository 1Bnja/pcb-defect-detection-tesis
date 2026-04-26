from unsloth import FastVisionModel
import torch
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer

# 1. Cargar Modelo y Tokenizer
model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Phi-3.5-vision-instruct",
    load_in_4bit = True,
    use_gradient_checkpointing = "unsloth",
)

# 2. Configurar PEFT (LoRA)
model = FastVisionModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
)

# 3. Cargar Dataset
dataset = load_dataset("json", data_files="data/processed/dataset.jsonl", split="train")

# 4. Configuración de Entrenamiento
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "conversations", # Unsloth maneja el formato de vision automáticamente
    max_seq_length = 2048,
    dataset_num_proc = 4,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60, # Prueba inicial corta, luego subimos a 500+
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        output_dir = "models/phi35-vision-pcb",
    ),
)

trainer.train()
