from unsloth import FastVisionModel
import torch
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from torch.utils.data import Dataset
from PIL import Image
import json

# 1. Cargar Modelo y Procesador
model, tokenizer = FastVisionModel.from_pretrained(
    "microsoft/Phi-3.5-vision-instruct",
    load_in_4bit = True,
    use_gradient_checkpointing = "unsloth",
    trust_remote_code = True,
    attn_implementation = "eager",
)

from transformers import AutoProcessor
processor = AutoProcessor.from_pretrained(
    "microsoft/Phi-3.5-vision-instruct",
    trust_remote_code=True,
    num_crops=4,
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

# 3. Dataset custom con imágenes
class PCBDataset(Dataset):
    def __init__(self, jsonl_path, processor):
        self.processor = processor
        self.samples = []
        with open(jsonl_path) as f:
            for line in f:
                self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["image"]).convert("RGB")
        human_msg = sample["conversations"][0]["value"].replace("<image>\n", "")
        gpt_msg = sample["conversations"][1]["value"]

        messages = [
            {"role": "user", "content": f"<|image_1|>\n{human_msg}"},
            {"role": "assistant", "content": gpt_msg},
        ]
        prompt = processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        inputs = processor(prompt, [image], return_tensors="pt")
        input_ids = inputs["input_ids"].squeeze(0)
        labels = input_ids.clone()

        # Enmascarar la parte del usuario, solo entrenar en respuesta
        assistant_token = processor.tokenizer.encode("<|assistant|>", add_special_tokens=False)
        for i in range(len(labels) - len(assistant_token)):
            if labels[i:i+len(assistant_token)].tolist() == assistant_token:
                labels[:i+len(assistant_token)] = -100
                break

        return {
            "input_ids": input_ids,
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "image_sizes": inputs["image_sizes"].squeeze(0),
            "labels": labels,
        }

# 4. Collator custom
def collate_fn(batch):
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [x["input_ids"] for x in batch], batch_first=True, padding_value=tokenizer.pad_token_id
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        [x["attention_mask"] for x in batch], batch_first=True, padding_value=0
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        [x["labels"] for x in batch], batch_first=True, padding_value=-100
    )
    pixel_values = torch.stack([x["pixel_values"] for x in batch])
    image_sizes = torch.stack([x["image_sizes"] for x in batch])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "image_sizes": image_sizes,
        "labels": labels,
    }

# 5. Instanciar dataset
dataset = PCBDataset("data/processed/dataset.jsonl", processor)

# 6. Entrenamiento
FastVisionModel.for_training(model)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    data_collator = collate_fn,
    max_seq_length = 2048,
    dataset_text_field = None,
    args = TrainingArguments(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 8,
        warmup_steps = 5,
        max_steps = 60,
        learning_rate = 2e-4,
        bf16 = True,
        logging_steps = 1,
        output_dir = "models/phi35-vision-pcb",
        remove_unused_columns = False,
        dataloader_pin_memory = False,
    ),
)

trainer.train()

model.save_pretrained("models/phi35-vision-pcb")
tokenizer.save_pretrained("models/phi35-vision-pcb")
print("✅ Modelo guardado")