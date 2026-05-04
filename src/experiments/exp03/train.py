"""
train.py — Experimento 03
--------------------------
Cambios respecto a Exp 02:
  - LoRA ampliado a capas MLP (gate_proj, up_proj, down_proj) además de atención
  - Learning rate reducido: 1e-4 (más conservador, mejor convergencia)
  - Epochs aumentados a 5 con early stopping basado en val loss
  - Guarda modelo en models/phi35-vision-pcb-exp03
"""

from unsloth import FastVisionModel
import torch
from transformers import TrainingArguments, AutoProcessor, EarlyStoppingCallback
from trl import SFTTrainer
from torch.utils.data import Dataset
from PIL import Image
import json

# ── 1. Modelo y procesador ─────────────────────────────────────────────────────
model, tokenizer = FastVisionModel.from_pretrained(
    "microsoft/Phi-3.5-vision-instruct",
    load_in_4bit=True,
    use_gradient_checkpointing="unsloth",
    trust_remote_code=True,
    attn_implementation="eager",
)

processor = AutoProcessor.from_pretrained(
    "microsoft/Phi-3.5-vision-instruct",
    trust_remote_code=True,
    num_crops=4,
)

# ── 2. LoRA ampliado (atención + MLP) ─────────────────────────────────────────
model = FastVisionModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",   # Atención
        "gate_proj", "up_proj", "down_proj",        # MLP
    ],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    random_state=3407,
)

# ── 3. Dataset ─────────────────────────────────────────────────────────────────
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
            {"role": "user",      "content": f"<|image_1|>\n{human_msg}"},
            {"role": "assistant", "content": gpt_msg},
        ]
        prompt = processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        inputs = processor(prompt, [image], return_tensors="pt")
        input_ids = inputs["input_ids"].squeeze(0)
        labels = input_ids.clone()

        assistant_token = processor.tokenizer.encode("<|assistant|>", add_special_tokens=False)
        for i in range(len(labels) - len(assistant_token)):
            if labels[i:i+len(assistant_token)].tolist() == assistant_token:
                labels[:i+len(assistant_token)] = -100
                break

        return {
            "input_ids":      input_ids,
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "pixel_values":   inputs["pixel_values"].squeeze(0),
            "image_sizes":    inputs["image_sizes"].squeeze(0),
            "labels":         labels,
        }

# ── 4. Collator ────────────────────────────────────────────────────────────────
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
    image_sizes  = torch.stack([x["image_sizes"]  for x in batch])

    return {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "pixel_values":   pixel_values,
        "image_sizes":    image_sizes,
        "labels":         labels,
    }

# ── 5. Cargar splits ───────────────────────────────────────────────────────────
train_dataset = PCBDataset("data/processed/splits/train.jsonl", processor)
val_dataset   = PCBDataset("data/processed/splits/val.jsonl",   processor)

print(f"Train: {len(train_dataset)} muestras")
print(f"Val  : {len(val_dataset)}   muestras")

# ── 6. Entrenamiento ───────────────────────────────────────────────────────────
FastVisionModel.for_training(model)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collate_fn,
    max_seq_length=2048,
    dataset_text_field=None,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    args=TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        warmup_steps=5,
        num_train_epochs=5,
        learning_rate=1e-4,
        bf16=True,
        logging_steps=1,
        output_dir="models/phi35-vision-pcb-exp03",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    ),
)

trainer.train()

model.save_pretrained("models/phi35-vision-pcb-exp03")
tokenizer.save_pretrained("models/phi35-vision-pcb-exp03")
print("✅ Modelo Experimento 03 guardado en models/phi35-vision-pcb-exp03")
