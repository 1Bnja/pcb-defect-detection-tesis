import json
import torch
from unsloth import FastVisionModel
from transformers import AutoProcessor
from PIL import Image
from sklearn.metrics import classification_report
from collections import defaultdict
import random

# ── Defectos posibles ──────────────────────────────────────────────────────────
DEFECT_CLASSES = [
    "open circuit",
    "short circuit",
    "mouse bite",
    "spurious copper",
    "missing hole",
    "spur",
]

# ── Cargar modelo ──────────────────────────────────────────────────────────────
model, tokenizer = FastVisionModel.from_pretrained(
    "microsoft/Phi-3.5-vision-instruct",
    load_in_4bit=True,
    trust_remote_code=True,
    attn_implementation="eager",
)
from peft import PeftModel
model = PeftModel.from_pretrained(model, "models/phi35-vision-pcb")

processor = AutoProcessor.from_pretrained(
    "microsoft/Phi-3.5-vision-instruct",
    trust_remote_code=True,
    num_crops=4,
)
FastVisionModel.for_inference(model)

# ── Split 80/20 ────────────────────────────────────────────────────────────────
with open("data/processed/dataset.jsonl") as f:
    samples = [json.loads(l) for l in f]

random.seed(42)
random.shuffle(samples)
split = int(len(samples) * 0.8)
test_samples = samples[split:]  # 300 ejemplos
print(f"Evaluando sobre {len(test_samples)} ejemplos de test...")

# ── Inferencia y métricas ──────────────────────────────────────────────────────
def predict(sample):
    image = Image.open(sample["image"]).convert("RGB")
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
    text = processor.tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    return text.split(".")[0].strip().lower()

def parse_defects(text):
    return set(d for d in DEFECT_CLASSES if d in text.lower())

def parse_gt(sample):
    gpt_msg = sample["conversations"][1]["value"].lower()
    return set(d for d in DEFECT_CLASSES if d in gpt_msg)

# ── Loop de evaluación ─────────────────────────────────────────────────────────
y_true = defaultdict(list)
y_pred = defaultdict(list)

for i, sample in enumerate(test_samples):
    pred_text = predict(sample)
    pred_defects = parse_defects(pred_text)
    gt_defects = parse_gt(sample)

    for cls in DEFECT_CLASSES:
        y_true[cls].append(1 if cls in gt_defects else 0)
        y_pred[cls].append(1 if cls in pred_defects else 0)

    if (i + 1) % 10 == 0:
        print(f"  [{i+1}/{len(test_samples)}]")

# ── Reporte ────────────────────────────────────────────────────────────────────
print("\n=== RESULTADOS POR DEFECTO ===")
all_true, all_pred = [], []
for cls in DEFECT_CLASSES:
    all_true.extend(y_true[cls])
    all_pred.extend(y_pred[cls])
    tp = sum(t == 1 and p == 1 for t, p in zip(y_true[cls], y_pred[cls]))
    fp = sum(t == 0 and p == 1 for t, p in zip(y_true[cls], y_pred[cls]))
    fn = sum(t == 1 and p == 0 for t, p in zip(y_true[cls], y_pred[cls]))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    print(f"  {cls:<20} P={precision:.2f}  R={recall:.2f}  F1={f1:.2f}")

tp = sum(t == 1 and p == 1 for t, p in zip(all_true, all_pred))
fp = sum(t == 0 and p == 1 for t, p in zip(all_true, all_pred))
fn = sum(t == 1 and p == 0 for t, p in zip(all_true, all_pred))
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
print(f"\n{'MACRO PROMEDIO':<20} P={precision:.2f}  R={recall:.2f}  F1={f1:.2f}")
