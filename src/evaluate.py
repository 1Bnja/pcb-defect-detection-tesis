"""
evaluate.py — Experimento 02
------------------------------
Evalúa el modelo fine-tuneado sobre el split de test (20%).
Calcula macro-average correctamente (promedio de métricas por clase).

Cambios respecto a Experimento 01:
  - Carga data/processed/splits/test.jsonl  (sin data leakage)
  - Macro-average real: promedio aritmético de P/R/F1 por clase
  - Guarda predicciones raw en results/exp02_predictions.jsonl
"""

import json
import torch
from unsloth import FastVisionModel
from transformers import AutoProcessor
from PIL import Image
from collections import defaultdict
import os

# Parche de compatibilidad: transformers>=4.51 eliminó get_max_length() de DynamicCache
# pero modeling_phi3_v.py todavía lo llama. Esto lo restaura y habilita use_cache=True.
from transformers.cache_utils import DynamicCache
if not hasattr(DynamicCache, "get_max_length"):
    DynamicCache.get_max_length = lambda self: None

# ── Defectos posibles ──────────────────────────────────────────────────────────
DEFECT_CLASSES = [
    "open circuit",
    "short circuit",
    "mouse bite",
    "spurious copper",
    "missing hole",
    "spur",
]

# ── Cargar modelo fine-tuneado ─────────────────────────────────────────────────
model, tokenizer = FastVisionModel.from_pretrained(
    "microsoft/Phi-3.5-vision-instruct",
    load_in_4bit=True,
    trust_remote_code=True,
    attn_implementation="eager",
)
from peft import PeftModel
model = PeftModel.from_pretrained(model, "models/phi35-vision-pcb-exp02")

processor = AutoProcessor.from_pretrained(
    "microsoft/Phi-3.5-vision-instruct",
    trust_remote_code=True,
    num_crops=4,
)
FastVisionModel.for_inference(model)

# ── Cargar test split ──────────────────────────────────────────────────────────
with open("data/processed/splits/test.jsonl") as f:
    test_samples = [json.loads(l) for l in f]

print(f"Evaluando sobre {len(test_samples)} ejemplos de test (split fijo)...")

# ── Inferencia ─────────────────────────────────────────────────────────────────
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
            use_cache=True,
            eos_token_id=tokenizer.eos_token_id,
        )
    text = processor.tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    return text.strip().lower()

def parse_defects(text):
    return set(d for d in DEFECT_CLASSES if d in text)

def parse_gt(sample):
    gpt_msg = sample["conversations"][1]["value"].lower()
    return set(d for d in DEFECT_CLASSES if d in gpt_msg)

# ── Loop de evaluación ─────────────────────────────────────────────────────────
y_true = defaultdict(list)
y_pred = defaultdict(list)
predictions_log = []

for i, sample in enumerate(test_samples):
    pred_text   = predict(sample)
    pred_defects = parse_defects(pred_text)
    gt_defects   = parse_gt(sample)

    for cls in DEFECT_CLASSES:
        y_true[cls].append(1 if cls in gt_defects  else 0)
        y_pred[cls].append(1 if cls in pred_defects else 0)

    predictions_log.append({
        "id": sample.get("id", i),
        "gt": list(gt_defects),
        "pred": list(pred_defects),
        "pred_text": pred_text,
    })

    if (i + 1) % 10 == 0:
        print(f"  [{i+1}/{len(test_samples)}]")

# ── Métricas por clase ─────────────────────────────────────────────────────────
print("\n=== RESULTADOS POR DEFECTO (Experimento 02) ===")

per_class = {}
for cls in DEFECT_CLASSES:
    tp = sum(t == 1 and p == 1 for t, p in zip(y_true[cls], y_pred[cls]))
    fp = sum(t == 0 and p == 1 for t, p in zip(y_true[cls], y_pred[cls]))
    fn = sum(t == 1 and p == 0 for t, p in zip(y_true[cls], y_pred[cls]))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    per_class[cls] = {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
    print(f"  {cls:<20} P={precision:.2f}  R={recall:.2f}  F1={f1:.2f}  (TP={tp}, FP={fp}, FN={fn})")

# ── Macro-average real (promedio aritmético de métricas por clase) ─────────────
macro_p  = sum(v["precision"] for v in per_class.values()) / len(DEFECT_CLASSES)
macro_r  = sum(v["recall"]    for v in per_class.values()) / len(DEFECT_CLASSES)
macro_f1 = sum(v["f1"]        for v in per_class.values()) / len(DEFECT_CLASSES)

print(f"\n{'MACRO PROMEDIO':<20} P={macro_p:.2f}  R={macro_r:.2f}  F1={macro_f1:.2f}")

# ── Guardar predicciones raw ───────────────────────────────────────────────────
os.makedirs("results", exist_ok=True)
with open("results/exp02_predictions.jsonl", "w") as f:
    for entry in predictions_log:
        f.write(json.dumps(entry) + "\n")
print("\n✅ Predicciones guardadas en results/exp02_predictions.jsonl")

# ── Guardar métricas resumidas ─────────────────────────────────────────────────
summary = {
    "experiment": "02",
    "test_samples": len(test_samples),
    "per_class": per_class,
    "macro": {"precision": macro_p, "recall": macro_r, "f1": macro_f1},
}
with open("results/exp02_metrics.json", "w") as f:
    json.dump(summary, f, indent=2)
print("✅ Métricas guardadas en results/exp02_metrics.json")
