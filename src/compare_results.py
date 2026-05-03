"""
compare_results.py
------------------
Genera results/experiment_02.md comparando baseline vs fine-tuned.
Corre después de baseline.py y evaluate.py.

Uso:
    python3 src/compare_results.py
"""

import json
import os
from datetime import datetime

DEFECT_CLASSES = [
    "open circuit",
    "short circuit",
    "mouse bite",
    "spurious copper",
    "missing hole",
    "spur",
]

def load(path):
    with open(path) as f:
        return json.load(f)

baseline = load("results/baseline_metrics.json")
exp02    = load("results/exp02_metrics.json")

lines = []
lines.append("# Experimento 02 — Fine-tuning Phi-3.5-vision en DeepPCB (Split Corregido)\n")
lines.append(f"_Generado: {datetime.now().strftime('%Y-%m-%d %H:%M')}_\n")

lines.append("## Configuración\n")
lines.append("| Parámetro | Valor |")
lines.append("|---|---|")
lines.append("| Modelo base | microsoft/Phi-3.5-vision-instruct |")
lines.append("| Método | LoRA (PEFT) vía Unsloth |")
lines.append("| LoRA rank | 16 |")
lines.append("| LoRA alpha | 16 |")
lines.append("| Target modules | q_proj, k_proj, v_proj, o_proj |")
lines.append("| Epochs | 3 |")
lines.append("| Learning rate | 2e-4 |")
lines.append("| Batch size efectivo | 8 (bs=1, grad_accum=8) |")
lines.append("| Cuantización | 4-bit |")
lines.append("| Dataset | DeepPCB |")
lines.append(f"| Train/Val/Test | {exp02.get('train_n','?')}/{exp02.get('val_n','?')}/{exp02['test_samples']} (70/10/20) |")
lines.append("| Split seed | 42 (reproducible) |")
lines.append("| eval_strategy | epoch |")
lines.append("")

lines.append("## Resultados — Baseline (sin fine-tuning)\n")
lines.append("| Defecto | Precisión | Recall | F1 |")
lines.append("|---|---|---|---|")
for cls in DEFECT_CLASSES:
    m = baseline["per_class"][cls]
    lines.append(f"| {cls} | {m['precision']:.2f} | {m['recall']:.2f} | {m['f1']:.2f} |")
bm = baseline["macro"]
lines.append(f"| **Macro promedio** | **{bm['precision']:.2f}** | **{bm['recall']:.2f}** | **{bm['f1']:.2f}** |")
lines.append("")

lines.append("## Resultados — Fine-tuned Exp 02\n")
lines.append("| Defecto | Precisión | Recall | F1 | Δ F1 vs Baseline |")
lines.append("|---|---|---|---|---|")
for cls in DEFECT_CLASSES:
    m  = exp02["per_class"][cls]
    bm_cls = baseline["per_class"][cls]
    delta = m["f1"] - bm_cls["f1"]
    sign  = "+" if delta >= 0 else ""
    lines.append(f"| {cls} | {m['precision']:.2f} | {m['recall']:.2f} | {m['f1']:.2f} | {sign}{delta:.2f} |")
em = exp02["macro"]
bm = baseline["macro"]
delta_macro = em["f1"] - bm["f1"]
sign = "+" if delta_macro >= 0 else ""
lines.append(f"| **Macro promedio** | **{em['precision']:.2f}** | **{em['recall']:.2f}** | **{em['f1']:.2f}** | **{sign}{delta_macro:.2f}** |")
lines.append("")

lines.append("## Comparación resumen\n")
lines.append("| Métrica | Baseline | Fine-tuned | Mejora |")
lines.append("|---|---|---|---|")
for metric in ["precision", "recall", "f1"]:
    b_val = baseline["macro"][metric]
    e_val = exp02["macro"][metric]
    delta = e_val - b_val
    sign  = "+" if delta >= 0 else ""
    lines.append(f"| Macro {metric.capitalize()} | {b_val:.2f} | {e_val:.2f} | {sign}{delta:.2f} |")
lines.append("")

lines.append("## Observaciones\n")
lines.append("- Split 70/10/20 con seed=42 garantiza reproducibilidad y elimina el data leakage del Exp 01.")
lines.append("- La métrica macro-average es ahora el promedio aritmético de F1 por clase (corrección respecto al Exp 01).")
lines.append("- La evaluación sobre validación epoch a epoch permite detectar posible overfitting.")
lines.append("")

lines.append("## Próximos pasos\n")
lines.append("- [ ] Analizar falsos positivos/negativos cualitativamente (imágenes)")
lines.append("- [ ] Ampliar LoRA a capas MLP (gate_proj, up_proj, down_proj)")
lines.append("- [ ] Experimentar con prompt estructurado (salida JSON)")
lines.append("- [ ] Aumentación de datos para mejorar robustez")

output = "\n".join(lines) + "\n"

os.makedirs("results", exist_ok=True)
with open("results/experiment_02.md", "w") as f:
    f.write(output)

print("✅ results/experiment_02.md generado.")
print(output)
