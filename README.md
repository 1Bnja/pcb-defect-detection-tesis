# PCB Defect Detection — Fine-tuning Phi-3.5-vision

Estudio de viabilidad para detección automática de defectos en placas de circuito impreso (PCB) mediante fine-tuning de un modelo de visión multimodal con LoRA.

> **Contexto:** Primera instancia experimental de una tesis de ingeniería civil en computación (Universidad de Talca). El objetivo es evaluar si un VLM (Vision Language Model) puede aprender a identificar defectos en PCBs a partir de un dataset etiquetado, usando fine-tuning eficiente con LoRA en hardware de consumo.

---

## Dataset

[DeepPCB](https://github.com/tangsanli5201/DeepPCB) — 1.500 imágenes de PCBs con 6 tipos de defectos etiquetados:

- open circuit
- short circuit
- mouse bite
- spurious copper
- missing hole
- spur

**Split reproducible:** 70 / 10 / 20 (train / val / test) con seed=42.

---

## Modelo

- **Base:** `microsoft/Phi-3.5-vision-instruct`
- **Método:** LoRA (PEFT) vía Unsloth
- **Cuantización:** 4-bit (bitsandbytes)
- **Hardware:** RTX 4070 SUPER (12 GB VRAM)

---

## Resultados

### Resumen ejecutivo

| | Baseline | Exp 01 | Exp 02 | Exp 03 |
|---|---|---|---|---|
| Split | — | ⚠️ Incorrecto (leakage) | ✅ 70/10/20 | ✅ 70/10/20 |
| Métrica macro | — | ⚠️ Micro (incorrecta) | ✅ Macro correcta | ✅ Macro correcta |
| LoRA modules | — | Atención | Atención | Atención + MLP |
| Learning rate | — | 2e-4 | 2e-4 | 1e-4 |
| Epochs | — | 3 | 3 | 5 (early stopping) |
| Macro F1 | 0.00 | ~0.94* | **0.93** | **0.93** |
| Macro Recall | 0.00 | ~0.98* | **0.99** | 0.97 |
| Macro Precisión | 0.00 | ~0.89* | 0.87 | **0.89** |

_*Exp 01 reportado con metodología incorrecta; valores indicativos._

### Experimento 02 — Detalle (modelo de referencia)

| Defecto | Precisión | Recall | F1 |
|---|---|---|---|
| open circuit | 0.96 | 0.96 | 0.96 |
| short circuit | 0.80 | 0.93 | 0.85 |
| mouse bite | 0.85 | 1.00 | 0.92 |
| spurious copper | 0.87 | 0.96 | 0.91 |
| missing hole | 0.88 | 1.00 | 0.94 |
| spur | 0.87 | 1.00 | 0.98 |
| **Macro promedio** | **0.87** | **0.99** | **0.93** |

### Trade-off Precisión / Recall

**Exp 02** es el modelo preferido para inspección industrial: tiene recall 0.99 (casi nunca omite un defecto real), a costa de más falsas alarmas (FP).

**Exp 03** es más conservador: menos FP pero más FN. Adecuado como segunda capa en un pipeline de doble revisión donde Exp 02 filtra y Exp 03 confirma.

Ver detalles completos en [`results/experiment_03.md`](results/experiment_03.md).

---

## Instalación

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch==2.6.0+cu124 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt --no-deps
```

> ⚠️ La instalación requiere atención especial por compatibilidad entre versiones de `transformers`, `unsloth` y `bitsandbytes`. Usar `--no-deps` en el paso de requirements.

---

## Uso

Todos los scripts se ejecutan desde la raíz del repositorio con el entorno virtual activo.

### 1. Preparar datos

```bash
# Procesar imágenes crudas de DeepPCB
python3 src/scripts/process_deeppcb.py

# Generar split 70/10/20 reproducible
python3 src/scripts/split_dataset.py
```

### 2. Baseline (sin fine-tuning)

```bash
python3 src/scripts/baseline.py
# → results/baseline_metrics.json
# → results/baseline_predictions.jsonl
```

### 3. Experimento 02 — LoRA en capas de atención

```bash
# Entrenamiento (~52 min en RTX 4070 SUPER)
python3 src/experiments/exp02/train.py

# Evaluación
python3 src/experiments/exp02/evaluate.py
# → results/exp02/metrics.json
# → results/exp02/predictions.jsonl
```

### 4. Experimento 03 — LoRA ampliado (atención + MLP)

```bash
# Entrenamiento (~74 min en RTX 4070 SUPER)
python3 src/experiments/exp03/train.py

# Evaluación
python3 src/experiments/exp03/evaluate.py
# → results/exp03/metrics.json
# → results/exp03/predictions.jsonl
```

### 5. Comparar resultados

```bash
python3 src/scripts/compare_results.py
# → results/exp02/report.md
```

### Inferencia sobre imagen individual

```bash
python3 src/inference.py
```

---

## Estructura

```
tesis-vision-circuits/
├── src/
│   ├── experiments/
│   │   ├── exp02/
│   │   │   ├── train.py       # LoRA atención, lr=2e-4, 3 epochs
│   │   │   └── evaluate.py    # Evaluación sobre test split
│   │   └── exp03/
│   │       ├── train.py       # LoRA atención+MLP, lr=1e-4, 5 epochs
│   │       └── evaluate.py    # Evaluación sobre test split
│   ├── scripts/
│   │   ├── process_deeppcb.py # Preprocesamiento del dataset
│   │   ├── split_dataset.py   # Genera splits 70/10/20 (seed=42)
│   │   ├── baseline.py        # Evalúa modelo base sin fine-tuning
│   │   └── compare_results.py # Genera report Markdown comparativo
│   └── inference.py           # Inferencia sobre imagen individual
├── results/
│   ├── baseline/
│   │   ├── metrics.json
│   │   └── predictions.jsonl
│   ├── exp01/
│   │   └── report.md          # Reporte indicativo (metodología incorrecta)
│   ├── exp02/
│   │   ├── metrics.json
│   │   ├── predictions.jsonl
│   │   └── report.md
│   └── exp03/
│       ├── metrics.json
│       ├── predictions.jsonl
│       └── report.md
├── data/                      # No incluido en el repo (DeepPCB)
│   └── processed/splits/      # train.jsonl · val.jsonl · test.jsonl
├── models/                    # No incluido en el repo (pesos LoRA)
│   ├── phi35-vision-pcb-exp02/
│   └── phi35-vision-pcb-exp03/
├── requirements.txt
└── README.md
```

---

## Próximos pasos

- [ ] Prompt estructurado (salida JSON) para mejorar robustez del parser
- [ ] Aumentación de datos para reducir FP en `short circuit` y `mouse bite`
- [ ] Pipeline de doble revisión: Exp 02 (filtrado inicial) + Exp 03 (confirmación)
- [ ] Fine-tuning sobre imágenes de PCBs reales del entorno de producción objetivo

---

## Estado

🟢 Tres experimentos completados con metodología correcta (split limpio, macro-average real, baseline documentado). Macro F1 = **0.93** sobre test set.
