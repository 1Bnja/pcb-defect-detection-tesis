# PCB Defect Detection — Fine-tuning Phi-3.5-vision

Estudio de viabilidad para detección automática de defectos en placas de circuito impreso (PCB) mediante fine-tuning de un modelo de visión multimodal con LoRA.

## Contexto

Este repositorio corresponde a la primera instancia experimental de una tesis de ingeniería civil en computación (Universidad de Talca). El objetivo es evaluar si un VLM (Vision Language Model) puede aprender a identificar defectos en PCBs a partir de un dataset etiquetado, usando fine-tuning eficiente con LoRA en hardware de consumo.

## Dataset

[DeepPCB](https://github.com/tangsanli5201/DeepPCB) — 1500 imágenes de PCBs con 6 tipos de defectos etiquetados:

- open circuit
- short circuit
- mouse bite
- spurious copper
- missing hole
- spur

## Modelo

- **Base:** `microsoft/Phi-3.5-vision-instruct`
- **Método:** LoRA (PEFT) vía Unsloth
- **Cuantización:** 4-bit

## Resultados (Experimento 01)

| Defecto | Precisión | Recall | F1 |
|---|---|---|---|
| open circuit | 0.96 | 0.98 | 0.97 |
| short circuit | 0.82 | 0.91 | 0.86 |
| mouse bite | 0.86 | 1.00 | 0.92 |
| spurious copper | 0.84 | 1.00 | 0.91 |
| missing hole | 0.88 | 1.00 | 0.94 |
| spur | 0.97 | 1.00 | 0.98 |
| **Macro promedio** | **0.89** | **0.98** | **0.94** |

Entrenamiento de ~61 minutos en RTX 4070 SUPER (12 GB VRAM).

## Instalación

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch==2.6.0+cu124 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt --no-deps
```

> ⚠️ La instalación de dependencias requiere atención especial por compatibilidad entre versiones. Ver historial de setup para detalles.

## Uso

**Entrenamiento:**
```bash
python3 src/train.py
```

**Inferencia:**
```bash
python3 src/inference.py
```

**Evaluación:**
```bash
python3 src/evaluate.py
```

## Estructura
```
pcb-defect-detection/
├── src/
│   ├── train.py        # Fine-tuning con LoRA
│   ├── inference.py    # Inferencia sobre imagen individual
│   └── evaluate.py     # Evaluación sobre set de test
├── results/
│   └── experiment_01.md
├── data/               # No incluido en el repo (DeepPCB)
├── models/             # No incluido en el repo (pesos LoRA)
├── requirements.txt
└── README.md
```
## Estado

🟡 Primera instancia — estudio de viabilidad completado con resultados positivos (F1 macro 0.94).
