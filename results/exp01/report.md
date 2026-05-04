# Experimento 01 — Fine-tuning Phi-3.5-vision en DeepPCB

## Configuración

| Parámetro | Valor |
|---|---|
| Modelo base | microsoft/Phi-3.5-vision-instruct |
| Método | LoRA (PEFT) |
| LoRA rank | 16 |
| LoRA alpha | 16 |
| Target modules | q_proj, k_proj, v_proj, o_proj |
| Epochs | 3 |
| Learning rate | 2e-4 |
| Batch size | 1 |
| Gradient accumulation | 8 |
| Cuantización | 4-bit |
| Dataset | DeepPCB (1500 ejemplos) |
| Train/Test split | 80/20 (1200/300) |
| GPU | NVIDIA GeForce RTX 4070 SUPER (12 GB) |
| Tiempo de entrenamiento | ~61 minutos |

## Resultados

| Defecto | Precisión | Recall | F1 |
|---|---|---|---|
| open circuit | 0.96 | 0.98 | 0.97 |
| short circuit | 0.82 | 0.91 | 0.86 |
| mouse bite | 0.86 | 1.00 | 0.92 |
| spurious copper | 0.84 | 1.00 | 0.91 |
| missing hole | 0.88 | 1.00 | 0.94 |
| spur | 0.97 | 1.00 | 0.98 |
| **Macro promedio** | **0.89** | **0.98** | **0.94** |

## Observaciones

- El modelo alcanza un F1 macro de **0.94** en el set de test.
- Recall macro de **0.98**: el modelo raramente omite defectos presentes.
- Precisión macro de **0.89**: el modelo ocasionalmente reporta defectos adicionales no etiquetados (falsos positivos), en particular `spur`.
- `short circuit` es la clase con menor F1 (0.86), posible candidato a mejora con más datos o ajuste de prompt.
- El defecto `spur` tiene F1 perfecto (0.98) pero no siempre estaba etiquetado en el ground truth, lo que puede subestimar la precisión real del modelo.

## ⚠️ Notas metodológicas

- Split incorrecto: 80/20 sin separar validación, posible data leakage.
- Macro-average calculado como micro (concatenación de vectores binarios) en lugar de promedio aritmético por clase.
- Los valores de F1 son **indicativos** y no reproducibles con la metodología corregida de Exp 02+.
