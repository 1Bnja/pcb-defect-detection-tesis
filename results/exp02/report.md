# Experimento 02 — Fine-tuning Phi-3.5-vision en DeepPCB (Split Corregido)

_Generado: 2026-05-02 23:59_

## Configuración

| Parámetro | Valor |
|---|---|
| Modelo base | microsoft/Phi-3.5-vision-instruct |
| Método | LoRA (PEFT) vía Unsloth |
| LoRA rank | 16 |
| LoRA alpha | 16 |
| Target modules | q_proj, k_proj, v_proj, o_proj |
| Epochs | 3 |
| Learning rate | 2e-4 |
| Batch size efectivo | 8 (bs=1, grad_accum=8) |
| Cuantización | 4-bit |
| Dataset | DeepPCB |
| Train/Val/Test | 1050/150/300 (70/10/20) |
| Split seed | 42 (reproducible) |
| eval_strategy | epoch |
| Tiempo entrenamiento | ~52 min |

## Resultados — Baseline (sin fine-tuning)

| Defecto | Precisión | Recall | F1 |
|---|---|---|---|
| open circuit | 0.00 | 0.00 | 0.00 |
| short circuit | 0.00 | 0.00 | 0.00 |
| mouse bite | 0.00 | 0.00 | 0.00 |
| spurious copper | 0.00 | 0.00 | 0.00 |
| missing hole | 0.00 | 0.00 | 0.00 |
| spur | 0.00 | 0.00 | 0.00 |
| **Macro promedio** | **0.00** | **0.00** | **0.00** |

## Resultados — Fine-tuned Exp 02

| Defecto | Precisión | Recall | F1 | Δ F1 vs Baseline |
|---|---|---|---|---|
| open circuit | 0.93 | 0.99 | 0.96 | +0.96 |
| short circuit | 0.76 | 0.96 | 0.85 | +0.85 |
| mouse bite | 0.86 | 1.00 | 0.92 | +0.92 |
| spurious copper | 0.84 | 0.99 | 0.91 | +0.91 |
| missing hole | 0.89 | 1.00 | 0.94 | +0.94 |
| spur | 0.97 | 1.00 | 0.98 | +0.98 |
| **Macro promedio** | **0.87** | **0.99** | **0.93** | **+0.93** |

## Comparación resumen

| Métrica | Baseline | Fine-tuned | Mejora |
|---|---|---|---|
| Macro Precision | 0.00 | 0.87 | +0.87 |
| Macro Recall | 0.00 | 0.99 | +0.99 |
| Macro F1 | 0.00 | 0.93 | +0.93 |

## Observaciones

- Split 70/10/20 con seed=42 garantiza reproducibilidad y elimina el data leakage del Exp 01.
- La métrica macro-average es ahora el promedio aritmético de F1 por clase (corrección respecto al Exp 01).
- La evaluación sobre validación epoch a epoch permite detectar posible overfitting.

## Próximos pasos

- [x] Ampliar LoRA a capas MLP (gate_proj, up_proj, down_proj) → Exp 03
- [ ] Analizar falsos positivos/negativos cualitativamente (imágenes)
- [ ] Experimentar con prompt estructurado (salida JSON)
- [ ] Aumentación de datos para mejorar robustez
