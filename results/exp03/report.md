# Experimento 03 — LoRA Ampliado (Atención + MLP)

_Generado: 2026-05-03_

## Configuración

| Parámetro | Exp 02 | Exp 03 |
|---|---|---|
| Modelo base | Phi-3.5-vision-instruct | Phi-3.5-vision-instruct |
| LoRA rank | 16 | 16 |
| LoRA alpha | 16 | 16 |
| Target modules | q/k/v/o_proj | q/k/v/o_proj + **gate/up/down_proj** |
| Learning rate | 2e-4 | **1e-4** |
| Epochs | 3 | **5 (early stopping patience=2)** |
| Batch size efectivo | 8 | 8 |
| Cuantización | 4-bit | 4-bit |
| Train / Val / Test | 1050 / 150 / 300 | 1050 / 150 / 300 |
| Split seed | 42 | 42 |
| Tiempo entrenamiento | 52 min | 74 min |
| Train loss final | 0.2499 | **0.1940** |
| Val loss final | 0.0907 | **0.0887** |

> El early stopping no se activó — la val_loss mejoró en los 5 epochs, indicando que el modelo seguía aprendiendo.

---

## Resultados — Experimento 03

| Defecto | Precisión | Recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|
| open circuit | 0.96 | 0.96 | 0.96 | 258 | 10 | 12 |
| short circuit | 0.80 | 0.91 | 0.85 | 181 | 44 | 19 |
| mouse bite | 0.86 | 1.00 | 0.92 | 258 | 42 | 0 |
| spurious copper | 0.85 | 0.96 | 0.90 | 241 | 43 | 10 |
| missing hole | 0.88 | 1.00 | 0.93 | 261 | 36 | 1 |
| spur | 0.97 | 1.00 | 0.98 | 290 | 10 | 0 |
| **Macro promedio** | **0.89** | **0.97** | **0.93** | | | |

---

## Comparación completa — Baseline vs Exp 02 vs Exp 03

| Defecto | Baseline F1 | Exp 02 F1 | Exp 03 F1 | Δ Exp02→Exp03 |
|---|---|---|---|---|
| open circuit | 0.00 | 0.96 | 0.96 | 0.00 |
| short circuit | 0.00 | 0.85 | 0.85 | 0.00 |
| mouse bite | 0.00 | 0.92 | 0.92 | 0.00 |
| spurious copper | 0.00 | 0.91 | 0.90 | −0.01 |
| missing hole | 0.00 | 0.94 | 0.93 | −0.01 |
| spur | 0.00 | 0.98 | 0.98 | 0.00 |
| **Macro F1** | **0.00** | **0.93** | **0.93** | **0.00** |

| Métrica macro | Baseline | Exp 02 | Exp 03 |
|---|---|---|---|
| Precisión | 0.00 | 0.87 | **0.89** |
| Recall | 0.00 | **0.99** | 0.97 |
| F1 | 0.00 | 0.93 | 0.93 |

---

## Análisis del trade-off Precisión / Recall

Agregar las capas MLP no aumentó el F1 macro, pero produjo un cambio cualitativo en el comportamiento del modelo:

**Exp 02 (solo atención)** — perfil de alta sensibilidad:
- Macro recall 0.99: casi nunca omite un defecto presente
- Macro precisión 0.87: genera más falsas alarmas
- Total FP: 215 · Total FN: 14

**Exp 03 (atención + MLP)** — perfil más conservador:
- Macro recall 0.97: omite algo más casos reales
- Macro precisión 0.89: genera menos falsas alarmas
- Total FP: 185 · Total FN: 42

Este trade-off tiene implicaciones prácticas directas. En inspección industrial de PCBs, **omitir un defecto real (FN) suele ser más costoso** que generar una falsa alarma (FP), ya que una PCB defectuosa que pasa desapercibida puede causar fallos en producción. Bajo este criterio, **Exp 02 es el modelo preferido** para un sistema de primera inspección.

Exp 03 sería más adecuado en un pipeline de doble revisión donde Exp 02 hace el filtrado inicial y Exp 03 confirma los casos detectados, reduciendo el volumen de falsas alarmas antes de la revisión humana.

---

## Resumen ejecutivo de los tres experimentos

| | Baseline | Exp 01 | Exp 02 | Exp 03 |
|---|---|---|---|---|
| Split | — | ⚠️ Incorrecto (leakage) | ✅ 70/10/20 | ✅ 70/10/20 |
| Métrica macro | — | ⚠️ Incorrecta (micro) | ✅ Correcta | ✅ Correcta |
| LoRA modules | — | Atención | Atención | Atención + MLP |
| Learning rate | — | 2e-4 | 2e-4 | 1e-4 |
| Epochs | — | 3 | 3 | 5 |
| Macro F1 | 0.00 | ~0.94* | **0.93** | **0.93** |
| Macro Recall | 0.00 | ~0.98* | **0.99** | 0.97 |
| Macro Precisión | 0.00 | ~0.89* | 0.87 | **0.89** |

_*Exp 01 reportado con metodología incorrecta; valores indicativos._

---

## Próximos pasos

- [ ] Prompt estructurado (salida JSON) para mejorar robustez del parser
- [ ] Aumentación de datos para reducir FP en short circuit y mouse bite
- [ ] Pipeline de doble revisión: Exp 02 (filtrado) + Exp 03 (confirmación)
- [ ] Fine-tuning sobre imágenes de PCBs reales del entorno de producción objetivo
