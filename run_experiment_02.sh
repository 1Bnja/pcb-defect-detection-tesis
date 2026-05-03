#!/bin/bash
# ============================================================
# run_experiment_02.sh — Prueba 2 completa
# ============================================================
# Orden de ejecución:
#   1. Generar split reproducible 70/10/20
#   2. Evaluar baseline (modelo sin fine-tuning)
#   3. Fine-tuning con split corregido + monitoreo de val loss
#   4. Evaluar modelo fine-tuneado
#   5. Comparar y generar experiment_02.md
# ============================================================

set -e  # Detener si cualquier comando falla

cd "$(dirname "$0")"
source .venv/bin/activate

echo "============================================"
echo "  EXPERIMENTO 02 — PCB Defect Detection"
echo "============================================"
echo ""

# 1. Split
echo "[1/5] Generando split 70/10/20..."
python3 src/scripts/split_dataset.py
echo ""

# 2. Baseline
echo "[2/5] Evaluando baseline (sin fine-tuning)..."
python3 src/baseline.py
echo ""

# 3. Fine-tuning
echo "[3/5] Fine-tuning con splits corregidos..."
python3 src/train.py
echo ""

# 4. Evaluación fine-tuned
echo "[4/5] Evaluando modelo fine-tuneado..."
python3 src/evaluate.py
echo ""

# 5. Comparación
echo "[5/5] Generando reporte comparativo..."
python3 src/compare_results.py
echo ""

echo "============================================"
echo "  ✅ Experimento 02 completado."
echo "  Ver resultados en: results/experiment_02.md"
echo "============================================"
