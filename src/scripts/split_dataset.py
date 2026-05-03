"""
split_dataset.py
----------------
Genera un split reproducible 70 / 10 / 20 sobre dataset.jsonl
y guarda los tres subconjuntos en data/processed/splits/.

Uso:
    python3 src/scripts/split_dataset.py

Salida:
    data/processed/splits/train.jsonl   (~70 %)
    data/processed/splits/val.jsonl     (~10 %)
    data/processed/splits/test.jsonl    (~20 %)
    data/processed/splits/split_info.json
"""

import json
import os
import random

SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.10
# TEST_RATIO  = 0.20  (lo que sobra)

BASE_PATH   = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
INPUT_FILE  = os.path.join(BASE_PATH, "data/processed/dataset.jsonl")
OUTPUT_DIR  = os.path.join(BASE_PATH, "data/processed/splits")


def main():
    # Cargar dataset
    with open(INPUT_FILE) as f:
        samples = [json.loads(line) for line in f]

    print(f"Total de muestras: {len(samples)}")

    # Barajar con seed fija
    random.seed(SEED)
    random.shuffle(samples)

    # Calcular índices de corte
    n = len(samples)
    train_end = int(n * TRAIN_RATIO)
    val_end   = train_end + int(n * VAL_RATIO)

    train = samples[:train_end]
    val   = samples[train_end:val_end]
    test  = samples[val_end:]

    print(f"  Train : {len(train)} ({len(train)/n*100:.1f}%)")
    print(f"  Val   : {len(val)}   ({len(val)/n*100:.1f}%)")
    print(f"  Test  : {len(test)}  ({len(test)/n*100:.1f}%)")

    # Guardar splits
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
        path = os.path.join(OUTPUT_DIR, f"{split_name}.jsonl")
        with open(path, "w") as f:
            for item in split_data:
                f.write(json.dumps(item) + "\n")
        print(f"  ✅ Guardado: {path}")

    # Guardar metadata del split para reproducibilidad
    info = {
        "seed": SEED,
        "total": n,
        "train": len(train),
        "val": len(val),
        "test": len(test),
        "ratios": {"train": TRAIN_RATIO, "val": VAL_RATIO, "test": round(1 - TRAIN_RATIO - VAL_RATIO, 2)},
    }
    info_path = os.path.join(OUTPUT_DIR, "split_info.json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"  ✅ Metadata guardada: {info_path}")


if __name__ == "__main__":
    main()
