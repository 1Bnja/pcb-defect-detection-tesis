import os
import json

def process_deeppcb():
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    raw_data_path = os.path.join(base_path, "data/raw/DeepPCB/PCBData")
    output_file = os.path.join(base_path, "data/processed/dataset.jsonl")
    
    dataset = []
    # Mapeo oficial de fallas de DeepPCB
    fault_map = {
        "1": "open circuit", 
        "2": "short circuit", 
        "3": "mouse bite", 
        "4": "spur", 
        "5": "spurious copper", 
        "6": "missing hole"
    }

    print(f"Iniciando escaneo en: {raw_data_path}")

    # Recorremos cada grupo (groupXXXXX)
    for group_folder in os.listdir(raw_data_path):
        group_path = os.path.join(raw_data_path, group_folder)
        if not os.path.isdir(group_path): continue

        # Extraemos el ID del grupo (ej: 00041)
        group_id = group_folder.replace("group", "")
        
        # Ruta donde están las imágenes y ruta donde están los TXT
        img_folder_path = os.path.join(group_path, group_id)
        txt_folder_path = os.path.join(group_path, f"{group_id}_not")

        if not os.path.exists(img_folder_path) or not os.path.exists(txt_folder_path):
            continue

        # Listamos solo las imágenes de prueba con fallas
        test_images = [f for f in os.listdir(img_folder_path) if f.endswith("_test.jpg")]

        for img_name in test_images:
            # ID base (ej: 00041000)
            base_id = img_name.split("_")[0]
            txt_name = f"{base_id}.txt"
            txt_path = os.path.join(txt_folder_path, txt_name)

            if os.path.exists(txt_path):
                faults = []
                with open(txt_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            f_type = fault_map.get(parts[4], "defect")
                            faults.append(f_type)
                
                unique_faults = list(set(faults))
                if unique_faults:
                    dataset.append({
                        "id": base_id,
                        "image": os.path.abspath(os.path.join(img_folder_path, img_name)),
                        "conversations": [
                            {"from": "human", "value": "<image>\nExamina esta placa de circuito e identifica fallas técnicas."},
                            {"from": "gpt", "value": f"Se han detectado los siguientes defectos: {', '.join(unique_faults)}."}
                        ]
                    })

    if dataset:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            for item in dataset:
                f.write(json.dumps(item) + '\n')
        print(f"✅ ¡Misión cumplida! Se generaron {len(dataset)} muestras para Phi-3.5 Vision.")
    else:
        print("❌ Increíble, pero sigue en 0. Verifica los permisos de las carpetas.")

if __name__ == "__main__":
    process_deeppcb()