import json
import os
from pathlib import Path

def collect():
    root = Path("/home/lzy123/deepresearch/Tianchi_DeepSearch_Agent")
    parts_dir = root / "answer_parts"
    output_path = root / "final_results.jsonl"
    
    results = []
    for part_file in parts_dir.glob("part_*.jsonl"):
        qid_str = part_file.stem.split("_")[1]
        qid = int(qid_str)
        
        with part_file.open("r", encoding="utf-8") as f:
            data = json.loads(f.read().strip())
            results.append({
                "id": qid,
                "answer": data["answer"]
            })
            
    # Sort by ID
    results.sort(key=lambda x: x["id"])
    
    with output_path.open("w", encoding="utf-8") as out:
        for res in results:
            out.write(json.dumps(res, ensure_ascii=False) + "\n")
            
    print(f"Collected {len(results)} answers to {output_path}")

if __name__ == "__main__":
    collect()
