#!/usr/bin/env python3
"""
Populate benchmarks.json from existing benchmark_history.jsonl data.
This script migrates historical benchmark data to the new JSON format.
"""

import json
import datetime
from pathlib import Path

def main():
    base_path = Path(__file__).parent
    history_path = base_path / "benchmark_history.jsonl"
    json_path = base_path / "benchmarks.json"
    
    # Initialize structure
    data = {
        "metadata": {
            "last_updated": datetime.datetime.now().isoformat(),
            "device": "mps",
            "system": "Apple M3 Max"
        },
        "sections": {}
    }
    
    # Section mappings from history keys to section names
    section_map = {
        "svd": ("SVD (metalcore) ⭐ GPU WINS", "Singular Value Decomposition using metalcore", ["Shape", "Config", "Metal", "CPU", "Ratio", "Status", "Recon Error"]),
        "qr_single": ("QR Single Matrix (metalcore)", "Single matrix QR factorization", ["Shape", "Config", "Metal", "CPU", "Ratio", "Status", "Q Error", "R Error"]),
        "qr_batched": ("QR Batched (metalcore) ⭐ GPU WINS", "Batched QR via Householder reflections", ["Shape", "Config", "Metal", "CPU", "Ratio", "Status", "Recon Error"]),
        "eigh": ("Eigendecomposition (metaleig)", "Symmetric eigenvalue decomposition", ["Shape", "Config", "Metal", "CPU", "Ratio", "Status", "Recon Error"]),
        "cholesky": ("Cholesky (metalcore) ⭐ GPU WINS", "Batched Cholesky decomposition with MAGMA-style shared memory", ["Shape", "Config", "Metal", "CPU", "Ratio", "Status", "Recon Error"]),
        "solve": ("Linear Solve (metalcore) ⭐ GPU WINS", "Fused LU decomposition with forward/back substitution", ["Shape", "Config", "Metal", "CPU", "Ratio", "Status", "Residual"]),
        "rmsnorm": ("RMSNorm (metalcore) ⭐ GPU WINS", "Fused RMSNorm kernel vs torch.nn.RMSNorm", ["Shape", "Config", "Metal", "CPU", "Ratio", "Status"]),
        "adamw": ("AdamW (metalcore) ⭐ GPU WINS", "Fused AdamW optimizer step vs torch.optim.AdamW", ["Params", "Size", "Metal", "CPU", "Ratio", "Status"]),
    }
    
    # Initialize all sections
    for key, (title, desc, headers) in section_map.items():
        data["sections"][title] = {
            "description": desc,
            "headers": headers,
            "rows": {}
        }
    
    # Add static sections
    data["sections"]["Activations (metalcore)"] = {
        "description": "GELU and SiLU activation functions",
        "headers": ["Name", "Shape", "Metal", "Torch", "Ratio", "Status"],
        "rows": {}
    }
    data["sections"]["SDPA (metalcore)"] = {
        "description": "Scaled Dot Product Attention (forward pass)",
        "headers": ["Config", "Params", "Metal", "Torch", "Ratio", "Status", "Max Err"],
        "rows": {}
    }
    data["sections"]["Pipeline Operations ⭐ GPU WINS (No Transfer)"] = {
        "description": "Chained operations showing GPU advantage when data stays on device",
        "headers": ["Operation", "Config", "Metal", "CPU", "Ratio", "Status"],
        "rows": {}
    }
    data["sections"]["Usage Recommendations"] = {
        "description": None,
        "headers": ["Operation", "When to Use Metal", "When to Use CPU"],
        "rows": {
            "EIGH": {"current": ["EIGH", "Batched symmetric matrices", "Single large matrices"], "history": []},
            "Pipeline": {"current": ["Pipeline", "Keep data on GPU to avoid transfer cost", "Single ops on CPU-resident data"], "history": []},
            "QR (batched)": {"current": ["QR (batched)", "Many small matrices (10x speedup!)", "Few matrices"], "history": []},
            "QR (single)": {"current": ["QR (single)", "—", "Always (sequential dependencies)"], "history": []},
            "SVD": {"current": ["SVD", "Batched small/medium matrices", "Single large matrices"], "history": []},
        }
    }
    
    # Process historical data
    if history_path.exists():
        with open(history_path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                timestamp = record.get("timestamp", "unknown")
                results = record.get("results", {})
                
                for key, (title, _, _) in section_map.items():
                    if key in results and results[key]:
                        for row in results[key]:
                            if len(row) >= 2:
                                # Row key is first two columns
                                row_key = f"{row[0]}|{row[1]}"
                                
                                if row_key not in data["sections"][title]["rows"]:
                                    data["sections"][title]["rows"][row_key] = {
                                        "current": None,
                                        "current_raw": {},
                                        "history": []
                                    }
                                
                                row_data = data["sections"][title]["rows"][row_key]
                                
                                # Add to history
                                row_data["history"].append({
                                    "timestamp": timestamp,
                                    "values": row,
                                    "raw": {}
                                })
                                
                                # Keep most recent as current
                                row_data["current"] = row
    
    # Sort history by timestamp for each row
    for section in data["sections"].values():
        for row_key, row_data in section.get("rows", {}).items():
            if "history" in row_data and row_data["history"]:
                row_data["history"].sort(key=lambda x: x.get("timestamp", ""))
                # Keep only last 50
                row_data["history"] = row_data["history"][-50:]
    
    # Save
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    # Count totals
    total_rows = sum(len(s.get("rows", {})) for s in data["sections"].values())
    total_history = sum(
        sum(len(r.get("history", [])) for r in s.get("rows", {}).values())
        for s in data["sections"].values()
    )
    
    print(f"Created {json_path}")
    print(f"  Sections: {len(data['sections'])}")
    print(f"  Total rows: {total_rows}")
    print(f"  Total historical entries: {total_history}")

if __name__ == "__main__":
    main()
