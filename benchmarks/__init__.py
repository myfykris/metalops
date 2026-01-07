"""
Benchmarks Package for MetalOps

Provides utilities and individual benchmark modules for all metalops operations.
"""

import torch
import time
import json
import platform
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

# Package-level constants
DEVICE = 'mps'

def color_ratio(ratio: float, is_timing: bool = True) -> str:
    """Return GitHub-flavored markdown color indicator for ratio."""
    if is_timing:
        if ratio <= 0.3:
            return "💚"  # GPU wins big (3x+ faster)
        elif ratio <= 0.7:
            return "🟢"  # GPU wins moderate
        elif ratio <= 1.3:
            return "⚪"  # Close
        elif ratio <= 3.0:
            return "🟠"  # CPU wins moderate
        else:
            return "🔴"  # CPU wins big (3x+ faster)
    return "⚪"


def format_time(ms: float) -> str:
    """Format time in appropriate units."""
    if ms < 1:
        return f"{ms*1000:.1f}µs"
    elif ms < 1000:
        return f"{ms:.1f}ms"
    else:
        return f"{ms/1000:.2f}s"


def format_accuracy(err: float) -> str:
    """Format accuracy error."""
    if err < 1e-6:
        return f"✓ {err:.0e}"
    elif err < 1e-4:
        return f"✓ {err:.0e}"
    elif err < 1e-2:
        return f"~ {err:.0e}"
    else:
        return f"✗ {err:.0e}"


class BenchmarkResults:
    """Container for benchmark results that can be serialized to markdown."""
    
    def __init__(self):
        self.sections: List[Dict[str, Any]] = []
    
    def add_section(self, title: str, description: Optional[str], 
                   headers: List[str], rows: List[List[Any]]):
        self.sections.append({
            'title': title,
            'description': description,
            'headers': headers,
            'rows': rows
        })
    
    def to_markdown(self) -> str:
        lines = [
            "# Metalops Benchmark Results",
            "",
            f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
            "",
            "**Legend:** 💚 GPU wins big (>3x) | 🟢 GPU wins | ⚪ Close | 🟠 CPU wins | 🔴 CPU wins big (>3x)",
            "",
        ]
        
        for section in self.sections:
            lines.append(f"## {section['title']}")
            lines.append("")
            if section['description']:
                lines.append(f"*{section['description']}*")
                lines.append("")
            
            lines.append("| " + " | ".join(section['headers']) + " |")
            lines.append("|" + "|".join(["---"] * len(section['headers'])) + "|")
            
            for row in section['rows']:
                lines.append("| " + " | ".join(str(x) for x in row) + " |")
            
            lines.append("")
        
        return "\n".join(lines)


def save_benchmark_history(results_dict: Dict[str, Any], 
                          history_path: Optional[Path] = None):
    """Save benchmark results to JSONL history file."""
    if history_path is None:
        history_path = Path(__file__).parent.parent / "benchmark_history.jsonl"
    
    try:
        gpu_info = subprocess.run(
            ["system_profiler", "SPDisplaysDataType"],
            capture_output=True, text=True, timeout=5
        ).stdout
        gpu_model = "Unknown"
        for line in gpu_info.split("\n"):
            if "Chipset Model" in line:
                gpu_model = line.split(":")[-1].strip()
                break
    except Exception:
        gpu_model = "Unknown"
    
    record = {
        "timestamp": datetime.now().isoformat(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "gpu_model": gpu_model,
        "results": results_dict
    }
    
    with open(history_path, "a") as f:
        f.write(json.dumps(record) + "\n")


def load_benchmarks_json(filepath: Path) -> Dict[str, Any]:
    """Load benchmark data from JSON file."""
    if not filepath.exists():
        return {"metadata": {}, "sections": {}}
    with open(filepath, 'r') as f:
        return json.load(f)


def save_benchmarks_json(filepath: Path, data: Dict[str, Any]):
    """Save benchmark data to JSON file."""
    import datetime as dt
    data['metadata']['last_updated'] = dt.datetime.now().isoformat()
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def update_benchmark_row(data: Dict[str, Any], section_title: str, 
                        row_key: str, row_values: List[Any], 
                        raw_values: Optional[Dict[str, float]] = None):
    """Update a single row in the benchmark data. Moves current to history."""
    if section_title not in data['sections']:
        data['sections'][section_title] = {'description': None, 'headers': [], 'rows': {}}
    
    section = data['sections'][section_title]
    
    if row_key not in section['rows']:
        section['rows'][row_key] = {'current': None, 'history': []}
    
    row = section['rows'][row_key]
    
    if row['current'] is not None:
        history_entry = {
            'timestamp': data['metadata'].get('last_updated', datetime.now().isoformat()),
            'values': row['current'],
            'raw': row.get('current_raw', {})
        }
        row['history'].append(history_entry)
        if len(row['history']) > 50:
            row['history'] = row['history'][-50:]
    
    row['current'] = row_values
    if raw_values:
        row['current_raw'] = raw_values


# Section ordering for output
SECTION_ORDER = [
    "SVD (metalcore) ⭐ GPU WINS",
    "QR Batched (metalcore) ⭐ GPU WINS",
    "Cholesky (metalcore) ⭐ GPU WINS",
    "Linear Solve (metalcore) ⭐ GPU WINS",
    "RMSNorm (metalcore) ⭐ GPU WINS",
    "AdamW (metalcore) ⭐ GPU WINS",
    "Pipeline Operations ⭐ GPU WINS (No Transfer)",
    "LLM: Llama", "LLM: Mistral", "LLM: Qwen", "LLM: Gemma", "LLM: Phi",
    "Activations (metalcore)",
    "Eigendecomposition (metaleig)",
    "SVD (metalsvd)",
    "QR Single Matrix (metalcore)",
    "SDPA (metalcore)",
    "Usage Recommendations",
]
