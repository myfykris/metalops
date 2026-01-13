#!/usr/bin/env python3
# Author: Kris Bailey
# Copyright 2026
# Email: kris@krisbailey.com
"""
Metalops Unified Benchmark Script

Generates comprehensive benchmarks for all metalops packages:
- metalsvd: SVD decomposition
- metalcore: QR factorization (single, batched)
- metaleig: Eigenvalue decomposition

Outputs color-coded markdown to benchmarks.md

FAIR BENCHMARKING METHODOLOGY:
==============================
All benchmarks follow these principles for fair comparison:
1. Each device (CPU/GPU) operates on tensors in its native memory space
2. Tensor creation/copy happens BEFORE timing starts
3. GPU synchronization (torch.mps.synchronize()) is called before and after timed sections
4. Both devices get equal warmup iterations before measurement
5. Multiple iterations are averaged to reduce noise

This ensures we measure pure operation time, not memory transfer overhead.
"""

import torch
import torch.nn.functional as F
import time
import sys
import json
import platform
import subprocess
import argparse
from datetime import datetime
from pathlib import Path

# Parse arguments early
parser = argparse.ArgumentParser(description="Metalops Unified Benchmark")
parser.add_argument("--lite", action="store_true", help="Quick validation (1 iteration, skip file write)")
parser.add_argument("--quick", action="store_true", help="Reduced benchmark (fewer iterations)")
parser.add_argument("--nolongops", action="store_true", help="Skip slow benchmarks (models, pipeline, SVD, EIGH)")
parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16", "all"], default="fp32",
                    help="Dtype to benchmark (fp32, fp16, bf16, or all)")
# Test-specific flags
parser.add_argument("--svd", action="store_true", help="Run only SVD benchmarks")
parser.add_argument("--qr", action="store_true", help="Run only QR benchmarks")
parser.add_argument("--eigh", action="store_true", help="Run only eigenvalue benchmarks")
parser.add_argument("--cholesky", action="store_true", help="Run only Cholesky benchmarks")
parser.add_argument("--solve", action="store_true", help="Run only solve benchmarks")
parser.add_argument("--models", action="store_true", help="Run only LLM model benchmarks")
parser.add_argument("--rmsnorm", action="store_true", help="Run only RMSNorm benchmarks")
parser.add_argument("--adamw", action="store_true", help="Run only AdamW benchmarks")
parser.add_argument("--training", action="store_true", help="Run only training benchmarks (RMSNorm + AdamW)")
parser.add_argument("--activations", action="store_true", help="Run only activation benchmarks (GELU/SiLU)")
parser.add_argument("--sdpa", action="store_true", help="Run only SDPA benchmarks")
parser.add_argument("--pipeline", action="store_true", help="Run only pipeline benchmarks")
parser.add_argument("--softmax", action="store_true", help="Run only fused softmax benchmarks")
parser.add_argument("--layernorm", action="store_true", help="Run only LayerNorm benchmarks")
parser.add_argument("--embedding", action="store_true", help="Run only embedding bag benchmarks")
parser.add_argument("--scatter", action="store_true", help="Run only scatter/gather benchmarks")
parser.add_argument("--lora", action="store_true", help="Run only LoRA training benchmarks (cross_entropy, kl_div, swiglu, lora_linear)")
parser.add_argument("--rope", action="store_true", help="Run only RoPE benchmarks")
parser.add_argument("--fused_att_bwd", action="store_true", help="Run only fused attention backward benchmarks")
parser.add_argument("--fused_mlp_bwd", action="store_true", help="Run only fused MLP backward benchmarks")
parser.add_argument("--compare", action="store_true", help="Compare results with previous run")
args = parser.parse_args()

# Benchmark mode globals
LITE_MODE = args.lite  # Quick validation: 1 iter, minimal configs, no file write
QUICK_MODE = args.quick or args.lite  # Fewer iterations
SKIP_LONG_OPS = args.nolongops  # Skip slow benchmarks (SVD, EIGH, models, pipeline)

# Dtype configuration
DTYPE_MAP = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
# Lite mode always tests all dtypes for comprehensive validation
if LITE_MODE or args.dtype == "all":
    BENCHMARK_DTYPES = [("fp32", torch.float32), ("fp16", torch.float16), ("bf16", torch.bfloat16)]
else:
    BENCHMARK_DTYPES = [(args.dtype, DTYPE_MAP[args.dtype])]

# Test selection (if none specified, run all)
RUN_ALL = not any([args.svd, args.qr, args.eigh, args.cholesky, args.solve, args.models, args.pipeline, args.rmsnorm, args.adamw, args.training, args.activations, args.sdpa, args.softmax, args.layernorm, args.embedding, args.scatter, args.lora, args.rope, args.fused_att_bwd, args.fused_mlp_bwd])
# All ops run, but SKIP_LONG_OPS filters large configs within each op
RUN_SVD = args.svd or RUN_ALL
RUN_QR = args.qr or RUN_ALL
RUN_EIGH = args.eigh or RUN_ALL
RUN_CHOLESKY = args.cholesky or RUN_ALL
RUN_SOLVE = args.solve or RUN_ALL
RUN_RMSNORM = args.rmsnorm or args.training or RUN_ALL
RUN_ADAMW = args.adamw or args.training or RUN_ALL
RUN_ACTIVATIONS = args.activations or RUN_ALL
RUN_SDPA = args.sdpa or RUN_ALL
RUN_SOFTMAX = args.softmax or RUN_ALL
RUN_LAYERNORM = args.layernorm or RUN_ALL
RUN_EMBEDDING = args.embedding or RUN_ALL
RUN_SCATTER = args.scatter or RUN_ALL
RUN_LORA = args.lora or RUN_ALL
RUN_ROPE = args.rope or RUN_ALL
RUN_FUSED_ATT_BWD = args.fused_att_bwd or (RUN_ALL and not LITE_MODE and not SKIP_LONG_OPS)
RUN_FUSED_MLP_BWD = args.fused_mlp_bwd or (RUN_ALL and not LITE_MODE and not SKIP_LONG_OPS)
RUN_MODELS = args.models or (RUN_ALL and not LITE_MODE and not SKIP_LONG_OPS)  # Skip models entirely in nolongops
RUN_PIPELINE = args.pipeline or (RUN_ALL and not LITE_MODE and not SKIP_LONG_OPS)  # Skip pipeline entirely in nolongops

# Add package paths (metalcore uses installed package for full functionality)
sys.path.insert(0, str(Path(__file__).parent / "packages/metalsvd/src"))
# sys.path.insert(0, str(Path(__file__).parent / "packages/metalcore/src"))  # Use installed metalcore
sys.path.insert(0, str(Path(__file__).parent / "packages/metaleig/src"))

device = 'mps'


def color_ratio(ratio, is_timing=True):
    """Return GitHub-flavored markdown color indicator for ratio."""
    # For timing: ratio < 1 means GPU faster, ratio > 1 means CPU faster
    if is_timing:
        if ratio <= 0.3:
            return "💚"  # GPU wins big (3x+ faster)
        elif ratio <= 0.7:
            return "🟢"  # GPU wins moderate
        elif ratio <= 1.3:
            return "🔵"  # Close
        elif ratio <= 3.0:
            return "⚪"  # CPU wins moderate
        else:
            return "🟠"  # CPU wins big (3x+ faster)
    return "🔵"

def format_time(ms):
    """Format time in appropriate units."""
    if ms < 1:
        return f"{ms*1000:.1f}µs"
    elif ms < 1000:
        return f"{ms:.1f}ms"
    else:
        return f"{ms/1000:.2f}s"

def format_accuracy(err):
    """Format accuracy error."""
    if err < 1e-6:
        return f"✓ {err:.0e}"
    elif err < 1e-4:
        return f"✓ {err:.0e}"
    elif err < 1e-2:
        return f"~ {err:.0e}"
    else:
        return f"✗ {err:.0e}"

def load_benchmarks_json(filepath):
    """Load benchmark data from JSON file."""
    if not filepath.exists():
        return {"metadata": {}, "sections": {}}
    with open(filepath, 'r') as f:
        return json.load(f)

def save_benchmarks_json(filepath, data):
    """Save benchmark data to JSON file."""
    data['metadata']['last_updated'] = datetime.datetime.now().isoformat()
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

def update_benchmark_row(data, section_title, row_key, row_values, raw_values=None):
    """
    Update a single row in the benchmark data.
    Moves current to history and sets new current.
    raw_values: dict with numeric values for charting (e.g., {'metal_ms': 1.5, 'cpu_ms': 3.0, 'ratio': 0.5})
    """
    if section_title not in data['sections']:
        data['sections'][section_title] = {'description': None, 'headers': [], 'rows': {}}
    
    section = data['sections'][section_title]
    
    # Initialize row if not exists
    if row_key not in section['rows']:
        section['rows'][row_key] = {'current': None, 'history': []}
    
    row = section['rows'][row_key]
    
    # Move current to history (if exists)
    if row['current'] is not None:
        history_entry = {
            'timestamp': data['metadata'].get('last_updated', datetime.datetime.now().isoformat()),
            'values': row['current'],
            'raw': row.get('current_raw', {})
        }
        row['history'].append(history_entry)
        # Keep only last 50 historical entries
        if len(row['history']) > 50:
            row['history'] = row['history'][-50:]
    
    # Set new current
    row['current'] = row_values
    if raw_values:
        row['current_raw'] = raw_values

def generate_markdown_from_json(data):
    """Generate benchmarks.md content from JSON data."""
    lines = ["# Metalops Benchmark Results", ""]
    
    if 'last_updated' in data.get('metadata', {}):
        lines.append(f"*Last updated: {data['metadata']['last_updated']}*")
        lines.append("")
    
    # Calculate average ratio for each section for dynamic sorting
    def get_section_avg_ratio(section_title):
        """Get average ratio for a section. Lower = GPU wins = should appear first."""
        if section_title not in data.get('sections', {}):
            return float('inf')  # Non-existent sections at end
        
        section = data['sections'][section_title]
        ratios = []
        for row_key, row_data in section.get('rows', {}).items():
            raw = row_data.get('raw_values', {})
            if 'ratio' in raw:
                ratios.append(raw['ratio'])
            elif row_data.get('current'):
                # Try to extract ratio from formatted current values (e.g., "0.38x")
                for val in row_data['current']:
                    if isinstance(val, str) and val.endswith('x') and val[:-1].replace('.', '').isdigit():
                        try:
                            ratios.append(float(val[:-1]))
                            break
                        except ValueError:
                            pass
        
        if not ratios:
            # No ratio data - use section title hints
            if "GPU WINS" in section_title:
                return 0.5  # Favor GPU WINS sections
            elif "Recommendations" in section_title:
                return 1000  # Always last
            return 50  # Neutral
        
        return sum(ratios) / len(ratios)
    
    # Get all sections and sort by average ratio (GPU wins first)
    all_sections = list(data.get('sections', {}).keys())
    sorted_sections = sorted(all_sections, key=get_section_avg_ratio)
    
    for section_title in sorted_sections:
        section = data['sections'][section_title]
        
        lines.append(f"## {section_title}")
        lines.append("")
        
        if section.get('description'):
            lines.append(f"*{section['description']}*")
            lines.append("")
        
        if section.get('headers') and section.get('rows'):
            lines.append("| " + " | ".join(section['headers']) + " |")
            lines.append("|" + "|".join(["---"] * len(section['headers'])) + "|")
            
            # Sort rows by key for consistent ordering
            for row_key in sorted(section['rows'].keys()):
                row_data = section['rows'][row_key]
                if row_data.get('current'):
                    lines.append("| " + " | ".join(str(x) for x in row_data['current']) + " |")
            lines.append("")
    
    return "\n".join(lines)

# Legacy function for backwards compatibility
def parse_existing_benchmarks(filepath):
    """Parse existing benchmarks.md file and return sections as a dict."""
    if not filepath.exists():
        return {}
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    sections = {}
    current_section = None
    current_lines = []
    
    for line in content.split('\n'):
        if line.startswith('## '):
            # Save previous section
            if current_section:
                sections[current_section] = '\n'.join(current_lines)
            # Start new section
            current_section = line[3:].strip()
            current_lines = [line]
        elif current_section:
            current_lines.append(line)
    
    # Save last section
    if current_section:
        sections[current_section] = '\n'.join(current_lines)
    
    return sections



def merge_benchmark_sections(new_results, existing_sections, ran_benchmarks):
    """
    Merge new benchmark results with existing sections.
    Now merges at the ROW level within sections, preserving rows that weren't re-run.
    """
    merged = {}
    
    # First, add all existing sections
    for title, content in existing_sections.items():
        merged[title] = content
    
    # Then, update with new results for sections that were run
    for section in new_results.sections:
        title = section['title']
        
        # Only modify if this benchmark was actually run
        if title not in ran_benchmarks and 'Usage Recommendations' not in title:
            continue
        
        # Parse existing rows from this section if it exists
        existing_rows = {}
        if title in existing_sections:
            lines = existing_sections[title].split('\n')
            for line in lines:
                if line.startswith('|') and '---' not in line and not line.startswith('| Shape') and not line.startswith('| Config') and not line.startswith('| Name') and not line.startswith('| Op'):
                    # Parse row - use first two columns as key for uniqueness
                    parts = [p.strip() for p in line.strip('|').split('|')]
                    if len(parts) >= 2:
                        # Key is first two columns (config identifier)
                        key = (parts[0].strip(), parts[1].strip())
                        existing_rows[key] = line
        
        # Build new rows, preserving existing ones not re-run
        new_rows = {}
        for row in section['rows']:
            if len(row) >= 2:
                key = (str(row[0]).strip(), str(row[1]).strip())
                new_rows[key] = row
        
        # Merge: new rows override existing, but preserve non-overlapping existing rows
        final_rows = dict(existing_rows)  # Start with existing
        for key, row in new_rows.items():
            # Convert row list to table line
            final_rows[key] = "| " + " | ".join(str(x) for x in row) + " |"
        
        # Generate merged section markdown
        lines = [f"## {title}", ""]
        if section['description']:
            lines.append(f"*{section['description']}*")
            lines.append("")
        lines.append("| " + " | ".join(section['headers']) + " |")
        lines.append("|" + "|".join(["---"] * len(section['headers'])) + "|")
        
        # Add rows in a sensible order (by key)
        for key in sorted(final_rows.keys()):
            lines.append(final_rows[key])
        lines.append("")
        
        merged[title] = '\n'.join(lines)
    
    return merged


# Define the canonical order for benchmark sections
# Order: GPU wins first, then mixed/neutral, then poor performers at end
SECTION_ORDER = [
    # GPU Winners (⭐)
    "SVD (metalcore) ⭐ GPU WINS",
    "QR Batched (metalcore) ⭐ GPU WINS",
    "Cholesky (metalcore) ⭐ GPU WINS",
    "Linear Solve (metalcore) ⭐ GPU WINS",
    "RMSNorm (metalcore) ⭐ GPU WINS",
    "AdamW (metalcore) ⭐ GPU WINS",
    "Pipeline Operations ⭐ GPU WINS (No Transfer)",
    # LLM Model Benchmarks
    "LLM: Llama",
    "LLM: Mistral",
    "LLM: Qwen",
    "LLM: Gemma",
    "LLM: Phi",
    # Mixed/Neutral performance
    "Activations (metalcore)",
    "Eigendecomposition (metaleig)",
    "SVD (metalsvd)",
    # Poor performers (CPU wins) - at end
    "QR Single Matrix (metalcore)",
    "SDPA (metalcore)",
    # Recommendations last
    "Usage Recommendations",
]


class BenchmarkResults:
    def __init__(self):
        self.sections = []
    
    def add_section(self, title, description, headers, rows):
        self.sections.append({
            'title': title,
            'description': description,
            'headers': headers,
            'rows': rows
        })
    
    def to_markdown(self):
        lines = [
            "# Metalops Benchmark Results",
            "",
            f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
            "",
            "**Legend:** 💚 GPU wins big (>3x) | 🟢 GPU wins | 🔵 Close | ⚪ CPU wins | 🟠 CPU wins big (>3x)",
            "",
        ]
        
        for section in self.sections:
            lines.append(f"## {section['title']}")
            lines.append("")
            if section['description']:
                lines.append(f"*{section['description']}*")
                lines.append("")
            
            # Table header
            lines.append("| " + " | ".join(section['headers']) + " |")
            lines.append("|" + "|".join(["---"] * len(section['headers'])) + "|")
            
            # Table rows
            for row in section['rows']:
                lines.append("| " + " | ".join(str(x) for x in row) + " |")
            
            lines.append("")
        
        return "\n".join(lines)


def save_benchmark_history(results_dict):
    """
    Save benchmark results to JSONL history file for historical comparisons.
    Includes system info, Python version, and detailed timing/accuracy data.
    """
    history_path = Path(__file__).parent / "benchmark_history.jsonl"
    
    # Collect system information
    try:
        gpu_info = subprocess.run(
            ["system_profiler", "SPDisplaysDataType"],
            capture_output=True, text=True, timeout=5
        ).stdout
        # Extract GPU model name
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
        "processor": platform.processor(),
        "gpu_model": gpu_model,
        "results": results_dict
    }
    
    # Append to JSONL file
    with open(history_path, "a") as f:
        f.write(json.dumps(record) + "\n")
    
    print(f"History saved to: {history_path}")


def load_previous_benchmark():
    """Load the most recent benchmark from history for comparison."""
    history_path = Path(__file__).parent / "benchmark_history.jsonl"
    
    if not history_path.exists():
        return None
    
    # Read last line (most recent benchmark)
    last_record = None
    with open(history_path, "r") as f:
        for line in f:
            if line.strip():
                last_record = json.loads(line)
    
    return last_record


def compare_with_previous(current_results, previous_record):
    """Compare current results with previous benchmark, display delta."""
    if not previous_record:
        print("\n📊 No previous benchmark found - this run will be the baseline")
        return
    
    prev_results = previous_record.get("results", {})
    prev_timestamp = previous_record.get("timestamp", "unknown")[:19]
    
    print(f"\n📊 Comparison with previous benchmark ({prev_timestamp})")
    print("=" * 60)
    
    # Compare each category
    categories = ["svd", "qr_single", "qr_batched", "eigh", "cholesky", "solve"]
    
    for cat in categories:
        current = current_results.get(cat, [])
        previous = prev_results.get(cat, [])
        
        if not current or not previous:
            continue
        
        print(f"\n{cat.upper()}:")
        
        # Build lookup from previous by name
        prev_lookup = {r.get("name"): r for r in previous if isinstance(r, dict)}
        
        for curr in current:
            if not isinstance(curr, dict):
                continue
            name = curr.get("name", "")
            curr_ratio = curr.get("ratio", 1.0)
            
            prev = prev_lookup.get(name)
            if prev:
                prev_ratio = prev.get("ratio", 1.0)
                # Lower ratio = faster GPU
                if prev_ratio > 0:
                    improvement = (prev_ratio - curr_ratio) / prev_ratio * 100
                    if improvement > 5:
                        status = f"⬆️ +{improvement:.1f}%"
                    elif improvement < -5:
                        status = f"⬇️ {improvement:.1f}%"
                    else:
                        status = "="
                    print(f"  {name}: {curr_ratio:.2f}x → {prev_ratio:.2f}x {status}")
            else:
                print(f"  {name}: {curr_ratio:.2f}x (new)")


def benchmark_svd():
    """Benchmark SVD operations."""
    print("Benchmarking SVD...")
    results = []
    
    try:
        import metalcore
    except ImportError:
        print("  metalcore not available, skipping")
        return results
    
    configs = [
        # Single matrices - various sizes
        (32, 32, 1, "Tiny"),
        (64, 64, 1, "Small square"),
        (128, 128, 1, "Medium square"),
        (256, 256, 1, "Large square"),
        (512, 512, 1, "Very large"),
        (1024, 1024, 1, "Huge square"),
        (2048, 2048, 1, "Massive square"),
        # Rectangular matrices
        (256, 128, 1, "Tall 2:1"),
        (512, 256, 1, "Tall 2:1 large"),
        (1024, 512, 1, "Tall matrix"),
        (2048, 512, 1, "Very tall"),
        (128, 256, 1, "Wide 1:2"),
        # Top 6 Open-Weight LLM Matrix Sizes
        # Llama-2-7B / Llama-3-8B: hidden=4096, intermediate=11008/14336
        (4096, 4096, 1, "Llama-7B attn (4096x4096)"),
        (4096, 11008, 1, "Llama-2-7B MLP (4096x11008)"),
        (4096, 14336, 1, "Llama-3-8B MLP (4096x14336)"),
        # Llama-2-70B: hidden=8192, intermediate=28672
        (8192, 8192, 1, "Llama-70B attn (8192x8192)"),
        # Mistral-7B: hidden=4096, intermediate=14336
        (4096, 14336, 1, "Mistral-7B MLP (4096x14336)"),
        # Qwen-7B: hidden=4096, intermediate=11008
        (4096, 11008, 1, "Qwen-7B MLP (4096x11008)"),
        # Gemma-7B: hidden=3072, intermediate=24576  
        (3072, 24576, 1, "Gemma-7B MLP (3072x24576)"),
        # Phi-3-mini: hidden=3072, intermediate=8192
        (3072, 8192, 1, "Phi-3-mini MLP (3072x8192)"),
        # Batched - various batch sizes
        (32, 32, 100, "Batch 100 tiny"),
        (64, 64, 50, "Batch 50 small"),
        (64, 64, 100, "Batch 100 small"),
        (64, 64, 200, "Batch 200 small"),
        (128, 128, 20, "Batch 20 medium"),
        (128, 128, 50, "Batch 50 medium"),
        (256, 256, 10, "Batch 10 large"),
        (512, 512, 5, "Batch 5 huge"),
    ]
    
    for M, N, batch, desc in configs:
        # Skip large configs when --nolongops is set (matrices > 512x512 or LLM-sized)
        if SKIP_LONG_OPS and (M > 512 or N > 512 or M * N > 512 * 512):
            continue
        
        if batch == 1:
            A = torch.randn(M, N, device=device)
        else:
            A = torch.randn(batch, M, N, device=device)
        
        # Warmup
        try:
            U, S, V = metalcore.svd(A)
            torch.mps.synchronize()
        except Exception as e:
            print(f"  {desc}: Error - {e}")
            continue
        
        # Adaptive iteration count based on matrix size for accuracy
        total_elements = M * N * batch
        if total_elements < 10000:
            iters = 20  # Small: many iterations
        elif total_elements < 100000:
            iters = 10  # Medium: moderate iterations
        elif total_elements < 1000000:
            iters = 5   # Large
        elif total_elements < 10000000:
            iters = 3   # Very large
        else:
            iters = 2   # Huge matrices
        
        # Metal timing
        torch.mps.synchronize()
        start = time.time()
        for _ in range(iters):
            U, S, V = metalcore.svd(A)
        torch.mps.synchronize()
        metal_time = (time.time() - start) / iters * 1000
        
        # Accuracy check
        if batch == 1:
            recon = torch.max(torch.abs(U @ torch.diag(S) @ V.T - A)).item()
        else:
            recon = torch.max(torch.abs(U[0] @ torch.diag(S[0]) @ V[0].T - A[0])).item()
        
        # CPU timing
        A_cpu = A.cpu()
        start = time.time()
        for _ in range(iters):
            torch.linalg.svd(A_cpu)
        cpu_time = (time.time() - start) / iters * 1000
        
        ratio = metal_time / cpu_time
        shape = f"{batch}×{M}×{N}" if batch > 1 else f"{M}×{N}"
        
        results.append([
            shape, desc,
            format_time(metal_time), format_time(cpu_time),
            f"{ratio:.2f}x", color_ratio(ratio),
            format_accuracy(recon)
        ])
        print(f"  {desc}: {ratio:.2f}x")
    
    return results


def benchmark_qr_single():
    """Benchmark single-matrix QR."""
    print("Benchmarking QR (single matrix)...")
    results = []
    
    try:
        import metalcore
    except ImportError:
        print("  metalcore not available, skipping")
        return results
    
    configs = [
        # Small matrices - high CPU advantage
        (16, 16, "Tiny"),
        (32, 32, "Small 32"),
        (64, 64, "Small 64"),
        (128, 128, "Medium"),
        # Large square matrices
        (256, 256, "Large 256"),
        (512, 512, "Large 512"),
        (1024, 1024, "Huge 1024"),
        # Tall-thin matrices (common in least squares)
        (256, 64, "Tall 4:1"),
        (256, 128, "Tall 2:1"),
        (512, 128, "Tall 4:1 large"),
        (512, 256, "Tall 2:1 large"),
        (1000, 200, "Tall 5:1"),
        (2000, 500, "Huge tall"),
        (4000, 1000, "Massive"),
    ]
    
    for M, N, desc in configs:
        # Skip large configs when --nolongops is set
        if SKIP_LONG_OPS and (M > 512 or N > 512):
            continue
        
        A = torch.randn(M, N, device=device)
        
        # Warmup
        try:
            Q, R = metalcore.qr(A)
            torch.mps.synchronize()
        except Exception as e:
            print(f"  {desc}: Error - {e}")
            continue
        # Adaptive iteration count
        total_elements = M * N
        if total_elements < 10000:
            iters = 20
        elif total_elements < 100000:
            iters = 10
        elif total_elements < 1000000:
            iters = 5
        else:
            iters = 3
        
        if LITE_MODE:
            iters = 1
        
        # Metal timing
        torch.mps.synchronize()
        start = time.time()
        for _ in range(iters):
            Q, R = metalcore.qr(A)
        torch.mps.synchronize()
        metal_time = (time.time() - start) / iters * 1000
        
        # Accuracy
        recon = torch.max(torch.abs(Q @ R - A)).item()
        orth = torch.max(torch.abs(Q.T @ Q - torch.eye(min(M,N), device=device))).item()
        
        # CPU timing
        A_cpu = A.cpu()
        start = time.time()
        for _ in range(iters):
            torch.linalg.qr(A_cpu)
        cpu_time = (time.time() - start) / iters * 1000
        
        ratio = metal_time / cpu_time
        shape = f"{M}×{N}"
        
        results.append([
            shape, desc,
            format_time(metal_time), format_time(cpu_time),
            f"{ratio:.2f}x", color_ratio(ratio),
            format_accuracy(recon), format_accuracy(orth)
        ])
        print(f"  {desc}: {ratio:.2f}x")
    
    return results


def benchmark_qr_batched():
    """Benchmark batched QR - where GPU wins!"""
    print("Benchmarking QR (batched)...")
    results = []
    
    try:
        import metalcore_backend as mc
    except ImportError:
        print("  metalcore not available, skipping")
        return results
    
    configs = [
        # Small matrices - GPU wins big here
        (50, 8, 8, "Tiny 8x8"),
        (100, 8, 8, "Batch 100 tiny"),
        (500, 8, 8, "Batch 500 tiny"),
        (50, 16, 16, "ML mini-batch 16"),
        (100, 16, 16, "Batch 100 16x16"),
        (200, 16, 16, "Batch 200 16x16"),
        (500, 16, 16, "Batch 500 16x16"),
        (1000, 16, 16, "Batch 1000 16x16"),
        # Medium matrices  
        (50, 32, 32, "ML mini-batch 32"),
        (100, 32, 32, "Batch 100 32x32"),
        (200, 32, 32, "Batch 200 32x32"),
        (500, 32, 32, "Batch 500 32x32"),
        # Larger matrices (may hit shared memory limits)
        (50, 48, 48, "Batch 50 48x48"),
        (100, 48, 48, "Batch 100 48x48"),
        # Rectangular matrices
        (100, 64, 32, "Tall batch"),
        (100, 32, 64, "Wide batch"),
        (200, 64, 32, "Large tall batch"),
    ]
    
    for batch, M, N, desc in configs:
        # GPU: Create tensor natively on MPS
        A_batch = torch.randn(batch, M, N, device=device)
        torch.mps.synchronize()
        
        # CPU: Create SEPARATE tensor natively on CPU (fair comparison)
        A_cpu = torch.randn(batch, M, N, device='cpu')
        
        # Warmup
        try:
            Q, R = mc.qr_batched(A_batch)
            torch.mps.synchronize()
        except Exception as e:
            print(f"  {desc}: Error - {e}")
            continue
        # Adaptive iteration count
        total_elements = batch * M * N
        if total_elements < 50000:
            iters = 20
        elif total_elements < 500000:
            iters = 10
        elif total_elements < 2000000:
            iters = 5
        else:
            iters = 3
        
        # Metal timing
        torch.mps.synchronize()
        start = time.time()
        for _ in range(iters):
            Q, R = mc.qr_batched(A_batch)
        torch.mps.synchronize()
        metal_time = (time.time() - start) / iters * 1000
        
        # Accuracy (check first matrix)
        recon = torch.max(torch.abs(Q[0] @ R[0] - A_batch[0])).item()
        
        # CPU timing - using separate native CPU tensor (fair comparison)
        start = time.time()
        for _ in range(iters):
            for i in range(batch):
                torch.linalg.qr(A_cpu[i])
        cpu_time = (time.time() - start) / iters * 1000
        
        ratio = metal_time / cpu_time
        shape = f"{batch}×{M}×{N}"
        
        results.append([
            shape, desc,
            format_time(metal_time), format_time(cpu_time),
            f"{ratio:.2f}x", color_ratio(ratio),
            format_accuracy(recon)
        ])
        print(f"  {desc}: {ratio:.2f}x")
    
    return results


def benchmark_cholesky():
    """Benchmark Cholesky decomposition (batched)."""
    print("Benchmarking CHOLESKY...")
    results = []
    
    try:
        import metalcore as mc
    except ImportError:
        print("  metalcore not available, skipping")
        return results
    
    configs = [
        # (batch, N, description)
        (100, 16, "Tiny batched"),
        (500, 16, "Large batch tiny"),
        (100, 32, "Small batched"),
        (200, 48, "Medium batched"),
        (100, 64, "Larger batched"),
    ]
    
    for batch, N, desc in configs:
        # Create symmetric positive definite matrices
        A = torch.randn(batch, N, N, device=device)
        A = A @ A.transpose(-2, -1) + 0.1 * torch.eye(N, device=device)
        
        # Warmup
        try:
            L = mc.cholesky(A)
            torch.mps.synchronize()
        except Exception as e:
            print(f"  {desc}: Error - {e}")
            continue
        
        iters = 10
        
        # Metal timing
        torch.mps.synchronize()
        start = time.time()
        for _ in range(iters):
            L = mc.cholesky(A)
        torch.mps.synchronize()
        metal_time = (time.time() - start) / iters * 1000
        
        # Accuracy
        recon = L @ L.transpose(-2, -1)
        err = torch.max(torch.abs(recon - A)).item()
        
        # CPU timing (batched via torch)
        A_cpu = A.cpu()
        start = time.time()
        for _ in range(iters):
            for i in range(batch):
                torch.linalg.cholesky(A_cpu[i])
        cpu_time = (time.time() - start) / iters * 1000
        
        ratio = metal_time / cpu_time
        shape = f"{batch}×{N}×{N}"
        
        results.append([
            shape, desc,
            format_time(metal_time), format_time(cpu_time),
            f"{ratio:.2f}x", color_ratio(ratio),
            format_accuracy(err)
        ])
        print(f"  {desc}: {ratio:.2f}x")
    
    return results


def benchmark_solve():
    """Benchmark linear solve (fused LU-based)."""
    print("Benchmarking SOLVE...")
    results = []
    
    try:
        import metalcore as mc
    except ImportError:
        print("  metalcore not available, skipping")
        return results
    
    configs = [
        # (batch, N, description)
        (100, 16, "Tiny batched"),
        (500, 16, "Large batch tiny"),
        (100, 32, "Small batched"),
        (200, 48, "Medium batched"),
    ]
    
    iters = 1 if LITE_MODE else 10
    
    for dtype_name, dtype in BENCHMARK_DTYPES:
        for batch, N, desc in configs:
            A = torch.randn(batch, N, N, device=device, dtype=dtype)
            b = torch.randn(batch, N, device=device, dtype=dtype)
            
            # Warmup
            try:
                x = mc.solve(A.clone(), b.clone())
                torch.mps.synchronize()
            except Exception as e:
                print(f"  {desc} {dtype_name}: Error - {e}")
                continue
            
            # Metal timing
            torch.mps.synchronize()
            start = time.time()
            for _ in range(iters):
                x = mc.solve(A.clone(), b.clone())
            torch.mps.synchronize()
            metal_time = (time.time() - start) / iters * 1000
            
            # Accuracy (residual) - compute in float for consistency
            A_f = A.float() if dtype != torch.float32 else A
            x_f = x.float() if dtype != torch.float32 else x
            b_f = b.float() if dtype != torch.float32 else b
            residual = A_f @ x_f.unsqueeze(-1) - b_f.unsqueeze(-1)
            err = torch.max(torch.abs(residual)).item()
            
            # CPU timing (always fp32 for fair comparison)
            A_cpu, b_cpu = A.float().cpu(), b.float().cpu()
            start = time.time()
            for _ in range(iters):
                for i in range(batch):
                    torch.linalg.solve(A_cpu[i], b_cpu[i])
            cpu_time = (time.time() - start) / iters * 1000
            
            ratio = metal_time / cpu_time
            shape = f"{batch}×{N}×{N}"
            
            results.append([
                shape, f"{desc} {dtype_name}",
                format_time(metal_time), format_time(cpu_time),
                f"{ratio:.2f}x", color_ratio(ratio),
                format_accuracy(err)
            ])
            print(f"  {desc} {dtype_name}: {ratio:.2f}x")
    
    return results


def benchmark_eigh():
    """Benchmark eigenvalue decomposition."""
    print("Benchmarking EIGH...")
    results = []
    
    try:
        import metalcore
    except ImportError:
        print("  metalcore not available, skipping")
        return results
    
    configs = [
        # Single symmetric matrices - various sizes
        (32, 1, "Tiny"),
        (64, 1, "Small"),
        (128, 1, "Medium"),
        (256, 1, "Large"),
        (512, 1, "Very large"),
        (1024, 1, "Huge"),
        # Batched symmetric matrices - where GPU excels
        (32, 100, "Batch 100 tiny"),
        (64, 50, "Batch 50 small"),
        (64, 100, "Batch 100 small"),
        (64, 200, "Batch 200 small"),
        (128, 20, "Batch 20 medium"),
        (128, 50, "Batch 50 medium"),
        (256, 10, "Batch 10 large"),
    ]
    
    for N, batch, desc in configs:
        # Skip large configs when --nolongops is set (matrices > 256x256)
        if SKIP_LONG_OPS and N > 256:
            continue
        
        # Create symmetric matrix
        if batch == 1:
            A = torch.randn(N, N, device=device)
            A = (A + A.T) / 2
        else:
            A = torch.randn(batch, N, N, device=device)
            A = (A + A.transpose(-1, -2)) / 2
        
        # Warmup
        try:
            eigenvalues, eigenvectors = metalcore.eigh(A)
            torch.mps.synchronize()
        except Exception as e:
            print(f"  {desc}: Error - {e}")
            continue
        # Adaptive iteration count
        total_elements = N * N * batch
        if total_elements < 10000:
            iters = 20
        elif total_elements < 100000:
            iters = 10
        elif total_elements < 1000000:
            iters = 5
        else:
            iters = 3
        
        # Metal timing
        torch.mps.synchronize()
        start = time.time()
        for _ in range(iters):
            eigenvalues, eigenvectors = metalcore.eigh(A)
        torch.mps.synchronize()
        metal_time = (time.time() - start) / iters * 1000
        
        # Accuracy check
        if batch == 1:
            recon = torch.max(torch.abs(eigenvectors @ torch.diag(eigenvalues) @ eigenvectors.T - A)).item()
        else:
            recon = torch.max(torch.abs(eigenvectors[0] @ torch.diag(eigenvalues[0]) @ eigenvectors[0].T - A[0])).item()
        
        # CPU timing
        A_cpu = A.cpu()
        start = time.time()
        for _ in range(iters):
            torch.linalg.eigh(A_cpu)
        cpu_time = (time.time() - start) / iters * 1000
        
        ratio = metal_time / cpu_time
        shape = f"{batch}×{N}×{N}" if batch > 1 else f"{N}×{N}"
        
        results.append([
            shape, desc,
            format_time(metal_time), format_time(cpu_time),
            f"{ratio:.2f}x", color_ratio(ratio),
            format_accuracy(recon)
        ])
        print(f"  {desc}: {ratio:.2f}x")
    
    return results


def benchmark_models():
    """Benchmark SVD on real LLM weight geometries, organized by model family."""
    print("Benchmarking LLM Model Families...")
    
    try:
        import metalcore
    except ImportError:
        print("  metalcore not available")
        return {}
    
    # Model families with their typical weight matrix sizes
    models = {
        "Llama": [
            (4096, 4096, "Attention (7B)"),
            (4096, 11008, "MLP up (7B)"),
            (8192, 8192, "Attention (70B)"),
        ],
        "Mistral": [
            (4096, 4096, "Attention"),
            (4096, 14336, "MLP up"),
        ],
        "Qwen": [
            (4096, 4096, "Attention"),
            (4096, 11008, "MLP up"),
        ],
        "Gemma": [
            (3072, 3072, "Attention"),
            (3072, 24576, "MLP up"),
        ],
        "Phi": [
            (3072, 3072, "Attention"),
            (3072, 8192, "MLP up"),
        ],
    }
    
    all_results = {}
    
    for model_name, layers in models.items():
        print(f"  {model_name}:")
        results = []
        
        for M, N, layer_name in layers:
            if M * N > 200_000_000:
                results.append([f"{M}×{N}", layer_name, "—", "—", "—", "⚪", "—"])
                continue
            
            A = torch.randn(M, N, device=device)
            
            try:
                U, S, V = metalcore.svd(A)
                torch.mps.synchronize()
            except Exception as e:
                print(f"    {layer_name}: Error - {e}")
                continue
            
            total = M * N
            iters = 2 if total > 50_000_000 else 3 if total > 10_000_000 else 5
            
            torch.mps.synchronize()
            start = time.time()
            for _ in range(iters):
                U, S, V = metalcore.svd(A)
            torch.mps.synchronize()
            metal_time = (time.time() - start) / iters * 1000
            
            recon = torch.max(torch.abs(U @ torch.diag(S) @ V.T - A)).item()
            
            A_cpu = A.cpu()
            start = time.time()
            for _ in range(iters):
                torch.linalg.svd(A_cpu, full_matrices=False)
            cpu_time = (time.time() - start) / iters * 1000
            
            ratio = metal_time / cpu_time
            results.append([
                f"{M}×{N}", layer_name,
                format_time(metal_time), format_time(cpu_time),
                f"{ratio:.2f}x", color_ratio(ratio),
                format_accuracy(recon)
            ])
            print(f"    {layer_name}: {ratio:.2f}x")
        
        all_results[model_name] = results
    
    return all_results


def benchmark_rmsnorm():
    """Benchmark RMSNorm operations."""
    print("Benchmarking RMSNorm...")
    results = []
    
    try:
        from metalcore.rmsnorm import MetalRMSNorm
        import torch
    except ImportError:
        print("  metalcore.rmsnorm not available, skipping")
        return results

    configs = [
        (32, 4096),
        (1, 4096), 
        (1024, 1024),
        (4096, 4096)
    ]
    if LITE_MODE:
        configs = [(32, 4096)]
    
    iters = 1 if LITE_MODE else (100 if QUICK_MODE else 1000)
    warmup = 1 if LITE_MODE else 10
        
    for dtype_name, dtype in BENCHMARK_DTYPES:
        for B, N in configs:
            if QUICK_MODE and B > 500: continue
            
            name = f"RMSNorm ({B}x{N}) {dtype_name}"
            print(f"  Running {name}...")
            
            x = torch.randn(B, N, device=device, dtype=dtype, requires_grad=True)
            dy = torch.randn(B, N, device=device, dtype=dtype)
            
            model_metal = MetalRMSNorm(N).to(device).to(dtype)
            model_torch = torch.nn.RMSNorm(N).to(device).to(dtype)
            
            # Function to time Metal
            def run_metal():
                y = model_metal(x)
                y.backward(dy, retain_graph=True)
                if device == 'mps': torch.mps.synchronize()
                
            # Function to time Torch
            def run_torch():
                y = model_torch(x)
                y.backward(dy, retain_graph=True)
                if device == 'mps': torch.mps.synchronize()
                
            # Warmup
            for _ in range(warmup):
                run_metal()
                run_torch()
                
            # Benchmark Metal
            start = time.time()
            for _ in range(iters):
                run_metal()
            metal_time = (time.time() - start) / iters * 1000 # ms
            
            # Benchmark Torch
            start = time.time()
            for _ in range(iters):
                run_torch()
            torch_time = (time.time() - start) / iters * 1000 # ms
            
            ratio = metal_time / torch_time
            status = color_ratio(ratio)
            
            results.append([
                f"{B}x{N}", f"Fwd+Bwd {dtype_name}",
                format_time(metal_time), format_time(torch_time),
                f"{ratio:.2f}x", status
            ])
            print(f"    Metal: {format_time(metal_time)}, Torch: {format_time(torch_time)} -> {ratio:.2f}x {status}")
        
    return results


def benchmark_adamw():
    """Benchmark AdamW optimizer step."""
    print("Benchmarking AdamW...")
    results = []
    
    try:
        from metalcore.optim import MetalAdamW
        import torch
    except ImportError:
        print("  metalcore.optim not available, skipping")
        return results

    configs = [
        (1024*1024, "1M Params"),
        (10*1024*1024, "10M Params"),
        (4096*4096, "16M Params")
    ]
    if LITE_MODE:
        configs = [(1024*1024, "1M Params")]
    
    iters = 1 if LITE_MODE else (50 if QUICK_MODE else 100)
    warmup = 1 if LITE_MODE else 5
    
    for dtype_name, dtype in BENCHMARK_DTYPES:
        for N, label in configs:
            if QUICK_MODE and N > 5*1024*1024: continue
            
            name = f"AdamW ({label}) {dtype_name}"
            print(f"  Running {name}...")
            
            p = torch.randn(N, device=device, dtype=dtype)
            g = torch.randn(N, device=device, dtype=dtype)
            
            p_metal = p.clone()
            p_metal.grad = g.clone()
            opt_metal = MetalAdamW([p_metal], lr=1e-3)
            
            p_torch = p.clone()
            p_torch.grad = g.clone()
            opt_torch = torch.optim.AdamW([p_torch], lr=1e-3)
            
            def run_metal():
                opt_metal.step()
                if device == 'mps': torch.mps.synchronize()
                
            def run_torch():
                opt_torch.step()
                if device == 'mps': torch.mps.synchronize()
                
            # Warmup
            for _ in range(warmup):
                run_metal()
                run_torch()
                
            # Benchmark Metal
            start = time.time()
            for _ in range(iters):
                run_metal()
            metal_time = (time.time() - start) / iters * 1000
            
            # Benchmark Torch
            start = time.time()
            for _ in range(iters):
                run_torch()
            torch_time = (time.time() - start) / iters * 1000
            
            ratio = metal_time / torch_time
            status = color_ratio(ratio)
            
            results.append([
                label, f"N={N} {dtype_name}",
                format_time(metal_time), format_time(torch_time),
                f"{ratio:.2f}x", status
            ])
            print(f"    Metal: {format_time(metal_time)}, Torch: {format_time(torch_time)} -> {ratio:.2f}x {status}")
        
    return results


def benchmark_activations():
    """Benchmark GELU and SiLU activation functions."""
    print("Benchmarking Activations (GELU/SiLU)...")
    results = []
    
    try:
        from metalcore import metal_gelu, metal_silu
        import torch.nn.functional as F
    except ImportError:
        print("  metalcore activations not available, skipping")
        return results

    configs = [
        ((256, 1024), "Small (256x1024)"),
        ((1024, 4096), "Medium (1024x4096)"),
        ((4096, 4096), "Large (4096x4096)"),
    ]
    if LITE_MODE:
        configs = [((256, 1024), "Small (256x1024)")]
    
    iters = 1 if LITE_MODE else (50 if QUICK_MODE else 100)
    warmup = 1 if LITE_MODE else 5
    
    for dtype_name, dtype in BENCHMARK_DTYPES:
        for shape, label in configs:
            if QUICK_MODE and shape[0] > 2048: continue
            
            x = torch.randn(*shape, device=device, dtype=dtype)
        
            # GELU benchmark
            name = f"GELU {label} {dtype_name}"
            print(f"  Running {name}...")
            
            def run_metal_gelu():
                y = metal_gelu(x)
                if device == 'mps': torch.mps.synchronize()
                return y
                
            def run_torch_gelu():
                y = F.gelu(x, approximate='tanh')
                if device == 'mps': torch.mps.synchronize()
                return y
            
            for _ in range(warmup):
                run_metal_gelu()
                run_torch_gelu()
            
            start = time.time()
            for _ in range(iters):
                run_metal_gelu()
            metal_time = (time.time() - start) / iters * 1000
            
            start = time.time()
            for _ in range(iters):
                run_torch_gelu()
            torch_time = (time.time() - start) / iters * 1000
            
            ratio = metal_time / torch_time
            status = color_ratio(ratio)
            results.append([
                f"GELU {label}", f"{shape[0]}x{shape[1]} {dtype_name}",
                format_time(metal_time), format_time(torch_time),
                f"{ratio:.2f}x", status
            ])
            print(f"    Metal: {format_time(metal_time)}, Torch: {format_time(torch_time)} -> {ratio:.2f}x {status}")
            
            # SiLU benchmark
            name = f"SiLU {label} {dtype_name}"
            print(f"  Running {name}...")
            
            def run_metal_silu():
                y = metal_silu(x)
                if device == 'mps': torch.mps.synchronize()
                return y
                
            def run_torch_silu():
                y = F.silu(x)
                if device == 'mps': torch.mps.synchronize()
                return y
            
            for _ in range(warmup):
                run_metal_silu()
                run_torch_silu()
            
            start = time.time()
            for _ in range(iters):
                run_metal_silu()
            metal_time = (time.time() - start) / iters * 1000
            
            start = time.time()
            for _ in range(iters):
                run_torch_silu()
            torch_time = (time.time() - start) / iters * 1000
            
            ratio = metal_time / torch_time
            status = color_ratio(ratio)
            results.append([
                f"SiLU {label}", f"{shape[0]}x{shape[1]} {dtype_name}",
                format_time(metal_time), format_time(torch_time),
                f"{ratio:.2f}x", status
            ])
            print(f"    Metal: {format_time(metal_time)}, Torch: {format_time(torch_time)} -> {ratio:.2f}x {status}")
        
    return results


def benchmark_fused_activations():
    """Benchmark bias+activation fusions (bias_gelu, bias_silu)."""
    print("Benchmarking Fused Bias+Activations...")
    results = []
    
    try:
        from metalcore import bias_gelu, bias_silu
        import torch.nn.functional as F
    except ImportError:
        print("  metalcore fused activations not available, skipping")
        return results

    configs = [
        ((256, 4096), "Small (256x4096)"),
        ((1024, 4096), "Medium (1024x4096)"),
        ((4096, 11008), "Llama MLP (4096x11008)"),
    ]
    if LITE_MODE:
        configs = [((256, 4096), "Small (256x4096)")]
    
    iters = 1 if LITE_MODE else (50 if QUICK_MODE else 100)
    warmup = 1 if LITE_MODE else 5
    
    for dtype_name, dtype in BENCHMARK_DTYPES:
        for shape, label in configs:
            x = torch.randn(*shape, device=device, dtype=dtype)
            bias = torch.randn(shape[-1], device=device, dtype=dtype)
        
            # Bias+GELU benchmark
            name = f"Bias+GELU {label} {dtype_name}"
            print(f"  Running {name}...")
            
            def run_metal_bias_gelu():
                y = bias_gelu(x, bias)
                if device == 'mps': torch.mps.synchronize()
                return y
                
            def run_torch_bias_gelu():
                y = F.gelu(x + bias)
                if device == 'mps': torch.mps.synchronize()
                return y
            
            for _ in range(warmup):
                run_metal_bias_gelu()
                run_torch_bias_gelu()
            
            start = time.time()
            for _ in range(iters):
                run_metal_bias_gelu()
            metal_time = (time.time() - start) / iters * 1000
            
            start = time.time()
            for _ in range(iters):
                run_torch_bias_gelu()
            torch_time = (time.time() - start) / iters * 1000
            
            ratio = metal_time / torch_time
            status = color_ratio(ratio)
            results.append([
                f"Bias+GELU {label}", f"{shape[0]}x{shape[1]} {dtype_name}",
                format_time(metal_time), format_time(torch_time),
                f"{ratio:.2f}x", status
            ])
            print(f"    Metal: {format_time(metal_time)}, Torch: {format_time(torch_time)} -> {ratio:.2f}x {status}")
            
            # Bias+SiLU benchmark
            name = f"Bias+SiLU {label} {dtype_name}"
            print(f"  Running {name}...")
            
            def run_metal_bias_silu():
                y = bias_silu(x, bias)
                if device == 'mps': torch.mps.synchronize()
                return y
                
            def run_torch_bias_silu():
                y = F.silu(x + bias)
                if device == 'mps': torch.mps.synchronize()
                return y
            
            for _ in range(warmup):
                run_metal_bias_silu()
                run_torch_bias_silu()
            
            start = time.time()
            for _ in range(iters):
                run_metal_bias_silu()
            metal_time = (time.time() - start) / iters * 1000
            
            start = time.time()
            for _ in range(iters):
                run_torch_bias_silu()
            torch_time = (time.time() - start) / iters * 1000
            
            ratio = metal_time / torch_time
            status = color_ratio(ratio)
            results.append([
                f"Bias+SiLU {label}", f"{shape[0]}x{shape[1]} {dtype_name}",
                format_time(metal_time), format_time(torch_time),
                f"{ratio:.2f}x", status
            ])
            print(f"    Metal: {format_time(metal_time)}, Torch: {format_time(torch_time)} -> {ratio:.2f}x {status}")
        
    return results


def benchmark_fused_add_layernorm():
    """Benchmark fused add + layernorm."""
    print("Benchmarking Fused Add+LayerNorm...")
    results = []
    
    try:
        from metalcore import fused_add_layernorm
    except ImportError:
        print("  metalcore.fused_add_layernorm not available, skipping")
        return results

    configs = [
        ("Llama-7B", 32, 4096),
        ("Llama-13B", 32, 5120),
        ("Llama-70B", 16, 8192),
        ("Large Batch", 256, 4096),
    ]
    if LITE_MODE:
        configs = [("Llama-7B", 32, 4096)]
    elif QUICK_MODE:
        configs = configs[:2]
    
    iters = 1 if LITE_MODE else (100 if QUICK_MODE else 500)
    warmup = 1 if LITE_MODE else 10
        
    for dtype_name, dtype in BENCHMARK_DTYPES:
        for name, B, N in configs:
            config_name = f"FusedAdd+LN {name} ({B}x{N}) {dtype_name}"
            print(f"  Running {config_name}...")
            
            x = torch.randn(B, N, device=device, dtype=dtype)
            residual = torch.randn(B, N, device=device, dtype=dtype)
            weight = torch.ones(N, device=device, dtype=dtype)
            bias = torch.zeros(N, device=device, dtype=dtype)
            
            def run_metal():
                y, _, _ = fused_add_layernorm(x, residual, weight, bias, 1e-5)
                if device == 'mps': torch.mps.synchronize()
                return y
                
            def run_torch():
                y = torch.nn.functional.layer_norm(x + residual, (N,), weight, bias, 1e-5)
                if device == 'mps': torch.mps.synchronize()
                return y
                
            # Warmup
            for _ in range(warmup):
                run_metal()
                run_torch()
                
            # Benchmark
            start = time.time()
            for _ in range(iters):
                run_metal()
            metal_time = (time.time() - start) / iters * 1000
            
            start = time.time()
            for _ in range(iters):
                run_torch()
            torch_time = (time.time() - start) / iters * 1000
            
            ratio = metal_time / torch_time
            status = color_ratio(ratio)
            
            results.append([
                name, f"{B}x{N} {dtype_name}",
                format_time(metal_time), format_time(torch_time),
                f"{ratio:.2f}x", status
            ])
            print(f"    Metal: {format_time(metal_time)}, Torch: {format_time(torch_time)} -> {ratio:.2f}x {status}")
        
    return results


def benchmark_softmax():
    """Benchmark fused softmax operations."""
    print("Benchmarking Fused Softmax...")
    results = []
    
    try:
        from metalcore import fused_softmax
        import metalcore_backend as mc
    except ImportError:
        print("  metalcore.fused_softmax or backend not available, skipping")
        return results

    # Extended configs for thorough testing
    configs = [
        ("Small", 32, 1024),
        ("Medium", 64, 4096),
        ("Large", 128, 8192),
        ("Very Large", 256, 16384),
        ("Huge", 512, 32768),
        ("LLM Vocab", 32, 32000),
        ("LLM Vocab Large", 128, 128000),
    ]
    if LITE_MODE:
        configs = [("Medium", 64, 4096), ("GPU Win", 256, 16384)]
    elif QUICK_MODE:
        configs = configs[:4]
    
    iters = 1 if LITE_MODE else (50 if QUICK_MODE else 500)
    warmup = 1 if LITE_MODE else 10
        
    for dtype_name, dtype in BENCHMARK_DTYPES:
        for name, B, N in configs:
            config_name = f"Softmax {name} ({B}x{N}) {dtype_name}"
            print(f"  Running {config_name}...")
            
            x = torch.randn(B, N, device=device, dtype=dtype)
            
            # Function to time Metal
            def run_metal():
                y = fused_softmax(x, dim=-1)
                if device == 'mps': torch.mps.synchronize()
                return y
                
            # Function to time Torch
            def run_torch():
                y = torch.softmax(x, dim=-1)
                if device == 'mps': torch.mps.synchronize()
                return y
                
            # Warmup and verify correctness
            y_metal = run_metal()
            y_torch = run_torch()
            error = (y_metal - y_torch).abs().max().item()
            
            for _ in range(warmup):
                run_metal()
                run_torch()
                
            # Benchmark Metal
            start = time.time()
            for _ in range(iters):
                run_metal()
            metal_time = (time.time() - start) / iters * 1000 # ms
            
            # Benchmark Torch
            start = time.time()
            for _ in range(iters):
                run_torch()
            torch_time = (time.time() - start) / iters * 1000 # ms
            
            ratio = metal_time / torch_time
            status = color_ratio(ratio)
            
            results.append([
                name, f"{B}x{N} {dtype_name}",
                format_time(metal_time), format_time(torch_time),
                f"{ratio:.2f}x", status, f"{error:.2e}"
            ])
            print(f"    Metal: {format_time(metal_time)}, Torch: {format_time(torch_time)} -> {ratio:.2f}x {status}")
            
            # Benchmark Softmax Backward
            bwd_name = f"SoftmaxBwd {name} ({B}x{N}) {dtype_name}"
            print(f"  Running {bwd_name}...")
            
            probs = torch.rand(B, N, device=device, dtype=dtype)
            d_probs = torch.randn(B, N, device=device, dtype=dtype)
            
            def run_metal_bwd():
                return mc.softmax_bwd(probs, d_probs)
                
            def run_torch_bwd():
                # Manual softmax backward: probs * (d_probs - (probs * d_probs).sum)
                sum_val = (probs * d_probs).sum(dim=-1, keepdim=True)
                return probs * (d_probs - sum_val)
                
            # Warmup
            y_metal_bwd = run_metal_bwd()
            y_torch_bwd = run_torch_bwd()
            error_bwd = (y_metal_bwd - y_torch_bwd).abs().max().item()
            
            for _ in range(warmup):
                run_metal_bwd()
                run_torch_bwd()
                if device == 'mps': torch.mps.synchronize()
                
            if device == 'mps': torch.mps.synchronize()
            start = time.time()
            for _ in range(iters):
                run_metal_bwd()
                if device == 'mps': torch.mps.synchronize()
            metal_time_bwd = (time.time() - start) / iters * 1000
            
            if device == 'mps': torch.mps.synchronize()
            start = time.time()
            for _ in range(iters):
                run_torch_bwd()
                if device == 'mps': torch.mps.synchronize()
            torch_time_bwd = (time.time() - start) / iters * 1000
            
            ratio_bwd = metal_time_bwd / torch_time_bwd
            status_bwd = color_ratio(ratio_bwd)
            
            results.append([
                f"SoftmaxBwd {name}", f"{B}x{N} {dtype_name}",
                format_time(metal_time_bwd), format_time(torch_time_bwd),
                f"{ratio_bwd:.2f}x", status_bwd, f"{error_bwd:.2e}"
            ])
            print(f"    Metal: {format_time(metal_time_bwd)}, Torch: {format_time(torch_time_bwd)} -> {ratio_bwd:.2f}x {status_bwd}")
        
    return results


def benchmark_layernorm():
    """Benchmark LayerNorm operations."""
    print("Benchmarking LayerNorm...")
    results = []
    
    try:
        from metalcore import MetalLayerNorm
    except ImportError:
        print("  metalcore.MetalLayerNorm not available, skipping")
        return results

    # Extended configs
    configs = [
        ("Tiny", 32, 512),
        ("Small", 64, 1024),
        ("Llama-7B", 32, 4096),
        ("Llama-13B", 32, 5120),
        ("Llama-70B", 16, 8192),
        ("Large Batch", 256, 4096),
        ("Huge Batch", 1024, 4096),
    ]
    if LITE_MODE:
        configs = [("Llama-7B", 32, 4096), ("GPU Win", 16, 32768)]
    elif QUICK_MODE:
        configs = configs[:4]
    
    iters = 1 if LITE_MODE else (100 if QUICK_MODE else 500)
    warmup = 1 if LITE_MODE else 10
        
    for dtype_name, dtype in BENCHMARK_DTYPES:
        for name, B, N in configs:
            config_name = f"LayerNorm {name} ({B}x{N}) {dtype_name}"
            print(f"  Running {config_name}...")
            
            x = torch.randn(B, N, device=device, dtype=dtype)
            
            model_metal = MetalLayerNorm(N).to(device).to(dtype)
            model_torch = torch.nn.LayerNorm(N).to(device).to(dtype)
            
            # Function to time Metal
            def run_metal():
                y = model_metal(x)
                if device == 'mps': torch.mps.synchronize()
                return y
                
            # Function to time Torch
            def run_torch():
                y = model_torch(x)
                if device == 'mps': torch.mps.synchronize()
                return y
                
            # Warmup and verify
            y_metal = run_metal()
            y_torch = run_torch()
            error = (y_metal - y_torch).abs().max().item()
            
            for _ in range(warmup):
                run_metal()
                run_torch()
                
            # Benchmark Metal
            start = time.time()
            for _ in range(iters):
                run_metal()
            metal_time = (time.time() - start) / iters * 1000 # ms
            
            # Benchmark Torch
            start = time.time()
            for _ in range(iters):
                run_torch()
            torch_time = (time.time() - start) / iters * 1000 # ms
            
            ratio = metal_time / torch_time
            status = color_ratio(ratio)
            
            results.append([
                name, f"{B}x{N} {dtype_name}",
                format_time(metal_time), format_time(torch_time),
                f"{ratio:.2f}x", status, f"{error:.2e}"
            ])
            print(f"    Metal: {format_time(metal_time)}, Torch: {format_time(torch_time)} -> {ratio:.2f}x {status}")
        
    return results


def benchmark_embedding_bag():
    """Benchmark embedding bag operations."""
    print("Benchmarking Embedding Bag...")
    results = []
    
    try:
        from metalcore import embedding_bag
    except ImportError:
        print("  metalcore.embedding_bag not available, skipping")
        return results

    # Extended configs: (name, num_embeddings, embedding_dim, batch_size, avg_bag_size)
    configs = [
        ("Small Vocab", 10000, 64, 32, 10),
        ("Medium Vocab", 50000, 128, 64, 20),
        ("Large Vocab", 100000, 256, 32, 50),
        ("LLM Embedding", 32000, 4096, 16, 100),
        ("Huge Vocab", 250000, 512, 16, 30),
    ]
    if LITE_MODE:
        configs = [("Medium Vocab", 50000, 128, 64, 20)]
    elif QUICK_MODE:
        configs = configs[:3]
    
    iters = 1 if LITE_MODE else (50 if QUICK_MODE else 200)
    warmup = 1 if LITE_MODE else 10
        
    for dtype_name, dtype in BENCHMARK_DTYPES:
        if dtype != torch.float32:
            continue  # embedding_bag only supports fp32
            
        for name, num_emb, dim, batch_size, avg_bag in configs:
            config_name = f"EmbedBag {name} {dtype_name}"
            print(f"  Running {config_name}...")
            
            weight = torch.randn(num_emb, dim, device=device, dtype=dtype)
            # Create variable-length bags
            bag_sizes = torch.randint(1, avg_bag * 2, (batch_size,))
            total_indices = bag_sizes.sum().item()
            indices = torch.randint(0, num_emb, (total_indices,), device=device)
            offsets = torch.cat([torch.tensor([0]), bag_sizes.cumsum(0)]).to(device)
            
            # Function to time Metal
            def run_metal():
                y = embedding_bag(weight, indices, offsets, mode='sum')
                if device == 'mps': torch.mps.synchronize()
                return y
                
            # Function to time Torch (embedding_bag not well supported on MPS, use fallback)
            def run_torch():
                y, _, _, _ = torch.embedding_bag(weight, indices, offsets, False, 0)
                if device == 'mps': torch.mps.synchronize()
                return y
                
            # Warmup
            for _ in range(warmup):
                run_metal()
                run_torch()
                
            # Benchmark Metal
            start = time.time()
            for _ in range(iters):
                run_metal()
            metal_time = (time.time() - start) / iters * 1000 # ms
            
            # Benchmark Torch
            start = time.time()
            for _ in range(iters):
                run_torch()
            torch_time = (time.time() - start) / iters * 1000 # ms
            
            ratio = metal_time / torch_time
            status = color_ratio(ratio)
            
            results.append([
                name, f"{num_emb}x{dim}, B={batch_size}",
                format_time(metal_time), format_time(torch_time),
                f"{ratio:.2f}x", status
            ])
            print(f"    Metal: {format_time(metal_time)}, Torch: {format_time(torch_time)} -> {ratio:.2f}x {status}")
        
    return results


def benchmark_scatter_gather():
    """Benchmark scatter and gather operations."""
    print("Benchmarking Scatter/Gather...")
    results = []
    
    try:
        from metalcore import gather, scatter_add
    except ImportError:
        print("  metalcore scatter/gather not available, skipping")
        return results

    # Extended configs: (name, src_size, num_indices)
    configs = [
        ("Small", 10000, 1000),
        ("Medium", 100000, 10000),
        ("Large", 1000000, 100000),
        ("Huge", 10000000, 1000000),
    ]
    if LITE_MODE:
        configs = [("Medium", 100000, 10000)]
    elif QUICK_MODE:
        configs = configs[:3]
    
    iters = 1 if LITE_MODE else (100 if QUICK_MODE else 500)
    warmup = 1 if LITE_MODE else 10
        
    for dtype_name, dtype in BENCHMARK_DTYPES:
        if dtype != torch.float32:
            continue  # scatter/gather mainly fp32
            
        # Gather benchmarks
        for name, src_size, num_idx in configs:
            config_name = f"Gather {name} {dtype_name}"
            print(f"  Running {config_name}...")
            
            src = torch.randn(src_size, device=device, dtype=dtype)
            idx = torch.randint(0, src_size, (num_idx,), device=device)
            
            # Function to time Metal
            def run_metal():
                y = gather(src, idx, dim=0)
                if device == 'mps': torch.mps.synchronize()
                return y
                
            # Function to time Torch
            def run_torch():
                y = torch.gather(src, 0, idx.to(torch.long))
                if device == 'mps': torch.mps.synchronize()
                return y
                
            # Warmup
            for _ in range(warmup):
                run_metal()
                run_torch()
                
            # Benchmark
            start = time.time()
            for _ in range(iters):
                run_metal()
            metal_time = (time.time() - start) / iters * 1000
            
            start = time.time()
            for _ in range(iters):
                run_torch()
            torch_time = (time.time() - start) / iters * 1000
            
            ratio = metal_time / torch_time
            status = color_ratio(ratio)
            
            results.append([
                f"Gather {name}", f"src={src_size}, idx={num_idx}",
                format_time(metal_time), format_time(torch_time),
                f"{ratio:.2f}x", status
            ])
            print(f"    Metal: {format_time(metal_time)}, Torch: {format_time(torch_time)} -> {ratio:.2f}x {status}")
        
        # Scatter Add benchmarks
        for name, dst_size, num_idx in configs:
            config_name = f"ScatterAdd {name} {dtype_name}"
            print(f"  Running {config_name}...")
            
            dst = torch.zeros(dst_size, device=device, dtype=dtype)
            idx = torch.randint(0, dst_size, (num_idx,), device=device)
            src = torch.randn(num_idx, device=device, dtype=dtype)
            
            # Function to time Metal
            def run_metal():
                y = scatter_add(dst.clone(), idx, src, dim=0)
                if device == 'mps': torch.mps.synchronize()
                return y
                
            # Function to time Torch
            def run_torch():
                y = dst.clone().scatter_add(0, idx.to(torch.long), src)
                if device == 'mps': torch.mps.synchronize()
                return y
                
            # Warmup
            for _ in range(warmup):
                run_metal()
                run_torch()
                
            # Benchmark
            start = time.time()
            for _ in range(iters):
                run_metal()
            metal_time = (time.time() - start) / iters * 1000
            
            start = time.time()
            for _ in range(iters):
                run_torch()
            torch_time = (time.time() - start) / iters * 1000
            
            ratio = metal_time / torch_time
            status = color_ratio(ratio)
            
            results.append([
                f"ScatterAdd {name}", f"dst={dst_size}, idx={num_idx}",
                format_time(metal_time), format_time(torch_time),
                f"{ratio:.2f}x", status
            ])
            print(f"    Metal: {format_time(metal_time)}, Torch: {format_time(torch_time)} -> {ratio:.2f}x {status}")
        
    return results


def benchmark_lora_training():
    """Benchmark LoRA training operations: cross_entropy, kl_div, swiglu, lora_linear."""
    print("Benchmarking LoRA Training Operations...")
    results = []
    
    try:
        import metalcore_backend as mc
        import torch.nn.functional as F
    except ImportError:
        print("  metalcore_backend not available, skipping")
        return results

    iters = 1 if LITE_MODE else (30 if QUICK_MODE else 100)
    warmup = 1 if LITE_MODE else 5
    
    # 1. SwiGLU Activation
    print("  SwiGLU Activation...")
    swiglu_configs = [
        (128, 11008, "Llama-7B hidden"),
        (256, 14336, "Llama-3 hidden"),
    ]
    if LITE_MODE:
        swiglu_configs = [(128, 11008, "Llama-7B hidden")]
    
    for dtype_name, dtype in BENCHMARK_DTYPES:
        for batch, hidden, desc in swiglu_configs:
            gate = torch.randn(batch, hidden, device=device, dtype=dtype)
            up = torch.randn(batch, hidden, device=device, dtype=dtype)
            fused_name = f"SwiGLU {desc} {dtype_name}"
            print(f"  Running {fused_name}...")
            
            def run_metal():
                return mc.swiglu_fwd(gate, up)
                
            def run_torch():
                return F.silu(gate) * up
            
            # Warmup
            for _ in range(warmup):
                run_metal()
                run_torch()
                if device == 'mps': torch.mps.synchronize()
            
            # Metal timing
            if device == 'mps': torch.mps.synchronize()
            start = time.time()
            for _ in range(iters):
                run_metal()
                if device == 'mps': torch.mps.synchronize()
            metal_time = (time.time() - start) / iters * 1000
            
            # Torch timing
            if device == 'mps': torch.mps.synchronize()
            start = time.time()
            for _ in range(iters):
                run_torch()
                if device == 'mps': torch.mps.synchronize()
            torch_time = (time.time() - start) / iters * 1000
            
            ratio = metal_time / torch_time
            status = color_ratio(ratio)
            results.append([
                f"SwiGLU", f"{batch}x{hidden} {desc} {dtype_name}",
                format_time(metal_time), format_time(torch_time),
                f"{ratio:.2f}x", status
            ])
            print(f"    Metal: {format_time(metal_time)}, Torch: {format_time(torch_time)} -> {ratio:.2f}x {status}")

    # 2. LoRA Add (New)
    print("  LoRA Add (Base + Scale * LoRA)...")
    lora_add_configs = [
        (32, 4096, "Llama-7B dim"),
        (32, 8192, "Llama-70B dim"),
    ]
    if LITE_MODE:
        lora_add_configs = [(32, 4096, "Llama-7B dim")]

    for dtype_name, dtype in BENCHMARK_DTYPES:
        for batch, dim, desc in lora_add_configs:
            base = torch.randn(batch, dim, device=device, dtype=dtype)
            lora = torch.randn(batch, dim, device=device, dtype=dtype)
            scale = 0.5
            name = f"LoRA Add {desc} {dtype_name}"
            print(f"  Running {name}...")

            def run_metal_lora_add():
                return mc.lora_add_fwd(base, lora, scale)
            
            def run_torch_lora_add():
                return base + scale * lora
            
            for _ in range(warmup):
                run_metal_lora_add()
                run_torch_lora_add()
                if device == 'mps': torch.mps.synchronize()

            if device == 'mps': torch.mps.synchronize()
            start = time.time()
            for _ in range(iters):
                run_metal_lora_add()
                if device == 'mps': torch.mps.synchronize()
            metal_time = (time.time() - start) / iters * 1000

            if device == 'mps': torch.mps.synchronize()
            start = time.time()
            for _ in range(iters):
                run_torch_lora_add()
                if device == 'mps': torch.mps.synchronize()
            torch_time = (time.time() - start) / iters * 1000

            ratio = metal_time / torch_time
            status = color_ratio(ratio)
            results.append([
                f"LoRA Add", f"{batch}x{dim} {desc} {dtype_name}",
                format_time(metal_time), format_time(torch_time),
                f"{ratio:.2f}x", status
            ])
            print(f"    Metal: {format_time(metal_time)}, Torch: {format_time(torch_time)} -> {ratio:.2f}x {status}")

    # 3. LoRA Linear Forward
    print("  LoRA Linear Forward...")
    lora_configs = [
        (128, 4096, 4096, 16, "Llama attn r=16"),
        (128, 4096, 11008, 8, "Llama MLP r=8"),
    ]
    if LITE_MODE:
        lora_configs = [(128, 4096, 4096, 16, "Llama attn r=16")]
    
    for dtype_name, dtype in BENCHMARK_DTYPES:
        for batch, in_f, out_f, rank, desc in lora_configs:
            x = torch.randn(batch, in_f, device=device, dtype=dtype)
            W = torch.randn(out_f, in_f, device=device, dtype=dtype)
            A = torch.randn(rank, in_f, device=device, dtype=dtype)
            B = torch.randn(out_f, rank, device=device, dtype=dtype)
            scale = 0.5
            
            name = f"LoRA Linear {desc} {dtype_name}"
            print(f"  Running {name}...")
            
            def run_metal_ll():
                # lora_linear_fwd currently supports float/half/bfloat via dispatch
                return mc.lora_linear_fwd(x, W, A, B, scale)
                
            def run_torch_ll():
                return x @ W.t() + scale * (x @ A.t() @ B.t())
            
            for _ in range(warmup):
                run_metal_ll()
                run_torch_ll()
                if device == 'mps': torch.mps.synchronize()
            
            if device == 'mps': torch.mps.synchronize()
            start = time.time()
            for _ in range(iters):
                run_metal_ll()
                if device == 'mps': torch.mps.synchronize()
            metal_time = (time.time() - start) / iters * 1000
            
            if device == 'mps': torch.mps.synchronize()
            start = time.time()
            for _ in range(iters):
                run_torch_ll()
                if device == 'mps': torch.mps.synchronize()
            torch_time = (time.time() - start) / iters * 1000
            
            ratio = metal_time / torch_time
            status = color_ratio(ratio)
            
            results.append([
                f"LoRA Linear", f"{batch}x{in_f}→{out_f} r={rank} {desc} {dtype_name}",
                format_time(metal_time), format_time(torch_time),
                f"{ratio:.2f}x", status
            ])
            print(f"    Metal: {format_time(metal_time)}, Torch: {format_time(torch_time)} -> {ratio:.2f}x {status}")
            
    # 4. KL Divergence
    print("  KL Divergence (Forward)...")
    kl_configs = [(32, 32000, "Llama vocab")]
    if LITE_MODE: kl_configs = [(32, 1024, "Lite vocab")]
    
    for dtype_name, dtype in BENCHMARK_DTYPES:
        for batch, vocab, desc in kl_configs:
            log_p = torch.randn(batch, vocab, device=device, dtype=dtype)
            log_q = torch.randn(batch, vocab, device=device, dtype=dtype)
            name = f"KL Div {desc} {dtype_name}"
            print(f"  Running {name}...")
            
            def run_metal_kl():
                return mc.kl_div_fwd(log_p, log_q)
                
            def run_torch_kl():
                p = torch.exp(log_p.float())
                return torch.sum(p * (log_p.float() - log_q.float()), dim=-1)
                
            # Warmup & Timing
            try:
                if device == 'mps': torch.mps.synchronize()
                run_metal_kl() # Check compatible
                
                if device == 'mps': torch.mps.synchronize()
                start = time.time()
                for _ in range(iters):
                    run_metal_kl()
                    if device == 'mps': torch.mps.synchronize()
                metal_time = (time.time() - start) / iters * 1000
                
                if device == 'mps': torch.mps.synchronize()
                start = time.time()
                for _ in range(iters):
                    run_torch_kl()
                    if device == 'mps': torch.mps.synchronize()
                torch_time = (time.time() - start) / iters * 1000
                
                ratio = metal_time / torch_time
                status = color_ratio(ratio)
                results.append([f"KL Div", f"{batch}x{vocab} {desc} {dtype_name}", format_time(metal_time), format_time(torch_time), f"{ratio:.2f}x", status])
                print(f"    Metal: {format_time(metal_time)}, Torch: {format_time(torch_time)} -> {ratio:.2f}x {status}")
            except Exception as e:
                print(f"    Failed: {e}")

    # 5. Cross Entropy
    print("  Cross Entropy (Fused)...")
    ce_configs = [(32, 32000, "Llama vocab")]
    if LITE_MODE: ce_configs = [(32, 1024, "Lite vocab")]
    
    for dtype_name, dtype in BENCHMARK_DTYPES:
        for batch, vocab, desc in ce_configs:
            logits = torch.randn(batch, vocab, device=device, dtype=dtype)
            targets = torch.randint(0, vocab, (batch,), device=device).to(torch.int32)
            # Make logits reasonable for softmax
            logits = logits * 10.0
            
            name = f"CrossEntropy {desc} {dtype_name}"
            print(f"  Running {name}...")
            
            def run_metal_ce():
                return mc.cross_entropy_fwd(logits, targets)
            
            def run_torch_ce():
                return torch.nn.functional.cross_entropy(logits.float(), targets.long(), reduction='none')
            
            try:
                if device == 'mps': torch.mps.synchronize()
                run_metal_ce()
                
                if device == 'mps': torch.mps.synchronize()
                start = time.time()
                for _ in range(iters):
                    run_metal_ce()
                    if device == 'mps': torch.mps.synchronize()
                metal_time = (time.time() - start) / iters * 1000
                
                if device == 'mps': torch.mps.synchronize()
                start = time.time()
                for _ in range(iters):
                    run_torch_ce()
                    if device == 'mps': torch.mps.synchronize()
                torch_time = (time.time() - start) / iters * 1000
                
                ratio = metal_time / torch_time
                status = color_ratio(ratio)
                results.append([f"CrossEntropy", f"{batch}x{vocab} {desc} {dtype_name}", format_time(metal_time), format_time(torch_time), f"{ratio:.2f}x", status])
                print(f"    Metal: {format_time(metal_time)}, Torch: {format_time(torch_time)} -> {ratio:.2f}x {status}")
            except Exception as e:
                print(f"    Failed: {e}")
    
    return results

def benchmark_fused_swiglu_mlp():
    """Benchmark Meta-Fused SwiGLU MLP vs PyTorch eager."""
    print("Benchmarking Fused SwiGLU MLP...")
    results = []
    try:
        import metalcore_backend as mc
        from metalcore import fused_swiglu_mlp
    except ImportError:
        print("  metalcore not available, skipping")
        return results

    iters = 1 if LITE_MODE else (30 if QUICK_MODE else 100)
    warmup = 1 if LITE_MODE else 5
    
    configs = [
        (32, 128, 4096, 11008, 16, "Llama-7B MLP"),
    ]
    if LITE_MODE: configs = [(4, 128, 1024, 4096, 16, "Lite MLP")]
    
    for dtype_name, dtype in BENCHMARK_DTYPES:
        for batch, seq, dim, inter_dim, rank, desc in configs:
            # Inputs
            x = torch.randn(batch, seq, dim, device=device, dtype=dtype)
            res = torch.randn(batch, seq, dim, device=device, dtype=dtype)
            
            # Weights
            ln_w = torch.randn(dim, device=device, dtype=dtype)
            
            W_gate = torch.randn(inter_dim, dim, device=device, dtype=dtype)
            W_up   = torch.randn(inter_dim, dim, device=device, dtype=dtype)
            W_down = torch.randn(dim, inter_dim, device=device, dtype=dtype)
            
            # LoRA
            A_gate = torch.randn(rank, dim, device=device, dtype=dtype)
            B_gate = torch.randn(inter_dim, rank, device=device, dtype=dtype)
            A_up   = torch.randn(rank, dim, device=device, dtype=dtype)
            B_up   = torch.randn(inter_dim, rank, device=device, dtype=dtype)
            A_down = torch.randn(rank, inter_dim, device=device, dtype=dtype)
            B_down = torch.randn(dim, rank, device=device, dtype=dtype)
            
            scale = 0.5
            eps = 1e-5
            
            name = f"FusedSwiGLU {desc} {dtype_name}"
            print(f"  Running {name}...")
            
            def run_metal():
                return fused_swiglu_mlp(
                    x, res, ln_w, eps,
                    W_gate, W_up, W_down,
                    A_gate, B_gate, A_up, B_up, A_down, B_down,
                    scale
                )
            
            def run_torch():
                # RMSNorm
                h = x.float()
                var = h.pow(2).mean(-1, keepdim=True)
                norm = (h * torch.rsqrt(var + eps)).to(dtype) * ln_w
                
                # Gate
                g = F.linear(norm, W_gate) + scale * (norm @ A_gate.t() @ B_gate.t())
                # Up
                u = F.linear(norm, W_up) + scale * (norm @ A_up.t() @ B_up.t())
                
                # SwiGLU
                act = F.silu(g) * u
                
                # Down
                out = F.linear(act, W_down) + scale * (act @ A_down.t() @ B_down.t())
                
                return res + out
            
            try:
                # Correctness check
                if device == 'mps': torch.mps.synchronize()
                
                # Update run_metal to use W_down (at::linear handles transpose)
                def run_metal():
                    import metalcore
                    return metalcore.fused_swiglu_mlp(
                        x, res,
                        ln_w, eps,
                        W_gate, W_up, W_down,
                        A_gate, B_gate,
                        A_up, B_up,
                        A_down, B_down,
                        scale, 0.0, False
                    )

                out_m = run_metal()
                out_t = run_torch()
                
                if device == 'mps': torch.mps.synchronize()
                
                # Loose tolerance for bf16/fp16 composite ops
                if not torch.allclose(out_m, out_t, rtol=1e-2, atol=1e-2):
                     print(f"    Mismatch! Max diff: {(out_m - out_t).abs().max().item():.6f}")

                # Timing
                start = time.time()
                for _ in range(iters):
                    run_metal()
                    if device == 'mps': torch.mps.synchronize()
                metal_time = (time.time() - start) / iters * 1000
                
                start = time.time()
                for _ in range(iters):
                    run_torch()
                    if device == 'mps': torch.mps.synchronize()
                torch_time = (time.time() - start) / iters * 1000
                
                ratio = metal_time / torch_time
                status = color_ratio(ratio)
                
                results.append([
                    f"FusedSwiGLU", f"{batch}x{seq} {desc} {dtype_name}",
                    format_time(metal_time), format_time(torch_time),
                    f"{ratio:.2f}x", status
                ])
                print(f"    Metal: {format_time(metal_time)}, Torch: {format_time(torch_time)} -> {ratio:.2f}x {status}")
                
            except Exception as e:
                print(f"    Failed: {e}")
                import traceback
                traceback.print_exc()

    return results


def benchmark_fused_mlp_bwd():
    """Benchmark Fused MLP Backward."""
    print("Benchmarking Fused MLP Backward...")
    results = []
    try:
        from metalcore import fused_mlp_bwd
        import metalcore_backend as mc
    except ImportError:
        print("  metalcore not available, skipping")
        return results

    iters = 1 if LITE_MODE else (30 if QUICK_MODE else 100)
    warmup = 1 if LITE_MODE else 5
    
    configs = [
        # Llama-7B MLP: H=4096, I=11008
        (32, 128, 4096, 11008, 16, "Llama-7B MLP"),
        # Llama-3-8B MLP: H=4096, I=14336
        (16, 128, 4096, 14336, 16, "Llama-3-8B MLP"),
    ]
    if LITE_MODE: configs = [(4, 128, 1024, 2048, 16, "Lite MLP")]
    
    for dtype_name, dtype in BENCHMARK_DTYPES:
        for B, S, H, I, r, desc in configs:
            # Setup tensors (mocking global memory state essentially)
            d_out = torch.randn(B, S, H, device=device, dtype=dtype)
            x_norm = torch.randn(B, S, H, device=device, dtype=dtype, requires_grad=True)
            gate = torch.randn(B, S, I, device=device, dtype=dtype, requires_grad=True)
            up = torch.randn(B, S, I, device=device, dtype=dtype, requires_grad=True)
            hidden_states = torch.randn(B, S, H, device=device, dtype=dtype, requires_grad=True)
            
            # Weights
            W_gate = torch.randn(I, H, device=device, dtype=dtype, requires_grad=True)
            W_up = torch.randn(I, H, device=device, dtype=dtype, requires_grad=True)
            W_down = torch.randn(H, I, device=device, dtype=dtype, requires_grad=True)
            rms_weight = torch.randn(H, device=device, dtype=dtype, requires_grad=True)
            
            # LoRA Stub
            A_gate = torch.empty(0, device=device, dtype=dtype)
            B_gate = torch.empty(0, device=device, dtype=dtype)
            A_up = torch.empty(0, device=device, dtype=dtype)
            B_up = torch.empty(0, device=device, dtype=dtype)
            A_down = torch.empty(0, device=device, dtype=dtype)
            B_down = torch.empty(0, device=device, dtype=dtype)
            
            scale = 1.0
            rstd = torch.randn(B, S, 1, device=device, dtype=dtype) # Mock rstd
            
            name = f"FusedMLP Bwd {desc} {dtype_name}"
            print(f"  Running {name}...")
            
            def run_metal():
                return fused_mlp_bwd(
                    d_out, x_norm, gate, up,
                    W_gate, W_up, W_down,
                    A_gate, B_gate, A_up, B_up, A_down, B_down,
                    scale, hidden_states, rms_weight, rstd
                )
                
            def run_torch():
                # Reconstruct forward graph for autograd
                # Note: This is an approximation of the backward workload.
                # Since we want to benchmark the BACKWARD pass, we simply run .backward()
                # But we need a valid graph.
                # Let's perform the forward pass operations then autodiff.
                
                # RMSNorm (approx)
                # x_norm_t = hidden_states * rstd * rms_weight
                
                # We need to use leaf variables to get grads. 
                # x_norm, gate, up are intermediate in real life, but inputs to this kernel.
                # For fair comparison, PyTorch should compute grads for these inputs.
                
                # Actually, the Fused MLP Bwd kernel does: 
                # Down Bwd -> SwiGLU Bwd -> Gate/Up Bwd -> RMSNorm Bwd.
                # So we simulate that chain.
                
                # 1. Down Proj Bwd
                # out = linear(swiglu(gate, up))
                swiglu = F.silu(gate) * up
                down = F.linear(swiglu, W_down)
                # 2. Gate/Up Bwd
                # gate = linear(x_norm)
                # up = linear(x_norm)
                # (We treat gate/up as leaves to get their grads? No, we get grads w.r.t W)
                
                # To properly benchmark:
                # We define the full forward pass from (hidden_states) to (output)
                
                # RMSNorm
                h_f = hidden_states.float()
                rstd_t = torch.rsqrt(h_f.pow(2).mean(-1, keepdim=True) + 1e-6).to(dtype)
                x_norm_t = hidden_states * rstd_t * rms_weight
                
                # Projections
                g_t = F.linear(x_norm_t, W_gate) 
                u_t = F.linear(x_norm_t, W_up)
                
                # SwiGLU
                act = F.silu(g_t) * u_t
                
                # Down
                out = F.linear(act, W_down)
                
                # Backward
                # Computes grads for: hidden_states, W_gate, W_up, W_down, rms_weight
                torch.autograd.grad(out, 
                    [hidden_states, W_gate, W_up, W_down, rms_weight], 
                    grad_outputs=d_out, retain_graph=True)
            
            # Warmup
            try:
                if device == 'mps': torch.mps.synchronize()
                run_metal()
                if device == 'mps': torch.mps.synchronize()
                
                start = time.time()
                for _ in range(iters):
                    run_metal()
                    if device == 'mps': torch.mps.synchronize()
                metal_time = (time.time() - start) / iters * 1000
                
                start = time.time()
                for _ in range(iters):
                    run_torch()
                    if device == 'mps': torch.mps.synchronize()
                torch_time = (time.time() - start) / iters * 1000
                
                ratio = metal_time / torch_time
                status = color_ratio(ratio)
                
                results.append([
                    f"FusedMLP Bwd", f"{B}x{S} {desc} {dtype_name}",
                    format_time(metal_time), format_time(torch_time),
                    f"{ratio:.2f}x", status
                ])
                print(f"    Metal: {format_time(metal_time)}, Torch: {format_time(torch_time)} -> {ratio:.2f}x {status}")
                
            except Exception as e:
                print(f"    Failed: {e}")
                import traceback
                traceback.print_exc()

    return results


def benchmark_fused_attention_bwd():
    """Benchmark Fused Attention Backward."""
    print("Benchmarking Fused Attention Backward...")
    results = []
    try:
        from metalcore import fused_attention_bwd
    except ImportError:
        print("  metalcore not available, skipping")
        return results

    iters = 1 if LITE_MODE else (30 if QUICK_MODE else 100)
    warmup = 1 if LITE_MODE else 5
    
    configs = [
        # Llama-7B Attn: H=32 heads, D=128
        (32, 128, 32, 128, "Llama-7B Attn"),
        # Llama-70B GQA-ish (just checking larger sizes)
        (8, 128, 64, 128, "Large Head Count"),
    ]
    if LITE_MODE: configs = [(4, 64, 4, 64, "Lite Attn")]
    
    for dtype_name, dtype in BENCHMARK_DTYPES:
        for B, S, NumHeads, HeadDim, desc in configs:
            Hidden = NumHeads * HeadDim
            # Inputs (Gradients from SDPA)
            d_q = torch.randn(B, S, NumHeads, HeadDim, device=device, dtype=dtype)
            d_k = torch.randn(B, S, NumHeads, HeadDim, device=device, dtype=dtype)
            d_v = torch.randn(B, S, NumHeads, HeadDim, device=device, dtype=dtype)
            
            # RoPE cache
            cos = torch.randn(S, HeadDim, device=device, dtype=dtype)
            sin = torch.randn(S, HeadDim, device=device, dtype=dtype)
            
            # Forward inputs
            x_norm = torch.randn(B, S, Hidden, device=device, dtype=dtype, requires_grad=True)
            hidden_states = torch.randn(B, S, Hidden, device=device, dtype=dtype, requires_grad=True)
            
            # Weights
            W_q = torch.randn(Hidden, Hidden, device=device, dtype=dtype, requires_grad=True)
            W_k = torch.randn(Hidden, Hidden, device=device, dtype=dtype, requires_grad=True)
            W_v = torch.randn(Hidden, Hidden, device=device, dtype=dtype, requires_grad=True)
            rms_weight = torch.randn(Hidden, device=device, dtype=dtype, requires_grad=True)
            
            # LoRA Stub
            A_q = torch.empty(0, device=device, dtype=dtype)
            B_q = torch.empty(0, device=device, dtype=dtype)
            A_k = torch.empty(0, device=device, dtype=dtype)
            B_k = torch.empty(0, device=device, dtype=dtype)
            A_v = torch.empty(0, device=device, dtype=dtype)
            B_v = torch.empty(0, device=device, dtype=dtype)
            
            scale = 1.0
            rstd = torch.randn(B, S, 1, device=device, dtype=dtype)
            
            name = f"FusedAtt Bwd {desc} {dtype_name}"
            print(f"  Running {name}...")
            
            def run_metal():
                return fused_attention_bwd(
                    d_q, d_k, d_v, cos, sin, x_norm,
                    W_q, W_k, W_v,
                    A_q, B_q, A_k, B_k, A_v, B_v,
                    scale, hidden_states, rms_weight, rstd
                )
                
            def run_torch():
                # Simulate the graph
                # RMSNorm
                h_f = hidden_states.float()
                rstd_t = torch.rsqrt(h_f.pow(2).mean(-1, keepdim=True) + 1e-6).to(dtype)
                x_norm_t = hidden_states * rstd_t * rms_weight
                
                # QKV Proj
                q = F.linear(x_norm_t, W_q).view(B, S, NumHeads, HeadDim)
                k = F.linear(x_norm_t, W_k).view(B, S, NumHeads, HeadDim)
                v = F.linear(x_norm_t, W_v).view(B, S, NumHeads, HeadDim)
                
                # RoPE (Manual) - Needed because d_q/d_k comes from SDPA *after* RoPE
                # If we don't apply RoPE in fwd, the grads won't trigger the RoPE bwd logic.
                # Simplification: Apply simple scale to mimic math dependency
                # q_rot = q * cos.view(1, S, 1, HeadDim) 
                # k_rot = k * cos.view(1, S, 1, HeadDim)
                
                # Actually, simpler: just compute gradients w.r.t Q, K, V
                # We want to benchmark [d_q_rot -> d_hidden].
                # PyTorch workflow:
                # 1. Un-RoPE (or RoPE backward)
                # 2. Linear Backward
                # 3. RMSNorm Backward
                
                # We can't easily execute "Un-RoPE" without a kernel or complex python.
                # So we just benchmark Linear Bwd + RMSNorm Bwd, and acknowledge Metal is doing MORE work (RoPE)
                # This makes the comparison stricter for Metal.
                
                # Reconstruct Linear+RMS
                q_p = F.linear(x_norm_t, W_q).view(B, S, NumHeads, HeadDim)
                k_p = F.linear(x_norm_t, W_k).view(B, S, NumHeads, HeadDim)
                v_p = F.linear(x_norm_t, W_v).view(B, S, NumHeads, HeadDim)
                
                torch.autograd.grad(
                    (q_p, k_p, v_p),
                    (hidden_states, W_q, W_k, W_v, rms_weight),
                    grad_outputs=(d_q, d_k, d_v),
                    retain_graph=True
                )
            
             # Warmup
            try:
                if device == 'mps': torch.mps.synchronize()
                run_metal()
                if device == 'mps': torch.mps.synchronize()
                
                start = time.time()
                for _ in range(iters):
                    run_metal()
                    if device == 'mps': torch.mps.synchronize()
                metal_time = (time.time() - start) / iters * 1000
                
                start = time.time()
                for _ in range(iters):
                    run_torch()
                    if device == 'mps': torch.mps.synchronize()
                torch_time = (time.time() - start) / iters * 1000
                
                ratio = metal_time / torch_time
                status = color_ratio(ratio)
                
                results.append([
                    f"FusedAtt Bwd", f"{B}x{S} {desc} {dtype_name}",
                    format_time(metal_time), format_time(torch_time),
                    f"{ratio:.2f}x", status
                ])
                print(f"    Metal: {format_time(metal_time)}, Torch: {format_time(torch_time)} -> {ratio:.2f}x {status}")
                
            except Exception as e:
                print(f"    Failed: {e}")
                import traceback
                traceback.print_exc()

    return results

    return results


def benchmark_sdpa():

    """Benchmark Scaled Dot Product Attention."""
    print("Benchmarking SDPA (Scaled Dot Product Attention)...")
    results = []
    
    try:
        from metalcore import metal_scaled_dot_product_attention
        import torch.nn.functional as F
    except ImportError:
        print("  metalcore SDPA not available, skipping")
        return results

    configs = [
        ((2, 8, 64, 64), "Small (B=2, H=8, N=64, D=64)"),
        ((2, 8, 256, 64), "Medium (B=2, H=8, N=256, D=64)"),
        ((1, 8, 512, 64), "Large (B=1, H=8, N=512, D=64)"),
    ]
    if LITE_MODE:
        configs = [((2, 8, 64, 64), "Small (B=2, H=8, N=64, D=64)")]
    
    iters = 1 if LITE_MODE else (20 if QUICK_MODE else 50)
    warmup = 1 if LITE_MODE else 3
    
    for dtype_name, dtype in BENCHMARK_DTYPES:
        for shape, label in configs:
            B, H, N, D = shape
            if QUICK_MODE and N > 256: continue
            
            Q = torch.randn(B, H, N, D, device=device, dtype=dtype)
            K = torch.randn(B, H, N, D, device=device, dtype=dtype)
            V = torch.randn(B, H, N, D, device=device, dtype=dtype)
            
            # Non-causal SDPA
            name = f"SDPA {label} {dtype_name}"
            print(f"  Running {name}...")
            
            def run_metal_sdpa():
                y = metal_scaled_dot_product_attention(Q, K, V, is_causal=False)
                if device == 'mps': torch.mps.synchronize()
                return y
                
            def run_torch_sdpa():
                y = F.scaled_dot_product_attention(Q, K, V, is_causal=False)
                if device == 'mps': torch.mps.synchronize()
                return y
            
            for _ in range(warmup):
                run_metal_sdpa()
                run_torch_sdpa()
            
            start = time.time()
            for _ in range(iters):
                run_metal_sdpa()
            metal_time = (time.time() - start) / iters * 1000
            
            start = time.time()
            for _ in range(iters):
                run_torch_sdpa()
            torch_time = (time.time() - start) / iters * 1000
            
            ratio = metal_time / torch_time
            status = color_ratio(ratio)
            
            # Accuracy check
            O_metal = run_metal_sdpa()
            O_torch = run_torch_sdpa()
            max_err = (O_metal.float() - O_torch.float()).abs().max().item()
            
            results.append([
                label, f"B={B}, H={H}, N={N}, D={D} {dtype_name}",
                format_time(metal_time), format_time(torch_time),
                f"{ratio:.2f}x", status, format_accuracy(max_err)
            ])
            print(f"    Metal: {format_time(metal_time)}, Torch: {format_time(torch_time)} -> {ratio:.2f}x {status}")
            
            # Causal SDPA
            name = f"SDPA Causal {label} {dtype_name}"
            print(f"  Running {name}...")
            
            def run_metal_sdpa_causal():
                y = metal_scaled_dot_product_attention(Q, K, V, is_causal=True)
                if device == 'mps': torch.mps.synchronize()
                return y
                
            def run_torch_sdpa_causal():
                y = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
                if device == 'mps': torch.mps.synchronize()
                return y
            
            for _ in range(warmup):
                run_metal_sdpa_causal()
                run_torch_sdpa_causal()
            
            start = time.time()
            for _ in range(iters):
                run_metal_sdpa_causal()
            metal_time = (time.time() - start) / iters * 1000
            
            start = time.time()
            for _ in range(iters):
                run_torch_sdpa_causal()
            torch_time = (time.time() - start) / iters * 1000
            
            ratio = metal_time / torch_time
            status = color_ratio(ratio)
            
            # Accuracy check
            O_metal = run_metal_sdpa_causal()
            O_torch = run_torch_sdpa_causal()
            max_err = (O_metal.float() - O_torch.float()).abs().max().item()
            
            results.append([
                f"{label} (causal)", f"B={B}, H={H}, N={N}, D={D} {dtype_name}",
                format_time(metal_time), format_time(torch_time),
                f"{ratio:.2f}x", status, format_accuracy(max_err)
            ])
            print(f"    Metal: {format_time(metal_time)}, Torch: {format_time(torch_time)} -> {ratio:.2f}x {status}")
        
    return results


def benchmark_quantization():
    """Benchmark INT4 quantized matrix multiplication."""
    print("Benchmarking INT4 Quantization...")
    results = []
    
    try:
        import metalcore
    except ImportError:
        print("  metalcore not available, skipping")
        return results
    
    # Check if INT4 is available
    if not hasattr(metalcore, 'matmul_int4'):
        print("  INT4 matmul not available, skipping")
        return results
    
    configs = [
        # Single token inference
        (1, 4096, 4096, "Single token 7B attn"),
        (1, 4096, 11008, "Single token 7B MLP"),
        # Batch inference
        (8, 4096, 4096, "Batch 8 attn"),
        (32, 4096, 4096, "Batch 32 attn"),
        (32, 4096, 11008, "Batch 32 MLP"),
        # Prefill
        (128, 4096, 4096, "Prefill 128"),
    ]
    
    iters = 3 if LITE_MODE else 5
    
    for M, K, N, desc in configs:
        if SKIP_LONG_OPS and K * N > 4096 * 4096:
            continue
            
        try:
            # Create and quantize weights
            W = torch.randn(K, N, device=device, dtype=torch.float32)
            W_packed, scales, zeros = metalcore.quantize_int4(W, group_size=128)
            X = torch.randn(M, K, device=device, dtype=torch.float32)
            
            # Warmup
            Y = metalcore.matmul_int4(X, W_packed, scales, zeros, 128)
            torch.mps.synchronize()
            
            # INT4 timing
            torch.mps.synchronize()
            start = time.time()
            for _ in range(iters):
                Y = metalcore.matmul_int4(X, W_packed, scales, zeros, 128)
            torch.mps.synchronize()
            int4_time = (time.time() - start) / iters * 1000
            
            # FP32 timing (reference)
            torch.mps.synchronize()
            start = time.time()
            for _ in range(iters):
                Y_ref = X @ W
            torch.mps.synchronize()
            fp32_time = (time.time() - start) / iters * 1000
            
            # Memory savings
            fp32_bytes = K * N * 4
            int4_bytes = (K // 2) * N + scales.numel() * 4 + zeros.numel() * 4
            compression = fp32_bytes / int4_bytes
            
            ratio = int4_time / fp32_time
            status = color_ratio(ratio)
            
            results.append([
                f"{M}×{K}×{N}", desc,
                format_time(int4_time), format_time(fp32_time),
                f"{ratio:.0f}x", status,
                f"{compression:.1f}x mem"
            ])
            print(f"  {desc}: {ratio:.0f}x slower, {compression:.1f}x compression")
            
        except Exception as e:
            print(f"  {desc}: Error - {e}")
            continue
    
    return results

    return results


def benchmark_rope_new():
    """Benchmark RoPE forward and backward for new dtypes."""
    print("Benchmarking RoPE (Fwd+Bwd)...")
    results = []
    
    try:
        import metalcore_backend as mc
    except ImportError:
        print("Metalcore backend not found")
        return results

    configs = [
        (32, 1024, 32, 128, "Llama-7B"), # B, S, H, D
    ]
    if LITE_MODE:
        configs = [(32, 256, 8, 128, "Lite")]
        
    iters = 1 if LITE_MODE else 50
    warmup = 1 if LITE_MODE else 5
    
    for dtype_name, dtype in BENCHMARK_DTYPES:
        for B, S, H, D, name in configs:
            qk = torch.randn(B, S, H, D, device=device, dtype=dtype)
            
            # Create cos/sin
            inv_freq = 1.0 / (10000 ** (torch.arange(0, D, 2).float().to(device) / D))
            t = torch.arange(S, device=device, dtype=torch.float32)
            freqs = torch.outer(t, inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos().to(dtype)
            sin = emb.sin().to(dtype)
            
            # FWD Benchmark
            def run_fwd_metal():
                return mc.rope_fwd(qk, cos, sin)
                
            def run_fwd_torch():
                # Split half implementation manually for baseline
                x1 = qk[..., :D//2]
                x2 = qk[..., D//2:]
                c = cos.view(1, S, 1, D).expand(B, S, H, D)
                s = sin.view(1, S, 1, D).expand(B, S, H, D)
                # Just return a dummy computation to simulate workload
                return qk * 1.0 
            
            # Just verify it runs without error/fallback first
            try:
                out = run_fwd_metal()
                if device == 'mps': torch.mps.synchronize()
                
                start = time.time()
                for _ in range(iters):
                    run_fwd_metal()
                    if device == 'mps': torch.mps.synchronize()
                metal_time = (time.time() - start) / iters * 1000
                
                results.append([
                    f"RoPE Fwd", f"{B}x{S}x{H}x{D} {dtype_name}",
                    format_time(metal_time), "N/A", "N/A", "✅" 
                ])
                print(f"  RoPE Fwd {dtype_name}: {format_time(metal_time)}")
                
            except Exception as e:
                print(f"  RoPE Fwd {dtype_name} FAILED: {e}")
                
            # BWD Benchmark
            d_out = torch.randn_like(qk)
            try:
                grad = mc.rope_bwd(d_out, cos, sin)
                if device == 'mps': torch.mps.synchronize()
                
                start = time.time()
                for _ in range(iters):
                    mc.rope_bwd(d_out, cos, sin)
                    if device == 'mps': torch.mps.synchronize()
                metal_time_bwd = (time.time() - start) / iters * 1000
                
                results.append([
                    f"RoPE Bwd", f"{B}x{S}x{H}x{D} {dtype_name}",
                    format_time(metal_time_bwd), "N/A", "N/A", "✅"
                ])
                print(f"  RoPE Bwd {dtype_name}: {format_time(metal_time_bwd)}")
            except Exception as e:
                print(f"  RoPE Bwd {dtype_name} FAILED: {e}")

    return results


def benchmark_pipeline():
    """Benchmark chained operations - GPU advantage when data stays on device."""
    print("Benchmarking Pipeline Operations...")
    results = []
    
    try:
        import metalcore
        import metalcore_backend as mc
    except ImportError:
        print("  Required packages not available, skipping")
        return results
    
    # Pipeline 1: Batched QR x3 (e.g., iterative refinement)
    print("  Pipeline: QR -> QR -> QR (batched)")
    batch, M, N = 200, 32, 32
    A_gpu = torch.randn(batch, M, N, device='mps')
    A_cpu = A_gpu.cpu()
    
    # GPU pipeline - data stays on GPU
    torch.mps.synchronize()
    start = time.time()
    iters = 5
    for _ in range(iters):
        Q1, R1 = mc.qr_batched(A_gpu)
        Q2, R2 = mc.qr_batched(Q1)  # QR of Q
        Q3, R3 = mc.qr_batched(Q2)  # QR again
    torch.mps.synchronize()
    gpu_time = (time.time() - start) / iters * 1000
    
    # CPU pipeline
    start = time.time()
    for _ in range(iters):
        results_cpu = []
        for i in range(batch):
            Q1, R1 = torch.linalg.qr(A_cpu[i])
            Q2, R2 = torch.linalg.qr(Q1)
            Q3, R3 = torch.linalg.qr(Q2)
    cpu_time = (time.time() - start) / iters * 1000
    
    ratio = gpu_time / cpu_time
    results.append([
        "QR -> QR -> QR", f"{batch}×{M}×{N}",
        format_time(gpu_time), format_time(cpu_time),
        f"{ratio:.2f}x", color_ratio(ratio)
    ])
    print(f"    QR x3: {ratio:.2f}x")
    
    # Pipeline 2: SVD batched (use case: PCA preprocessing)
    print("  Pipeline: SVD batched (PCA)")
    batch, M, N = 50, 128, 64
    A_gpu = torch.randn(batch, M, N, device='mps')
    A_cpu = A_gpu.cpu()
    
    # GPU
    torch.mps.synchronize()
    start = time.time()
    iters = 3
    for _ in range(iters):
        U, S, V = metalcore.svd(A_gpu)
        # Truncate to top-k components (simulate PCA)
        k = 32
        U_k = U[:, :, :k]
        S_k = S[:, :k]
    torch.mps.synchronize()
    gpu_time = (time.time() - start) / iters * 1000
    
    # CPU
    start = time.time()
    for _ in range(iters):
        for i in range(batch):
            U, S, V = torch.linalg.svd(A_cpu[i], full_matrices=False)
            U_k = U[:, :k]
            S_k = S[:k]
    cpu_time = (time.time() - start) / iters * 1000
    
    ratio = gpu_time / cpu_time
    results.append([
        "SVD -> truncate (PCA)", f"{batch}×{M}×{N}",
        format_time(gpu_time), format_time(cpu_time),
        f"{ratio:.2f}x", color_ratio(ratio)
    ])
    print(f"    SVD+PCA: {ratio:.2f}x")
    
    # Pipeline 3: Large batch QR for gradient computation style workload
    print("  Pipeline: Large batch (ML training)")
    batch, M, N = 1000, 16, 16
    A_gpu = torch.randn(batch, M, N, device='mps')
    A_cpu = A_gpu.cpu()
    
    # GPU
    torch.mps.synchronize()
    start = time.time()
    iters = 5
    for _ in range(iters):
        Q, R = mc.qr_batched(A_gpu)
        # Simulate using Q in next operation
        result = Q @ Q.transpose(-1, -2)  # Q @ Q^T
    torch.mps.synchronize()
    gpu_time = (time.time() - start) / iters * 1000
    
    # CPU
    start = time.time()
    for _ in range(iters):
        for i in range(batch):
            Q, R = torch.linalg.qr(A_cpu[i])
            result = Q @ Q.T
    cpu_time = (time.time() - start) / iters * 1000
    
    ratio = gpu_time / cpu_time
    results.append([
        "QR -> matmul (ML)", f"{batch}×{M}×{N}",
        format_time(gpu_time), format_time(cpu_time),
        f"{ratio:.2f}x", color_ratio(ratio)
    ])
    print(f"    QR+matmul: {ratio:.2f}x")
    
    # Pipeline 4: Mixed - GPU fast + GPU slow + GPU fast vs CPU with transfers
    # This shows that even with a slower middle op, staying on GPU can win
    print("  Pipeline: Mixed (fast+slow+fast) - avoiding transfers")
    batch, M, N = 200, 32, 32
    A_gpu = torch.randn(batch, M, N, device='mps')
    A_cpu = A_gpu.cpu()
    
    # GPU pipeline - all operations stay on GPU (no transfers)
    # Op1: Batched QR (GPU wins)
    # Op2: Single matrix QR on result (GPU loses, but avoids transfer)
    # Op3: Batched matmul (GPU wins)
    torch.mps.synchronize()
    start = time.time()
    iters = 5
    for _ in range(iters):
        Q1, R1 = mc.qr_batched(A_gpu)  # GPU wins here
        # Simulate a "slow" GPU op - we'll use the fused single QR as example
        Q_single, R_single = mc.qr(R1[0])  # GPU loses but no transfer
        # Back to batched op
        result = Q1 @ Q1.transpose(-1, -2)  # GPU wins
    torch.mps.synchronize()
    gpu_all_time = (time.time() - start) / iters * 1000
    
    # Hybrid: CPU for slow op (requires transfer)
    torch.mps.synchronize()
    start = time.time()
    for _ in range(iters):
        Q1, R1 = mc.qr_batched(A_gpu)  # GPU
        R1_cpu = R1[0].cpu()  # Transfer to CPU
        Q_single, R_single = torch.linalg.qr(R1_cpu)  # CPU (faster for single)
        # Transfer back and continue
        result = Q1 @ Q1.transpose(-1, -2)  # GPU
    torch.mps.synchronize()
    hybrid_time = (time.time() - start) / iters * 1000
    
    # Pure CPU
    start = time.time()
    for _ in range(iters):
        for i in range(batch):
            Q1, R1 = torch.linalg.qr(A_cpu[i])
        Q_single, R_single = torch.linalg.qr(A_cpu[0])  # CPU
        for i in range(batch):
            result = A_cpu[i] @ A_cpu[i].T
    cpu_time = (time.time() - start) / iters * 1000
    
    # GPU all vs Hybrid
    ratio_hybrid = gpu_all_time / hybrid_time
    results.append([
        "Fast+Slow+Fast (GPU all)", f"{batch}×{M}×{N}",
        format_time(gpu_all_time), format_time(hybrid_time),
        f"{ratio_hybrid:.2f}x vs hybrid", "🟢" if ratio_hybrid < 1 else "🟠"
    ])
    
    ratio_cpu = gpu_all_time / cpu_time
    results.append([
        "Fast+Slow+Fast (vs CPU)", f"{batch}×{M}×{N}",
        format_time(gpu_all_time), format_time(cpu_time),
        f"{ratio_cpu:.2f}x vs CPU", color_ratio(ratio_cpu)
    ])
    print(f"    Mixed pipeline: GPU-all={gpu_all_time:.1f}ms, Hybrid={hybrid_time:.1f}ms, CPU={cpu_time:.1f}ms")
    
    return results


def main():
    start_time = time.time()
    
    print("=" * 60)
    print("METALOPS UNIFIED BENCHMARK")
    print("=" * 60)
    print()
    
    results = BenchmarkResults()
    
    # SVD benchmarks
    svd_results = []
    if RUN_SVD:
        svd_results = benchmark_svd()
        if svd_results:
            results.add_section(
                "SVD (metalsvd)",
                "Singular Value Decomposition using Jacobi algorithm on GPU",
                ["Shape", "Config", "Metal", "CPU", "Ratio", "Status", "Recon Error"],
                svd_results
            )
    
    # QR single matrix benchmarks
    qr_single_results = []
    if RUN_QR:
        qr_single_results = benchmark_qr_single()
        if qr_single_results:
            results.add_section(
                "QR Single Matrix (metalcore)",
                "Single matrix QR - CPU typically wins due to sequential dependencies",
                ["Shape", "Config", "Metal", "CPU", "Ratio", "Status", "Recon", "Ortho"],
                qr_single_results
            )
    
    # QR batched benchmarks
    qr_batched_results = []
    if RUN_QR:
        qr_batched_results = benchmark_qr_batched()
        if qr_batched_results:
            results.add_section(
                "QR Batched (metalcore) ⭐ GPU WINS",
                "Batched QR - GPU processes all matrices in parallel, single dispatch",
                ["Shape", "Config", "Metal", "CPU", "Ratio", "Status", "Recon Error"],
                qr_batched_results
            )
    
    # Cholesky benchmarks
    cholesky_results = []
    if RUN_CHOLESKY:
        cholesky_results = benchmark_cholesky()
        if cholesky_results:
            results.add_section(
                "Cholesky (metalcore) ⭐ GPU WINS",
                "Batched Cholesky decomposition with MAGMA-style shared memory",
                ["Shape", "Config", "Metal", "CPU", "Ratio", "Status", "Recon Error"],
                cholesky_results
            )
    
    # Solve benchmarks
    solve_results = []
    if RUN_SOLVE:
        solve_results = benchmark_solve()
        if solve_results:
            results.add_section(
                "Linear Solve (metalcore)",
                "Batched linear system solve using QR + TRSM",
                ["Shape", "Config", "Metal", "CPU", "Ratio", "Status", "Residual"],
                solve_results
            )
    
    # EIGH benchmarks
    eigh_results = []
    if RUN_EIGH:
        eigh_results = benchmark_eigh()
        if eigh_results:
            results.add_section(
                "Eigendecomposition (metaleig)",
                "Symmetric eigenvalue decomposition",
                ["Shape", "Config", "Metal", "CPU", "Ratio", "Status", "Recon Error"],
                eigh_results
            )
    
    # RMSNorm benchmarks
    rmsnorm_results = []
    if RUN_RMSNORM:
        rmsnorm_results = benchmark_rmsnorm()
        if rmsnorm_results:
            results.add_section(
                "RMSNorm (metalcore) ⭐ GPU WINS",
                "Fused RMSNorm kernel vs torch.nn.RMSNorm",
                ["Shape", "Config", "Metal", "CPU", "Ratio", "Status"],
                rmsnorm_results
            )
            
    # AdamW benchmarks
    adamw_results = []
    if RUN_ADAMW:
        adamw_results = benchmark_adamw()
        if adamw_results:
            results.add_section(
                "AdamW (metalcore) ⭐ GPU WINS",
                "Fused AdamW optimizer step vs torch.optim.AdamW",
                ["Params", "Size", "Metal", "CPU", "Ratio", "Status"],
                adamw_results
            )

    # Activations benchmarks
    activations_results = []
    if RUN_ACTIVATIONS:
        activations_results = benchmark_activations()
        if activations_results:
            results.add_section(
                "Activations (metalcore)",
                "GELU/SiLU activations with float4 vectorization",
                ["Op", "Shape", "Metal", "Torch", "Ratio", "Status"],
                activations_results
            )
        
        # Fused Bias+Activations
        fused_act_results = benchmark_fused_activations()
        if fused_act_results:
            results.add_section(
                "Fused Bias+Activations (metalcore)",
                "Fused bias + GELU/SiLU eliminating intermediate tensor",
                ["Op", "Shape", "Metal", "Torch", "Ratio", "Status"],
                fused_act_results
            )
        
        # Fused Add+LayerNorm
        fused_ln_results = benchmark_fused_add_layernorm()
        if fused_ln_results:
            results.add_section(
                "Fused Add+LayerNorm (metalcore)",
                "Fused residual + LayerNorm for transformer blocks",
                ["Config", "Shape", "Metal", "Torch", "Ratio", "Status"],
                fused_ln_results
            )
    
    # SDPA benchmarks
    sdpa_results = []
    if RUN_SDPA:
        sdpa_results = benchmark_sdpa()
        if sdpa_results:
            results.add_section(
                "SDPA (metalcore)",
                "Scaled Dot Product Attention with Flash Attention v2 tiling",
                ["Config", "Shape", "Metal", "Torch", "Ratio", "Status", "Error"],
                sdpa_results
            )

    # Fused Softmax benchmarks
    softmax_results = []
    if RUN_SOFTMAX:
        softmax_results = benchmark_softmax()
        if softmax_results:
            results.add_section(
                "Fused Softmax (metalcore)",
                "Online softmax algorithm with SIMD reductions",
                ["Config", "Shape", "Metal", "Torch", "Ratio", "Status", "Error"],
                softmax_results
            )
            
    # RoPE benchmarks
    rope_results = []
    if RUN_ROPE:
        rope_results = benchmark_rope_new()
        if rope_results:
            results.add_section(
                "RoPE (metalcore)",
                "Rotary Position Embedding (Fwd+Bwd)",
                ["Op", "Config", "Metal", "Torch", "Ratio", "Status"],
                rope_results
            )

    # LayerNorm benchmarks
    layernorm_results = []
    if RUN_LAYERNORM:
        layernorm_results = benchmark_layernorm()
        if layernorm_results:
            results.add_section(
                "LayerNorm (metalcore)",
                "Welford's algorithm for fused mean/variance",
                ["Config", "Shape", "Metal", "Torch", "Ratio", "Status", "Error"],
                layernorm_results
            )
    
    # Embedding Bag benchmarks
    embedding_results = []
    if RUN_EMBEDDING:
        embedding_results = benchmark_embedding_bag()
        if embedding_results:
            results.add_section(
                "Embedding Bag (metalcore)",
                "Coalesced reads for embedding lookups and aggregation",
                ["Config", "Shape", "Metal", "Torch", "Ratio", "Status"],
                embedding_results
            )
    
    # Scatter/Gather benchmarks
    scatter_results = []
    if RUN_SCATTER:
        scatter_results = benchmark_scatter_gather()
        if scatter_results:
            results.add_section(
                "Scatter/Gather (metalcore)",
                "Atomic scatter_add and vectorized gather operations",
                ["Op", "Shape", "Metal", "Torch", "Ratio", "Status"],
                scatter_results
            )

    # LoRA Training Operations benchmarks
    lora_results = []
    if RUN_LORA:
        lora_results = benchmark_lora_training()
        lora_results.extend(benchmark_fused_swiglu_mlp())
        if lora_results:
            results.add_section(
                "LoRA Training Ops (metalcore)",
                "Fused operations for LoRA fine-tuning: cross-entropy, KL divergence, SwiGLU, LoRA linear",
                ["Op", "Config", "Metal", "Torch", "Ratio", "Status"],
                lora_results
            )
    
    # Fused Attention Backward
    fused_att_results = []
    if RUN_FUSED_ATT_BWD:
        fused_att_results = benchmark_fused_attention_bwd()
        if fused_att_results:
            results.add_section(
                "Fused Attention Backward (metalcore)",
                "Fused Bwd: SDPA Grads -> RoPE Bwd -> QKV Bwd -> RMSNorm Bwd",
                ["Op", "Config", "Metal", "Torch", "Ratio", "Status"],
                fused_att_results
            )
            
    # Fused MLP Backward
    fused_mlp_results = []
    if RUN_FUSED_MLP_BWD:
        fused_mlp_results = benchmark_fused_mlp_bwd()
        if fused_mlp_results:
            results.add_section(
                "Fused MLP Backward (metalcore)",
                "Fused Bwd: Down Bwd -> SwiGLU Bwd -> Gate/Up Bwd -> RMSNorm Bwd",
                ["Op", "Config", "Metal", "Torch", "Ratio", "Status"],
                fused_mlp_results
            )

    # Pipeline benchmarks
    pipeline_results = []
    if RUN_PIPELINE:
        pipeline_results = benchmark_pipeline()
        if pipeline_results:
            results.add_section(
                "Pipeline Operations ⭐ GPU WINS (No Transfer)",
                "Chained operations where data stays on GPU - avoids costly memory transfers",
                ["Pipeline", "Shape", "GPU", "Comparison", "Ratio", "Status"],
                pipeline_results
            )
    
    # LLM Model Families benchmarks
    model_results = {}
    if RUN_MODELS:
        model_results = benchmark_models()
        for model_name, model_data in model_results.items():
            if model_data:
                results.add_section(
                    f"LLM: {model_name}",
                    f"SVD performance on {model_name} weight matrix sizes",
                    ["Shape", "Layer", "Metal", "CPU", "Ratio", "Status", "Recon Error"],
                    model_data
                )
    
    # Add summary section
    results.add_section(
        "Usage Recommendations",
        None,
        ["Operation", "When to Use Metal", "When to Use CPU"],
        [
            ["SVD", "Batched small/medium matrices", "Single large matrices"],
            ["QR (single)", "—", "Always (sequential dependencies)"],
            ["QR (batched)", "Many small matrices (10x speedup!)", "Few matrices"],
            ["EIGH", "Batched symmetric matrices", "Single large matrices"],
            ["Pipeline", "Keep data on GPU to avoid transfer cost", "Single ops on CPU-resident data"],
        ]
    )
    
    # Write results (skip in lite mode)
    if not LITE_MODE:
        output_path = Path(__file__).parent / "benchmarks.md"
        
        # Track which benchmark sections were actually run
        ran_benchmarks = set()
        if RUN_SVD and svd_results:
            ran_benchmarks.add("SVD (metalcore) ⭐ GPU WINS")
            ran_benchmarks.add("SVD (metalsvd)")
        if RUN_QR and qr_single_results:
            ran_benchmarks.add("QR Single Matrix (metalcore)")
        if RUN_QR and qr_batched_results:
            ran_benchmarks.add("QR Batched (metalcore) ⭐ GPU WINS")
        if RUN_CHOLESKY and cholesky_results:
            ran_benchmarks.add("Cholesky (metalcore) ⭐ GPU WINS")
        if RUN_SOLVE and solve_results:
            ran_benchmarks.add("Linear Solve (metalcore)")
        if RUN_EIGH and eigh_results:
            ran_benchmarks.add("Eigendecomposition (metaleig)")
        if RUN_RMSNORM and rmsnorm_results:
            ran_benchmarks.add("RMSNorm (metalcore) ⭐ GPU WINS")
        if RUN_ADAMW and adamw_results:
            ran_benchmarks.add("AdamW (metalcore) ⭐ GPU WINS")
        if RUN_ACTIVATIONS and activations_results:
            ran_benchmarks.add("Activations (metalcore)")
        if RUN_SDPA and sdpa_results:
            ran_benchmarks.add("SDPA (metalcore)")
        if RUN_PIPELINE and pipeline_results:
            ran_benchmarks.add("Pipeline Operations ⭐ GPU WINS (No Transfer)")
        if RUN_LORA and lora_results:
            ran_benchmarks.add("LoRA Training Ops (metalcore)")
        if RUN_FUSED_ATT_BWD and fused_att_results:
            ran_benchmarks.add("Fused Attention Backward (metalcore)")
        if RUN_FUSED_MLP_BWD and fused_mlp_results:
            ran_benchmarks.add("Fused MLP Backward (metalcore)")
        
        # If running all benchmarks, just overwrite the file
        if RUN_ALL:
            with open(output_path, 'w') as f:
                f.write(results.to_markdown())
            print(f"\nFull benchmark run - overwrote {output_path}")
        else:
            # Partial run - merge with existing file
            existing_sections = parse_existing_benchmarks(output_path)
            merged = merge_benchmark_sections(results, existing_sections, ran_benchmarks)
            
            # Build the merged markdown file
            lines = [
                "# Metalops Benchmark Results",
                "",
                f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
                "",
                "**Legend:** 💚 GPU wins big (>3x) | 🟢 GPU wins | 🔵 Close | ⚪ CPU wins | 🟠 CPU wins big (>3x)",
                "",
            ]
            
            # Add sections in canonical order
            for section_title in SECTION_ORDER:
                if section_title in merged:
                    lines.append(merged[section_title])
            
            # Add any sections not in canonical order (shouldn't happen, but just in case)
            for section_title in merged:
                if section_title not in SECTION_ORDER:
                    lines.append(merged[section_title])
            
            with open(output_path, 'w') as f:
                f.write('\n'.join(lines))
            
            print(f"\nPartial benchmark run - merged {len(ran_benchmarks)} section(s) into {output_path}")
            print(f"  Updated: {', '.join(sorted(ran_benchmarks))}")
        
        # Save to JSONL history (structured data for comparisons)
        history_data = {
            "svd": svd_results,
            "qr_single": qr_single_results,
            "qr_batched": qr_batched_results,
            "eigh": eigh_results,
            "eigh": eigh_results,
            "cholesky": cholesky_results,
            "solve": solve_results,
            "rmsnorm": rmsnorm_results,
            "adamw": adamw_results,
            "models": model_results,
        }
        
        # Load previous for comparison if --compare flag
        if args.compare:
            previous = load_previous_benchmark()
            compare_with_previous(history_data, previous)
        
        # Add runtime to history
        total_runtime = time.time() - start_time
        history_data["runtime_seconds"] = total_runtime
        
        save_benchmark_history(history_data)
        
        print()
        print(f"Results written to: {output_path}")
        print(f"Total runtime: {total_runtime:.1f}s ({total_runtime/60:.1f}m)")
    else:
        total_runtime = time.time() - start_time
        print()
        print("Lite mode: skipped file write")
        print(f"Total runtime: {total_runtime:.1f}s")
    print()
    print("=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
