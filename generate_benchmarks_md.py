#!/usr/bin/env python3
"""
Generate benchmarks.md from benchmarks.json

This script reads the benchmark data from JSON and generates
a clean markdown file with current benchmark results.

Usage:
    python generate_benchmarks_md.py
"""

import json
from pathlib import Path

# Canonical section order
SECTION_ORDER = [
    "Activations (metalcore)",
    "SDPA (metalcore)",
    "SVD (metalcore) ⭐ GPU WINS",
    "SVD (metalsvd)",
    "QR Single Matrix (metalcore)",
    "QR Batched (metalcore) ⭐ GPU WINS",
    "Cholesky (metalcore) ⭐ GPU WINS",
    "Linear Solve (metalcore) ⭐ GPU WINS",
    "Eigendecomposition (metaleig)",
    "RMSNorm (metalcore) ⭐ GPU WINS",
    "AdamW (metalcore) ⭐ GPU WINS",
    "Pipeline Operations ⭐ GPU WINS (No Transfer)",
    "LLM: Llama",
    "LLM: Mistral",
    "LLM: Qwen",
    "LLM: Gemma",
    "LLM: Phi",
    "Usage Recommendations",
]


def get_section_avg_ratio(data, section_title):
    """Get average ratio for a section. Lower = GPU wins = should appear first."""
    if section_title not in data.get('sections', {}):
        return float('inf')
    
    section = data['sections'][section_title]
    ratios = []
    for row_key, row_data in section.get('rows', {}).items():
        # Try raw_values first
        raw = row_data.get('raw_values', {})
        if 'ratio' in raw:
            ratios.append(raw['ratio'])
        elif row_data.get('current'):
            # Extract ratio from formatted values (e.g., "0.38x")
            for val in row_data['current']:
                if isinstance(val, str) and val.endswith('x'):
                    try:
                        ratios.append(float(val.rstrip('x')))
                        break
                    except ValueError:
                        pass
    
    if not ratios:
        if "GPU WINS" in section_title:
            return 0.5  # Favor GPU WINS sections
        elif "Recommendations" in section_title:
            return 1000  # Always last
        return 50  # Neutral
    
    return sum(ratios) / len(ratios)


def main():
    base_path = Path(__file__).parent
    json_path = base_path / "benchmarks.json"
    md_path = base_path / "benchmarks.md"
    
    if not json_path.exists():
        print(f"Error: {json_path} not found")
        return 1
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Generate markdown
    lines = [
        "# Metalops Benchmark Results",
        "",
    ]
    
    if 'last_updated' in data.get('metadata', {}):
        lines.append(f"*Last updated: {data['metadata']['last_updated']}*")
        lines.append("")
    
    lines.extend([
        "**Legend:** 💚 GPU wins big (>3x) | 🟢 GPU wins | ⚪ Close | 🟠 CPU wins | 🔴 CPU wins big (>3x)",
        "",
    ])
    
    # Sort sections dynamically by performance (GPU wins first)
    all_sections = list(data.get('sections', {}).keys())
    sorted_sections = sorted(all_sections, key=lambda s: get_section_avg_ratio(data, s))
    
    section_count = 0
    row_count = 0
    
    for section_title in sorted_sections:
        section = data['sections'][section_title]
        
        # Skip empty sections
        if not section.get('rows'):
            continue
        
        section_count += 1
        
        lines.append(f"## {section_title}")
        lines.append("")
        
        if section.get('description'):
            lines.append(f"*{section['description']}*")
            lines.append("")
        
        if section.get('headers'):
            lines.append("| " + " | ".join(section['headers']) + " |")
            lines.append("|" + "|".join(["---"] * len(section['headers'])) + "|")
            
            # Sort rows by key for consistent ordering
            for row_key in sorted(section['rows'].keys()):
                row_data = section['rows'][row_key]
                if row_data.get('current'):
                    row_count += 1
                    lines.append("| " + " | ".join(str(x) for x in row_data['current']) + " |")
            lines.append("")
    
    # Write markdown
    with open(md_path, 'w') as f:
        f.write("\n".join(lines))
    
    print(f"Generated: {md_path}")
    print(f"  Sections: {section_count}")
    print(f"  Total rows: {row_count}")
    
    return 0


if __name__ == "__main__":
    exit(main())
