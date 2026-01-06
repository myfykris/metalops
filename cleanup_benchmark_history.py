#!/usr/bin/env python3
"""
Clean up benchmark_history.jsonl by:
1. Removing entries with mostly empty results (test runs)
2. Keeping only the most recent clean runs
"""

import json
from pathlib import Path
from datetime import datetime

def main():
    base_path = Path(__file__).parent
    history_path = base_path / "benchmark_history.jsonl"
    
    # Load all entries
    entries = []
    with open(history_path, 'r') as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    
    print(f"Loaded {len(entries)} entries")
    
    # Analyze each entry
    cleaned = []
    removed = []
    
    for entry in entries:
        timestamp = entry.get('timestamp', 'unknown')
        results = entry.get('results', {})
        
        # Count non-empty result categories
        non_empty_count = 0
        total_rows = 0
        
        for key, value in results.items():
            if key == 'runtime_seconds' or key == 'models':
                continue
            if isinstance(value, list) and len(value) > 0:
                non_empty_count += 1
                total_rows += len(value)
        
        runtime = results.get('runtime_seconds', 0)
        
        # Keep entry if it has meaningful data
        # - At least 2 non-empty categories OR
        # - At least 5 total rows OR
        # - Runtime > 5 seconds (likely a real run)
        if non_empty_count >= 2 or total_rows >= 5 or runtime > 5:
            cleaned.append(entry)
            print(f"✓ Keep: {timestamp} - {non_empty_count} categories, {total_rows} rows, {runtime:.1f}s")
        else:
            removed.append(entry)
            print(f"✗ Remove: {timestamp} - {non_empty_count} categories, {total_rows} rows, {runtime:.1f}s")
    
    print(f"\nKept {len(cleaned)} entries, removed {len(removed)}")
    
    # Write cleaned entries
    with open(history_path, 'w') as f:
        for entry in cleaned:
            f.write(json.dumps(entry) + '\n')
    
    print(f"\nCleaned history written to: {history_path}")
    
    # Also update benchmarks.json if it exists
    json_path = base_path / "benchmarks.json"
    if json_path.exists():
        print("\nNote: benchmarks.json is separate and was not modified.")
        print("To regenerate it from the new history, run: python populate_benchmarks_json.py")

if __name__ == "__main__":
    main()
