#!/usr/bin/env python3
from safetensors.torch import load_file, save_file
import torch
import os
import argparse
import re
import sys
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser(description='Combine partitioned safetensors model files.')
    parser.add_argument('--input-dir', type=str, required=True, help='Directory containing partition files')
    parser.add_argument('--output-path', type=str, help='Output file path (default: INPUT_DIR/model.safetensors)')
    parser.add_argument('--file-pattern', type=str, default='model-', help='Prefix pattern for partition files')
    parser.add_argument('--num-partitions', type=int, help='Expected number of partitions (optional, auto-detects)')
    parser.add_argument('--force', action='store_true', help='Overwrite output file if it exists')
    args = parser.parse_args()

    output_path = args.output_path or os.path.join(args.input_dir, "model.safetensors")

    if os.path.exists(output_path) and not args.force:
        print(f"Error: Output file {output_path} exists. Use --force to overwrite.", file=sys.stderr)
        sys.exit(1)

    file_pattern_regex = re.compile(rf"^{re.escape(args.file_pattern)}(\d+)-of-(\d+)\.safetensors$")
    partitions = {}
    detected_totals = set()

    try:
        filenames = os.listdir(args.input_dir)
    except FileNotFoundError:
        print(f"Error: Input directory not found: {args.input_dir}", file=sys.stderr)
        sys.exit(1)
    except OSError as e:
        print(f"Error accessing input directory {args.input_dir}: {e}", file=sys.stderr)
        sys.exit(1)


    for filename in filenames:
        match = file_pattern_regex.match(filename)
        if match:
            part_num = int(match.group(1))
            total_num = int(match.group(2))
            if part_num in partitions:
                 print(f"Warning: Duplicate partition number {part_num} found ({filename} and {os.path.basename(partitions[part_num])}). Using {filename}.", file=sys.stderr)
            partitions[part_num] = os.path.join(args.input_dir, filename)
            detected_totals.add(total_num)

    if not partitions:
        print(f"Error: No files matching pattern '{args.file_pattern}*-of-*.safetensors' found in {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    if len(detected_totals) > 1:
        print(f"Warning: Inconsistent total partition counts found in filenames: {detected_totals}", file=sys.stderr)

    detected_total = next(iter(detected_totals)) if detected_totals else 0
    final_total_partitions = args.num_partitions or detected_total

    if args.num_partitions and args.num_partitions != detected_total and detected_totals:
         print(f"Warning: Specified partition count ({args.num_partitions}) differs from detected ({detected_total})", file=sys.stderr)

    if final_total_partitions <= 0:
         print(f"Error: Could not determine total number of partitions. Please specify with --num-partitions.", file=sys.stderr)
         sys.exit(1)

    expected_partitions = set(range(1, final_total_partitions + 1))
    found_partitions = set(partitions.keys())
    missing_partitions = expected_partitions - found_partitions
    extra_partitions = found_partitions - expected_partitions

    if missing_partitions:
        print(f"Error: Missing partitions: {sorted(list(missing_partitions))}", file=sys.stderr)
        sys.exit(1)
    if extra_partitions:
         print(f"Warning: Found partitions numbered higher than expected total ({final_total_partitions}): {sorted(list(extra_partitions))}", file=sys.stderr)
         # Decide whether to proceed or exit; currently proceeds using expected range

    sorted_partition_files = [partitions[i] for i in range(1, final_total_partitions + 1)]

    print(f"Found {len(found_partitions)} partition files. Combining {len(sorted_partition_files)} files for {final_total_partitions} total partitions.")

    combined_tensors = {}
    total_keys_loaded = 0
    for i, file_path in enumerate(sorted_partition_files, 1):
        print(f"Loading file {i}/{len(sorted_partition_files)}: {os.path.basename(file_path)}")
        try:
            tensors = load_file(file_path)
            keys_in_file = set(tensors.keys())
            duplicate_keys = keys_in_file.intersection(combined_tensors.keys())
            if duplicate_keys:
                 print(f"  Warning: Duplicate keys found: {duplicate_keys}. Overwriting.", file=sys.stderr)

            combined_tensors.update(tensors)
            total_keys_loaded += len(keys_in_file)
            print(f"  Loaded {len(keys_in_file)} tensors")
        except Exception as e:
            print(f"Error loading {file_path}: {e}", file=sys.stderr)
            sys.exit(1)

    print(f"Saving combined model ({len(combined_tensors)} unique tensors) to {output_path}")
    try:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        save_file(combined_tensors, output_path)
        print(f"Successfully saved combined model.")
    except Exception as e:
        print(f"Error saving combined model: {e}", file=sys.stderr)
        sys.exit(1)

    print("Done.")

if __name__ == "__main__":
    main()