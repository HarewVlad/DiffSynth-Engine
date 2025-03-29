#!/usr/bin/env python3
from safetensors.torch import load_file, save_file
import torch
import os
import argparse
import re

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Combine partitioned safetensors model files into a single file.')
    parser.add_argument('--input-dir', type=str, required=True, help='Directory containing the model files')
    parser.add_argument('--output-path', type=str, help='Path to save the combined model (default: INPUT_DIR/model.safetensors)')
    parser.add_argument('--file-pattern', type=str, default='diffusion_pytorch_model-', help='Pattern to match for model files')
    parser.add_argument('--num-partitions', type=int, help='Number of partitions (if not specified, detect automatically)')
    parser.add_argument('--force', action='store_true', help='Overwrite output file if it exists')
    args = parser.parse_args()

    # Set output path if not specified
    if not args.output_path:
        args.output_path = os.path.join(args.input_dir, "model.safetensors")
    
    # Check if output file exists
    if os.path.exists(args.output_path) and not args.force:
        print(f"Error: Output file {args.output_path} already exists. Use --force to overwrite.")
        return

    # Find all files matching the pattern
    file_pattern_regex = f"{re.escape(args.file_pattern)}(\\d+)-of-(\\d+)\\.safetensors"
    model_files = []
    
    # Get all safetensors files in the directory
    for file in os.listdir(args.input_dir):
        match = re.match(file_pattern_regex, file)
        if match:
            partition_num = int(match.group(1))
            total_partitions = int(match.group(2))
            file_path = os.path.join(args.input_dir, file)
            model_files.append((file_path, partition_num, total_partitions))
    
    if not model_files:
        print(f"Error: No files matching pattern '{args.file_pattern}*-of-*.safetensors' found in {args.input_dir}")
        return

    # Check if all files mention the same total number of partitions
    total_partitions_set = set(total for _, _, total in model_files)
    if len(total_partitions_set) > 1:
        print(f"Warning: Inconsistent total partition counts found: {total_partitions_set}")
    
    total_partitions = next(iter(total_partitions_set))
    
    # Check if specified number of partitions matches detected number
    if args.num_partitions:
        if args.num_partitions != total_partitions:
            print(f"Warning: Specified number of partitions ({args.num_partitions}) differs from detected ({total_partitions})")
        total_partitions = args.num_partitions

    # Check if all expected partitions are present
    partition_nums = set(partition for _, partition, _ in model_files)
    expected_partitions = set(range(1, total_partitions + 1))
    missing_partitions = expected_partitions - partition_nums
    
    if missing_partitions:
        print(f"Error: Missing partitions: {missing_partitions}")
        return

    # Sort files by partition number
    model_files.sort(key=lambda x: x[1])  # Sort by partition number
    
    # Extract just the file paths after sorting
    model_file_paths = [file_path for file_path, _, _ in model_files]

    print(f"Found {len(model_files)} partition files out of {total_partitions} expected")
    
    # Load and combine all tensor dictionaries
    combined_tensors = {}
    total_keys = 0
    for i, file_path in enumerate(model_file_paths, 1):
        print(f"Loading file {i}/{len(model_file_paths)}: {os.path.basename(file_path)}")
        try:
            tensors = load_file(file_path)
            # Add tensors to combined dictionary
            for key, tensor in tensors.items():
                if key in combined_tensors:
                    print(f"Warning: Duplicate key {key} found. Overwriting.")
                combined_tensors[key] = tensor
            total_keys += len(tensors)
            print(f"  Loaded {len(tensors)} tensors")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return

    # Save the combined tensors
    print(f"Saving combined model to {args.output_path}")
    try:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
        save_file(combined_tensors, args.output_path)
        print(f"Successfully saved combined model with {len(combined_tensors)} tensors (from total of {total_keys} tensors in partitions)")
    except Exception as e:
        print(f"Error saving combined model: {e}")
        return

    print("Done!")

if __name__ == "__main__":
    main()