import torch
import safetensors.torch
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description='Convert PyTorch model to safetensors format')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Path to the input PyTorch model file')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Path for the output safetensors file (defaults to input with .safetensors extension)')
    parser.add_argument('--device', '-d', type=str, default="cpu",
                        help='Device to load the model onto (e.g., "cpu", "cuda")')
    
    args = parser.parse_args()
    
    output_path = args.output
    if output_path is None:
        base, _ = os.path.splitext(args.input)
        output_path = base + ".safetensors"
    
    print(f"Loading {args.input} to {args.device}...")
    model = torch.load(args.input, map_location=args.device)
    
    print(f"Saving model to {output_path}...")
    safetensors.torch.save_file(model, output_path)
    
    print(f"Successfully saved to {output_path}")

if __name__ == "__main__":
    main()