import torch
import safetensors.torch
import argparse

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Convert PyTorch model to safetensors format')
    parser.add_argument('--input', '-i', type=str, 
                        default="/root/Wan2.1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
                        help='Path to the input PyTorch model file')
    parser.add_argument('--output', '-o', type=str, 
                        default=None,
                        help='Path where the output safetensors file will be saved (defaults to input path with .safetensors extension)')
    parser.add_argument('--device', '-d', type=str, default="cpu",
                        help='Device to load the model onto (e.g., "cpu", "cuda")')
    
    # Parse arguments
    args = parser.parse_args()
    
    # If output path is not specified, generate it from input path
    if args.output is None:
        # Replace the file extension with .safetensors or add it if no extension exists
        if "." in args.input:
            args.output = args.input.rsplit(".", 1)[0] + ".safetensors"
        else:
            args.output = args.input + ".safetensors"
    
    print(f"Loading model from {args.input} to {args.device}...")
    
    # Load the PyTorch model
    model = torch.load(args.input, map_location=args.device)
    
    print(f"Converting and saving model to {args.output}...")
    
    # Save in safetensors format
    safetensors.torch.save_file(model, args.output)
    
    print(f"Model successfully converted and saved to {args.output}")

if __name__ == "__main__":
    main()