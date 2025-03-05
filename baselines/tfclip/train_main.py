import os
import argparse


def main():
    parser = argparse.ArgumentParser(description="Visualization script with command-line arguments")
    parser.add_argument("--output_dir", type=str, default="logs/run", help="Output directory for logs")
    args = parser.parse_args()
    
    output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    train_cmd = (
        f"python train.py "
        f"OUTPUT_DIR {output_dir}"
    )
    
    os.system(train_cmd)

if __name__ == "__main__":
    main()
