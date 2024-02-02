import shutil
import os
import argparse

def extract_archive(file_name, output_dir, format):
    '''
    Quick Guide:
    python script_name.py "./mnist-original.zip" --output_dir "./extracted" --format zip

    :param file_name: Path to the archive file
    :param output_dir: Output directory (default is the current working directory)
    :param format: Archive format (choices: "zip", "tar", "gztar", "bztar"; default is "zip")

    :return: None
    :process: Unpacks the archive file into the specified or default output directory
    '''
    # Check if output_dir is specified, otherwise use the current working directory
    if output_dir is None:
        output_dir = os.getcwd()

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Validate format option
    supported_formats = ["zip", "tar", "gztar", "bztar"]
    if format not in supported_formats:
        raise ValueError(f"Invalid format. Supported formats: {', '.join(supported_formats)}")

    # Unpack the archive
    shutil.unpack_archive(file_name, output_dir, format)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract files from an archive.")
    
    parser.add_argument("file_name", type=str, help="Path to the archive file")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (default is the current working directory)")
    parser.add_argument("--format", type=str, choices=["zip", "tar", "gztar", "bztar"], default="zip", help="Archive format (default is zip)")

    args = parser.parse_args()

    extract_archive(args.file_name, args.output_dir, args.format)
