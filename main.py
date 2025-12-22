import argparse
import os
import asyncio
import sys

sys.stdout.reconfigure(line_buffering=True)

from src.process import process_subtitles

async def main():
    parser = argparse.ArgumentParser(description="SRT file post-processor using LLM.")
    parser.add_argument("--file", type=str, help="Source SRT file path.")
    parser.add_argument("--folder", type=str, help="Source folder of SRT files.")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size for processing.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode.")
    args = parser.parse_args()

    if args.file and args.folder:
        raise ValueError("Please specify either --file or --folder, not both.")

    if args.folder:
        for filename in os.listdir(args.folder):
            if filename.endswith(".srt"):
                file_path = os.path.join(args.folder, filename)
                await process_subtitles(source_srt_file=file_path,
                                        batch_size=args.batch_size,
                                        debug=args.debug)
    elif args.file:
        await process_subtitles(source_srt_file=args.file,
                                batch_size=args.batch_size,
                                debug=args.debug)
    else:
        raise ValueError("The following arguments are required: --file or --folder")

if __name__ == "__main__":
    asyncio.run(main())
