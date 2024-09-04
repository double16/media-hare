#!/usr/bin/env python3

import os
import argparse

import common


# Define the metadata extensions
metadata_extensions = [".bak.edl", ".edl", ".mkv.ini", ".comskip.ini"]

def find_orphaned_files(root_dir, dry_run=False):
    # Walk through all directories and files within the root directory
    for dirpath, dirnames, filenames in os.walk(root_dir):
        removed_files = False
        for filename in filenames:
            # Check if the file is a metadata file
            for ext in metadata_extensions:
                if filename.endswith(ext):
                    # Construct the base filename by removing the metadata extension
                    base_filename = filename[:-len(ext)] + ".mkv"
                    if base_filename.endswith(".bak.mkv"):
                        # special case of .bak.edl with .edl removed
                        continue
                    # Check if the corresponding .mkv file exists
                    if not os.path.exists(os.path.join(dirpath, base_filename)):
                        # If the .mkv file does not exist, it's an orphaned metadata file
                        orphan_file = os.path.join(dirpath, filename)
                        removed_files = True
                        if dry_run:
                            print(f"[DRY RUN] Would remove: {orphan_file}")
                        else:
                            os.remove(orphan_file)
                            print(f"Removed: {orphan_file}")
        # After processing files, check if the directory is empty
        if not os.listdir(dirpath) and removed_files:
            if dry_run:
                print(f"[DRY RUN] Would remove empty directory: {dirpath}")
            else:
                os.rmdir(dirpath)
                print(f"Removed empty directory: {dirpath}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove orphaned metadata files.")
    parser.add_argument("--root_dir", help="The root directory of your media files.")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without removing files.")
    args = parser.parse_args()

    if args.root_dir:
        find_orphaned_files(args.root_dir, dry_run=args.dry_run)
    else:
        for root_dir in common.get_media_paths():
            find_orphaned_files(root_dir, dry_run=args.dry_run)
