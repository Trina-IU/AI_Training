#!/usr/bin/env python3
"""
Sanitize filenames under a dataset root by removing/replacing non-ASCII or problematic characters.

Usage:
  # Dry-run, just show mappings
  python sanitize_filenames.py --dataset .\dataset --dry-run

  # Actually perform renames (will skip if target exists)
  python sanitize_filenames.py --dataset .\dataset --rename

This script is conservative: it shows proposed mappings and only renames when --rename is passed.
It replaces common problematic characters (degree sign -> deg) and strips non-ASCII.
"""
import argparse
from pathlib import Path
import unicodedata
import re


def safe_name(name: str) -> str:
    # Replace degree sign with 'deg'
    name = name.replace('°', 'deg')
    # Normalize and decompose accents
    name = unicodedata.normalize('NFKD', name)
    # Remove combining marks
    name = ''.join(ch for ch in name if not unicodedata.combining(ch))
    # Replace slashes and other path chars
    name = name.replace('/', '_').replace('\\', '_')
    # Replace non-ASCII with empty
    name = ''.join(ch for ch in name if ord(ch) < 128)
    # Replace multiple spaces/underscores with single underscore
    name = re.sub(r'[\s]+', '_', name)
    name = re.sub(r'[^0-9A-Za-z_\-\.\(\)]', '', name)
    name = name.strip(' _')
    if name == '':
        name = 'file'
    return name


def find_pngs(root: Path):
    for p in root.rglob('*.png'):
        yield p


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--rename', action='store_true')
    args = p.parse_args()

    root = Path(args.dataset)
    if not root.exists():
        print('Dataset root not found:', root)
        return

    mappings = []
    for fp in find_pngs(root):
        orig = fp.name
        new = safe_name(orig)
        # ensure extension preserved and not rename only case
        if new == orig:
            continue
        # avoid duplicate names: append counter if exists
        target = fp.with_name(new)
        counter = 1
        while target.exists() and target != fp:
            stem = Path(new).stem
            suffix = Path(new).suffix
            candidate = f"{stem}_{counter}{suffix}"
            target = fp.with_name(candidate)
            counter += 1

        mappings.append((fp, target))

    if not mappings:
        print('No filenames require sanitization.')
        return

    print(f'Found {len(mappings)} files to sanitize:')
    for src, tgt in mappings:
        print(f'  {src} -> {tgt.name}')

    if args.rename:
        print('\nRenaming now...')
        for src, tgt in mappings:
            try:
                src.rename(tgt)
            except Exception as e:
                print(f'Failed to rename {src} -> {tgt}: {e}')
        print('Renaming complete.')
    else:
        print('\nDry-run complete. Rerun with --rename to perform changes.')


if __name__ == '__main__':
    main()
