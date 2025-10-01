"""
Fix labels.csv files to extract actual text from filenames.

The script will:
1. Read each labels.csv in dataset subfolders
2. Extract the actual text from filenames (text before parenthesis and _frame_)
3. Create corrected labels.csv files

Example:
  albuterol(1)_frame_00000.png,albuterol(1)_frame_00000
  becomes:
  albuterol(1)_frame_00000.png,albuterol
"""

import csv
import re
from pathlib import Path


def extract_text_from_filename(filename):
    """
    Extract the actual text label from filename.
    
    Examples:
      albuterol(1)_frame_00000.png -> albuterol
      paracetamol_orig_00021.png -> paracetamol
      0.3ml(1)_frame_00000.png -> 0.3ml
      BD_(1)_frame_00000.png -> BD
    """
    # Remove file extension
    name = Path(filename).stem
    
    # Pattern 1: text(number)_frame_xxxxx -> extract text before (
    match = re.match(r'^(.+?)\(\d+\)_frame_\d+$', name)
    if match:
        return match.group(1)
    
    # Pattern 2: text_orig_xxxxx -> extract text before _orig
    match = re.match(r'^(.+?)_orig_\d+$', name)
    if match:
        return match.group(1)
    
    # Pattern 3: text_frame_xxxxx -> extract text before _frame
    match = re.match(r'^(.+?)_frame_\d+$', name)
    if match:
        return match.group(1)
    
    # If no pattern matches, return the whole stem (fallback)
    return name


def fix_labels_csv(csv_path):
    """Fix a single labels.csv file."""
    print(f"\nProcessing: {csv_path}")
    
    # Read the CSV
    with open(csv_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    
    # Create backup
    backup_path = csv_path.with_suffix('.csv.backup')
    csv_path.rename(backup_path)
    print(f"  Backup saved: {backup_path}")
    
    # Fix the labels
    fixed_rows = []
    changes = 0
    for row in rows:
        if len(row) < 2:
            continue
        
        filename = row[0]
        old_label = row[1]
        new_label = extract_text_from_filename(filename)
        
        if old_label != new_label:
            changes += 1
            if changes <= 5:  # Show first 5 changes
                print(f"  ✓ {filename}: '{old_label}' -> '{new_label}'")
        
        fixed_rows.append([filename, new_label])
    
    # Write the fixed CSV
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(fixed_rows)
    
    print(f"  Fixed {changes} labels out of {len(rows)} total")
    print(f"  ✅ Saved: {csv_path}")


def main():
    dataset_root = Path('./dataset')
    
    if not dataset_root.exists():
        print(f"Error: Dataset folder not found: {dataset_root}")
        return
    
    # Find all labels.csv files in subdirectories
    labels_files = list(dataset_root.rglob('labels.csv'))
    
    if len(labels_files) == 0:
        print("No labels.csv files found!")
        return
    
    print(f"Found {len(labels_files)} labels.csv files:")
    for f in labels_files:
        print(f"  - {f}")
    
    print("\n" + "="*60)
    print("Starting to fix labels...")
    print("="*60)
    
    for labels_file in labels_files:
        try:
            fix_labels_csv(labels_file)
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    print("\n" + "="*60)
    print("✅ All labels.csv files have been fixed!")
    print("="*60)
    print("\nNext steps:")
    print("1. Verify the fixed labels look correct")
    print("2. Re-train your model with the corrected labels")
    print("3. The old labels are backed up as labels.csv.backup")


if __name__ == '__main__':
    main()
