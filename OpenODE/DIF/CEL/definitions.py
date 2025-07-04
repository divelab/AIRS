import pathlib
import os


ROOT_DIR = pathlib.Path(__file__).parent.parent.absolute()
STORAGE_DIR = ROOT_DIR / 'storage'
os.environ['ROOT_DIR'] = str(ROOT_DIR)
os.environ['STORAGE_DIR'] = str(STORAGE_DIR)

# Check if the storage directory exists
if not STORAGE_DIR.exists():
    # Ask user for a correct storage directory
    true_storage_dir = pathlib.Path(input('Enter a storage directory: '))
    # Create a soft link at STORAGE_DIR to true_storage_dir
    STORAGE_DIR.symlink_to(true_storage_dir)
    print(f'Soft link created at {STORAGE_DIR} to {true_storage_dir}')

assert STORAGE_DIR.exists(), f'Storage directory {STORAGE_DIR} does not exist'