from pathlib import Path
import os

scan_path = Path(input("scan folder (ex: scan1, scan2,...): "))
count = 1
for stack in scan_path.iterdir():
	image_folder = stack.joinpath('E1400')
	images = sorted(image_folder.glob('*.jpg'))
	os.remove(images[0])
	print(f"deleted {count} file(s)", flush=True)
print('Finished')
