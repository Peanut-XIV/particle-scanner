import os
import sashimi.helicon_stack as hs
import multiprocessing as mp
from sashimi.utils import make_unique_subdir
from pathlib import Path


if __name__ == "__main__":
	input_dir = Path(input("input directory: "))
	output_dir = Path(input("output directory: "))
	
	if output_dir is None:
		output_dir = make_unique_subdir()
		os.makedirs(output_dir)
		print(f'No output dir was given. Output will be saved on :\n{output_dir}')
	
	error_logs = input_dir.joinpath('error_logs.txt')
	queue = mp.Queue()
	stacking_process = mp.Process(target=hs.parallel_stack, args=(queue, None, error_logs))
	
	scan_directories = [d for d in input_dir.iterdir() if 'scan' in d.name and d.is_dir()]
	for scan_dir in scan_directories:
		queue.put((scan_dir, output_dir))
