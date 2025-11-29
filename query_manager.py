import argparse
import re
from pathlib import Path

def find_query_dirs(base: Path, reverse: bool = True):
	"""Return list of (path, number) for subdirectories named Q<digits> under base."""
	res = []
	if not base.exists():
		return res
	
	assert( base.joinpath('queries').exists())
	queries_base = base / 'queries'
    
	for _class in queries_base.iterdir():
		
		for query_dir in _class.iterdir():
			if query_dir.is_dir():
				query_index = re.fullmatch(r"Q(\d+)$", query_dir.name)
				if query_index:
					res.append((query_dir, int(query_index.group(1))))

	res.sort(key=lambda x: x[1], reverse=reverse)
	return res

def find_eval_dirs(eval_base: Path, reverse: bool = True):
	assert(eval_base.exists())
	res = []

	for _class in eval_base.iterdir():
		if not _class.is_dir():
			continue
		for eval_dir in _class.iterdir():
			if eval_dir.is_dir():
				eval_index = re.fullmatch(r"Q(\d+)$", eval_dir.name)
				if eval_index:
					res.append((eval_dir, int(eval_index.group(1))))	

	res.sort(key=lambda x: x[1], reverse=reverse)
	return res	

def update_query_content(dirpath: Path, old_idx: int, new_idx: int):
	"""Update contents of files in dirpath replacing old_idx with new_idx."""

	for filepath in dirpath.rglob("*"):
		if filepath.is_file():
			content = filepath.read_text()

			# Replace Q{old_idx} and q{old_idx} (even in larger numbers like Q10)
			new_content = content.replace(f"Q{old_idx}", f"Q{new_idx}").replace(f"q{old_idx}", f"q{new_idx}")

			if new_content != content:
				filepath.write_text(new_content)
				print(f"Updated {filepath}")
			
			if (filepath.name == f"q{old_idx}.py"):
				new_file = filepath.with_name(f"q{new_idx}.py")
				print(f"Renaming the file: {filepath} -> {new_file}")
				filepath.rename(new_file)

def update_eval_content(dirpath: Path, old_idx: int, new_idx: int):
	scripts_path = dirpath / "eval_scripts"

	for filepath in scripts_path.rglob("*"):
		if filepath.is_file():
			content = filepath.read_text()

			new_content = content.replace(f"Q{old_idx}", f"Q{new_idx}").replace(f"q{old_idx}", f"q{new_idx}")

			if new_content != content:
				filepath.write_text(new_content)
				print(f"Updated {filepath}")

			if f"q{old_idx}" in filepath.name or f"Q{old_idx}" in filepath.name:
				new_name = filepath.name.replace(f"q{old_idx}", f"q{new_idx}").replace(f"Q{old_idx}", f"Q{new_idx}")
				new_file = filepath.with_name(new_name)
				print(f"Renaming the file: {filepath} -> {new_file}")
				filepath.rename(new_file)

	bash_script = Path(dirpath / "eval_runs.sh")
	if bash_script.exists() and bash_script.is_file():
		content = bash_script.read_text()
		new_content = content.replace(f"Q{old_idx}", f"Q{new_idx}").replace(f"q{old_idx}", f"q{new_idx}")
		if new_content != content:
			bash_script.write_text(new_content)
			print(f"Updated {bash_script}")		

	results_path = dirpath / "results"
	if results_path.exists():
		for filepath in results_path.rglob("*"):
			if filepath.is_file():
				if f"Q{old_idx}" in filepath.name or f"q{old_idx}" in filepath.name:
					new_name = filepath.name.replace(f"Q{old_idx}", f"Q{new_idx}").replace(f"q{old_idx}", f"q{new_idx}")
					new_file = filepath.with_name(new_name)
					print(f"Renaming the file: {filepath} -> {new_file}")
					filepath.rename(new_file)
	
def create_query_dir(system: Path, idx: int, target_class: str):
	"""Create a new query directory with Q{idx}.py and runs.sh files."""

	queries_base = system / 'queries' / target_class
	query_dir = queries_base / f"Q{idx}"
	
	# Create directory if it doesn't exist
	query_dir.mkdir(parents=True, exist_ok=True)
	print(f"Created directory: {query_dir}")
	
	# Create Q{idx}.py file
	py_file = query_dir / f"q{idx}.py"
	py_file.write_text(f"# Query {idx}\n")
	print(f"Created file: {py_file}")
	
	# Create runs.sh file
	runs_file = query_dir / "runs.sh"
	runs_file.write_text("#!/bin/bash\n# Runs for Q{}\n".format(idx))
	runs_file.chmod(0o755)
	print(f"Created file: {runs_file}")

def create_eval_dir(idx: int, target_class: str):
	"""Create a new evaluation directory with eval_scripts and necessary files."""

	eval_base = Path("evaluation") / target_class
	eval_dir = eval_base / f"Q{idx}"
	scripts_dir = eval_dir / "eval_scripts"
	
	# Create directories if they don't exist
	scripts_dir.mkdir(parents=True, exist_ok=True)
	print(f"Created directory: {scripts_dir}")
	
	# Create eval_runs.sh file
	runs_file = scripts_dir / "eval_runs.sh"
	runs_file.write_text("#!/bin/bash\n# Evaluation Runs for Q{}\n".format(idx))
	runs_file.chmod(0o755)
	print(f"Created file: {runs_file}")

def add_query(base: Path, new_idx: int, target_class: str = None):
	# query_dirs = find_query_dirs(base)

	# # Shift existing directories and update contents
	# for dirpath, idx in query_dirs:
	# 	if not dirpath.name.startswith("Q"):
	# 		continue  # Only process Q directories

	# 	if idx >= new_idx:
	# 		update_query_content(dirpath, idx, idx + 1)

	# 		new_dir = dirpath.with_name(f"Q{idx + 1}")
	# 		print(f"Renaming directory: {dirpath} -> {new_dir}")
	# 		dirpath.rename(new_dir)
	
	# # Create new query directory if target_class is provided
	# if target_class:
	# 	create_query_dir(base, new_idx, target_class)

	eval_dirs = find_eval_dirs(base)

	# Shift evaluation directories as well
	for dirpath, idx in eval_dirs:
		print("dirpath:", dirpath, "idx:", idx)
		if idx >= new_idx:
			update_eval_content(dirpath, idx, idx + 1)

			new_dir = dirpath.with_name(f"Q{idx + 1}")
			print(f"Renaming evaluation directory: {dirpath} -> {new_dir}")
			dirpath.rename(new_dir)

	if target_class:
		create_eval_dir(new_idx, target_class)

def remove_query(system: Path, idx: int):
	query_dirs = find_query_dirs(system, reverse=False)

	# Remove the specified directory and update contents
	for dirpath, current_idx in query_dirs:
		if not dirpath.name.startswith("Q"):
			continue  # Only process Q directories

		if current_idx == idx:
			print(f"Removing directory: {dirpath}")
			for item in dirpath.rglob("*"):
				if item.is_file():
					item.unlink()
			dirpath.rmdir()
		elif current_idx > idx:
			update_query_content(dirpath, current_idx, current_idx - 1)

			new_dir = dirpath.with_name(f"Q{current_idx - 1}")
			print(f"Renaming directory: {dirpath} -> {new_dir}")
			dirpath.rename(new_dir)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Manage System query indices by inserting and shifting Q directories.")

	parser.add_argument("--target_class", required=True, type=str, help="Class subfolder (e.g., aggregation, derivation)")
	parser.add_argument("--index", required=True, type=int, help="Index of the query to add (e.g., 2)")
	# parser.add_argument("--remove", action="store_true", help="If set, remove the query at the specified index instead of adding.")

	args = parser.parse_args()

	# for system in [Path("lotus"), Path("palimpzest"), Path("blendsql")]:
	# 	if (args.remove):
	# 		remove_query(system, args.index)
	# 	else:
	# 		add_query(system, args.index, args.target_class)

	add_query(Path("evaluation"), args.index, args.target_class)