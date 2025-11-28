import argparse
import re
from pathlib import Path

def find_query_dirs(base: Path):
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

	res.sort(key=lambda x: x[1], reverse=True)
	return res

def update_content(dirpath: Path, old_idx: int, new_idx: int):
	"""Update contents of files in dirpath replacing old_idx with new_idx."""

	for filepath in dirpath.rglob("*"):
		if filepath.is_file():
			content = filepath.read_text()

			# Replace occurrences of the old index with the new one
			new_content = content.replace(str(old_idx), str(new_idx))

			if new_content != content:
				filepath.write_text(new_content)
				print(f"Updated {filepath}")
			
			if (filepath.name == f"q{old_idx}.py"):
				new_file = filepath.with_name(f"q{new_idx}.py")
				print(f"Renaming the file: f{filepath} -> {new_file}")
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

def add_query(system: Path, new_idx: int, target_class: str = None):
	query_dirs = find_query_dirs(system)

	# Shift existing directories and update contents
	for dirpath, idx in query_dirs:
		if not dirpath.name.startswith("Q"):
			continue  # Only process Q directories

		if idx >= new_idx:
			update_content(dirpath, idx, idx + 1)

			new_dir = dirpath.with_name(f"Q{idx + 1}")
			print(f"Renaming directory: {dirpath} -> {new_dir}")
			dirpath.rename(new_dir)
	
	# Create new query directory if target_class is provided
	if target_class:
		create_query_dir(system, new_idx, target_class)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Manage System query indices by inserting and shifting Q directories.")

	parser.add_argument("--target_class", required=True, type=str, help="Class subfolder (e.g., aggregation, derivation)")
	parser.add_argument("--index", required=True, type=int, help="Index of the query to add (e.g., 2)")

	args = parser.parse_args()

	for system in [Path("lotus"), Path("palimpzest"), Path("blendsql")]:
		add_query(Path("blendsql"), args.index, args.target_class)