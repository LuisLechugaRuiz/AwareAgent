import os


def list_directories(path: str) -> str:
    """
    Inspect the local directory structure and return a human-readable representation.

    Returns:
        str: A human-readable representation of the directory structure.
    """
    found_files = []

    def process_directory(dir_path, prefix):
        for entry in os.scandir(dir_path):
            if entry.is_file() and not entry.name.startswith("."):
                found_files.append(f"{prefix} {entry.name}")
            elif entry.is_dir() and not entry.name.startswith("."):
                found_files.append(f"{prefix} {entry.name}/")
                process_directory(entry.path, prefix + "  ")

    process_directory(path, "-")

    # Join the list items with line breaks to create a human-readable format
    if found_files:
        result_msg = "\n".join(found_files)
    else:
        result_msg = "No directory yet."
    return result_msg
