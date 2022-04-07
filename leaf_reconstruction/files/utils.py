from pathlib import Path


def get_files(
    folder: Path, pattern: str, sort=True, require_dir=False, require_file=False
):
    """
    args:

    folder:
        The input folder
    pattern:
        The file to search for within that folder
    sort:
        Return the sorted generator
    require_file:
        Only return files 
    require_dir:
        Only return directories

    returns:
        a list or generator of Path 's 
    """
    if require_dir and require_file:
        raise ValueError(
            "Both require_dir and require_file set. Nothing satisfies both."
        )

    files = Path(folder).glob(pattern)
    if sort:
        files = sorted(files)

    if require_dir:
        files = [f for f in files if f.is_dir()]

    if require_file:
        files = [f for f in files if f.is_file()]

    return files
