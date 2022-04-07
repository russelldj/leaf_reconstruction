from pathlib import Path
from ubelt import ensuredir


def pad_filename(
    filename: Path, start_index: int = None, end_index: int = None, pad_length: int = 6
):
    stem = int(filename[start_index:end_index])
    extension = Path(filename).suffix
    formatted_stem = f"{stem:0{pad_length}d}"
    return formatted_stem, extension


def pad_filepath(
    filepath: Path, start_index: int = None, end_index: int = None, pad_length: int = 6
):
    """
    filename:
        The filename to pad. Can be relative or absolute path
    pad_length:
        How digits it should have

    returns:
        The padded filename, at the same depth it was originally
    """
    parent = filepath.parent
    formatted_stem, extension = pad_filename(
        filepath.stem,
        start_index=start_index,
        end_index=end_index,
        pad_length=pad_length,
    )
    output = Path(parent, formatted_stem + extension)
    return output


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


def ensure_dir_normal_bits(folder):
    ensuredir(folder, mode=0o0755)
