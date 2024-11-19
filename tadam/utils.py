import hashlib
import json
import os
from typing import Any, Iterable, Optional

import joblib
import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype


def load(path: str | os.PathLike, hash_type: bool | Iterable[str] = False) -> Any:
    """Load an object from a file using joblib.load.

    :param hash_type: If True, the hash of the object is computed and verified against
        the hash stored next to the file. If hash is an iterable of strings, the hash is
        computed using the specified attributes of the object. If False, no hash is
    :param path: Path to file to load from.
    :return: Loaded object
    """
    if hash_type:
        hash_from_file = read_hash(path=path, hash_type=hash_type)
        hash_from_path = calculate_hash(
            path=path, hash_type=hash_from_file.keys(), save=False
        )
        assert (
            hash_from_file == hash_from_path
        ), f"Hashes do not match: {hash_from_file} != {hash_from_path}"
        print(hash_from_file)
    return joblib.load(path)


def dump(
    obj: Any,
    path: str | os.PathLike,
    compress: bool | tuple[str, int] = True,
    protocol: Optional[int] = None,
    hash_type: str | Iterable[str] | None = None,
) -> None:
    """Dump an object to a file using joblib.dump.

    Passes the compress and protocol arguments to joblib.dump. If compress is True, the
    default compression method is used (gzip, level 6). If compress is a tuple, it is
    passed to joblib.dump as is. If compress is False, no compression is used.

    :param obj: Serializable object
    :param path: Path to file to dump to. If the file exists, it will be overwritten.
     if the directory does not exist, it will be created.
    :param compress: Tuple of compression method and level, or True or False.
    :param protocol: Pickle protocol to use. If None, the default protocol is used.
    :param hash_type: If True, the hash of the object is computed and stored with the
     file appending the hash-name as an extension. If hash is an iterable of strings,
     the hash is computed using the specified attributes of the object. If False, no
     hash is computed. Available hash algorithms are those in
     hashlib.algorithms_guaranteed.
    :return: None
    """

    path_dir = os.path.expandvars(os.path.split(path)[0])
    os.makedirs(path_dir, exist_ok=True)

    if compress:
        if isinstance(compress, tuple):
            joblib.dump(obj, path, compress=compress, protocol=protocol)
        else:
            joblib.dump(obj, path, compress=("gzip", 6), protocol=protocol)
    else:
        joblib.dump(obj, path, protocol=protocol)

    if hash_type:
        hash_read = calculate_hash(path=path, hash_type=hash_type, save=True)
        print(hash_read)


def read_hash(
    path: str | os.PathLike, hash_type: bool | Iterable[str]
) -> dict[str, str]:
    """Read the hash from a file. The hash is appended to the file name.

    :param path: Path to file to read hash from.
    :param hash_type:
    :return:
    """

    hash_read = dict()

    if isinstance(hash_type, bool):
        hash_type = hashlib.algorithms_guaranteed
    else:
        unsupported_hashes = [
            h for h in hash_type if h not in hashlib.algorithms_guaranteed
        ]
        assert len(unsupported_hashes) == 0, (
            f"Hashes {unsupported_hashes} not " f"available."
        )

    hash_files = {h: str(path) + f".{h}" for h in hash_type}
    files_not_found = [h for h, f in hash_files.items() if not os.path.isfile(f)]
    assert len(files_not_found) == 0, f"Hash files for {files_not_found} not found."

    for h, save_path in hash_files.items():
        with open(save_path, "r") as f:
            hash_read[h] = f.read()
    return hash_read


def calculate_hash(
    path: str | os.PathLike,
    hash_type: str | Iterable[str] = "sha256",
    save: bool = False,
) -> dict[str, str]:
    """Calculate the hash of a file.

    The hash is appended to the file name. If save is True, the hash is saved to a file
     with the same name as the original file, but with the hash appended as an
     extension.

    :param path: Path to file to calculate hash of. String or os.PathLike
    :param hash_type: Hash algorithm to use. If "all", all algorithms in
     hashlib.algorithms_guaranteed which are compatible with updates and reading blocks.
    :param save: bool to save the hash to a file with the same name as the original
    file, but with the hash appended as an extension.
    :return: dictionary of hash_type: hash
    """

    shake_length = 32  # variable length not implemented yet

    if isinstance(hash_type, str):
        match hash_type:
            case "all":
                hash_type = hashlib.algorithms_guaranteed
            case "common":
                hash_type = ["md5", "sha1", "sha256", "sha512"]
            case "default":
                hash_type = ["sha256"]
            case _:
                hash_type = [hash_type]
    elif isinstance(hash_type, bool):
        if hash_type:
            hash_type = ["sha256"]

    hash_type = set(hash_type)

    for h in hash_type:
        assert h in hashlib.algorithms_guaranteed, f"Hash {h} not available."

    hash_read = dict()
    for h in hash_type:
        save_path = str(path) + f".{h}"
        hasher = hashlib.new(h)
        with open(path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                hasher.update(byte_block)
            if h == "shake_128" or h == "shake_256":
                hash_read[h] = hasher.hexdigest(length=shake_length)  # type: ignore
            else:
                hash_read[h] = hasher.hexdigest()
        if save:
            with open(save_path, "w") as f:
                f.write(hash_read[h])

    if save:
        save_path = str(path) + ".osstat.json"
        os_stat = os.stat(path)
        os_stat_as_dict = {
            k: getattr(os_stat, k) for k in dir(os_stat) if k.startswith("st_")
        }
        with open(save_path, "w") as f:
            json.dump(os_stat_as_dict, f, indent=4)
    return hash_read


def summary(data: pd.DataFrame) -> pd.DataFrame:
    """Create a summary of a pandas DataFrame.

    :param data: pandas DataFrame
    :return: pandas DataFrame with columns
        [name, non-nulls, nulls, type, unique, memory, min, max, mean, std,
        p01, p25, p50, p75, p99, top]
    """
    return pd.DataFrame(
        {
            "name": data.columns,
            "non-nulls": len(data) - data.isnull().sum().values,
            "nulls": data.isnull().sum().values,
            "type": data.dtypes.values,
            "unique": data.nunique().values,
            "memory": data.memory_usage(index=False, deep=True),
            "min": [
                data[col].min() if is_numeric_dtype(data[col]) else np.nan
                for col in data
            ],
            "max": [
                data[col].max() if is_numeric_dtype(data[col]) else np.nan
                for col in data
            ],
            "mean": [
                data[col].mean() if is_numeric_dtype(data[col]) else np.nan
                for col in data
            ],
            "std": [
                data[col].std() if is_numeric_dtype(data[col]) else np.nan
                for col in data
            ],
            "p01": [
                np.percentile(data[col], 1)
                if is_numeric_dtype(data[col]) and not is_bool_dtype(data[col])
                else np.nan
                for col in data
            ],
            "p25": [
                np.percentile(data[col], 25)
                if is_numeric_dtype(data[col]) and not is_bool_dtype(data[col])
                else np.nan
                for col in data
            ],
            "p50": [
                np.percentile(data[col], 50)
                if is_numeric_dtype(data[col]) and not is_bool_dtype(data[col])
                else np.nan
                for col in data
            ],
            "p75": [
                np.percentile(data[col], 75)
                if is_numeric_dtype(data[col]) and not is_bool_dtype(data[col])
                else np.nan
                for col in data
            ],
            "p99": [
                np.percentile(data[col], 99)
                if is_numeric_dtype(data[col]) and not is_bool_dtype(data[col])
                else np.nan
                for col in data
            ],
            "top": data.mode(axis=0).head(1).values[0, :],
        }
    )
