import os
import zipfile
from typing import Union


def unzip_file(zip_name: str):
    filename = os.path.basename(zip_name)
    dirname = os.path.dirname(zip_name)
    with zipfile.ZipFile(zip_name) as zip_ref:
        print("Unzipping {}...".format(filename))
        zip_ref.extractall(dirname)


def get_file_and_unzip(full_name: str, unzip: bool = True):
    dirname = os.path.dirname(full_name)

    if os.path.exists(full_name):
        return full_name
    elif os.path.exists(full_name + ".zip"):
        if unzip:
            unzip_file(full_name + ".zip")
            return full_name
        else:
            return full_name + ".zip"
    elif os.path.exists(dirname + ".zip") and not os.path.exists(dirname):
        if unzip:
            unzip_file(dirname + ".zip")
            return get_file_and_unzip(full_name, unzip)
        else:
            return dirname + ".zip"
    else:
        return None


def get_repository_path(full_name: str, return_none_on_error: bool = False) -> Union[str, None]:
    """
    Executes some lookups for a file_name. If the full_name ends with .zip, this postfix will be removed.
    This function will look at the following paths, where dirname is the
    the name of the files parent directory and filename is the name of the file."
    Therefore: `fullname := dirname + "/" + filename`.
    The first occurrence found will be used.

    * 1: ./dirname/filename
    * 2: ./dirname/filename.zip -> extract -> ./dirname/filename
    * 3: ./dirname.zip -> extract -> ./dirname/filename
    * 4: ./additional-data/dirname/filename
    * 5: ./additional-data/dirname/filename.zip -> extract -> ./additional-data/dirname/filename
    * 6: ./additional-data/dirname.zip -> extract -> ./additional-data/dirname/filename
    * 7: replace "build" with "prebuild" in fullname and repeat 1 - 3

    Args:
        full_name (str):                The name of the file
        return_none_on_error (bool):    If set to true the function will return None if no file is found. Otherwise it
                                        will raise an error.

    Returns:
        A transformed file name.
    """

    if full_name is None:
        return None

    full_name = os.path.relpath(full_name)

    final_path = [None] * 3

    def filtered_path():
        return [p for p in final_path if p is not None]

    final_path[0] = get_file_and_unzip(full_name, True)
    final_path[1] = get_file_and_unzip(os.path.join("additional-data", full_name), len(filtered_path()) <= 0)

    if "build" in full_name:
        full_name = full_name.replace("build", "prebuild")
        final_path[2] = get_file_and_unzip(full_name, len(filtered_path()) <= 0)

    if len(filtered_path()) <= 0:
        if return_none_on_error:
            return None
        else:
            raise IOError("Cannot find file {} in repository.".format(full_name))

    final_path = filtered_path()
    for i in range(1, len(final_path)):
        print("Using file '{}' but also found '{}'.".format(final_path[0], final_path[i]))

    return final_path[0]
