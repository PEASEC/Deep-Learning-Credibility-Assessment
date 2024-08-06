import time
import datetime


def format_time(elapsed):
    """
    Takes a time in seconds and returns a string hh:mm:ss
    """
    # Round to the nearest second.
    elapsed_rounded = int(round(elapsed))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def read_simple_config_file(file_path: str) -> dict:
    content: dict = {}
    with open(file_path, "r") as file:
        for line in file:
            parts = line.split("=")
            key = parts[0].strip()
            value = parts[1].strip()
            content[key] = value

    return content