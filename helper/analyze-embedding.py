"""
analyze-embedding.py

python analyze-embedding.py ../resources/embeddings/glove.twitter.27B.25d.txt

Script to analyze the std and mean of a embedding file
"""

import sys
from tqdm import tqdm
import torch

if __name__ == "__main__":
    file_name = sys.argv[1]

    global_values = []
    value_len = -1
    with open(file_name, "r", encoding="utf-8") as file_from:
        loop = tqdm(enumerate(file_from), desc="Analyzing embeddings")
        for index, line in loop:
            line_stripped: str = line.rstrip()
            values = line_stripped.split(" ")[1:]
            if value_len <= -1:
                value_len = len(values)

            if len(values) != value_len:
                raise ValueError("Expected {} elements in line {}.".format(values, index))

            global_values.append([float(v) for v in values])

    print("Calculating values")
    np_array = torch.FloatTensor(global_values)
    mean = torch.mean(np_array, dim=0)
    std = torch.std(np_array, dim=0)
    std_corrected = torch.std(np_array - mean, dim=0)

    print("Mean")
    print(mean.numpy())
    print("Std")
    print(std.numpy())
