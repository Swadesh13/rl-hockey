import pickle
import sys

with open(sys.argv[1], "rb") as f:
    data = pickle.load(f)

print(data)
