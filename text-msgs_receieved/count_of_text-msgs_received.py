import matplotlib.pyplot as plt
import numpy as np
import torch

count_data = torch.from_numpy(np.loadtxt("txtdata.txt"))
n_count_data = len(count_data)
plt.bar(np.arange(n_count_data), count_data, color="#348ABD")
plt.xlabel("Time (days)")
plt.ylabel("count of text-msgs received")
plt.xlim(0, n_count_data);
plt.savefig("count_of_text-msgs_received.png")