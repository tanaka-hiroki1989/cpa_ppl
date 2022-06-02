import matplotlib.pyplot as plt
import numpy as np

def plot(data):
    plt.bar(np.arange(len(data)), data, color="#348ABD")
    plt.xlabel("Time (days)")
    plt.ylabel("count of text-msgs received")
    plt.xlim(0, len(data));
    plt.savefig("count_of_text-msgs_received.png")