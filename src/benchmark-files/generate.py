import numpy as np
from sklearn.cluster import DBSCAN

def random(size):
    with open("random-{}.in".format(size), 'w') as f:
        xs = np.random.uniform(-10,10,size)
        ys = np.random.uniform(-10,10,size)
        for x,y in zip(xs,ys):
            f.write("{} {}\n".format(x,y))

def mixture(size):
    with open("mixture-{}.in".format(size), 'w') as f:
        pass

def rings(size):
    radius = 8
    noise_num = int(size * 0.1)
    ring_num = size - noise_num
    ns = np.random.normal(0, 1, (ring_num, 2))
    ns = ns/np.linalg.norm(ns, axis=1, keepdims=True)
    rs = radius + np.random.normal(0, 0.5, (ring_num, 1))
    noises = np.random.uniform(-10,10,(noise_num, 2))
    points = np.concatenate([ns * rs, noises], axis=0)
    with open("rings-{}.in".format(size), 'w') as f: 
        for x,y in points:
            f.write("{} {}\n".format(x,y))

all_cases = [mixture, rings]

def main():
    for case in all_cases:
        for size in [1e3, 1e4, 1e5, 1e6]:
            case(int(size))

if __name__ == "__main__":
    main()

