import numpy as np
from sklearn.cluster import DBSCAN

def random(size):
    with open("random-{}.in".format(size), 'w') as f:
        xs = np.random.uniform(-10,10,size)
        ys = np.random.uniform(-10,10,size)
        for x,y in zip(xs,ys):
            f.write("{} {}\n".format(x,y))
        points = np.array([xs,ys]).T
        clustering

def mixture(size):
    with open("mixture-{}.in".format(size), 'w') as f:
        pass

def rings(size):
    with open("rings-{}.in".format(size), 'w') as f:
        pass

all_cases = [random, mixture, rings]

def main():
    for case in all_cases:
        for size in [1e3, 1e4, 1e5, 1e6]:
            case(int(size))

if __name__ == "__main__":
    main()

