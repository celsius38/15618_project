import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from checker import dbscan_invoke, dbscan_ref

PLOT_DIR = "plot"

def get_args():
    parser = argparse.ArgumentParser("plot.py arguments")
    parser.add_argument("--eps", type=float, default=1.)
    parser.add_argument("--minPts", type=int, default=10)
    parser.add_argument("--input", type=str, help="Input file", nargs="+",
                        default=["src/benchmark-files/random-1000.in"],
                        metavar="src/benchmark-files/random-1000.in")
    parser.add_argument("--scannerTypes", type=str, nargs="+",
                        default=["ref"],
                        choices=["ref", "seq", "gdbscan"],
                        help="List of scanner types to test with")
    return parser.parse_args() 

def plot(scanner_type: str, in_file: str, eps: float, min_points: int):
    points = np.loadtxt(in_file)
    result = None
    if scanner_type == "ref": 
        result = dbscan_ref(in_file, eps, min_points) 
    else:
        result = dbscan_invoke(scanner_type, in_file, eps, min_points)
    # mapping from cluster to color
    colors = cm.rainbow(np.linspace(0, 1, result.num_clusters + 1)) 
    colors = np.concatenate([colors,[[0,0,0,0.2]]], axis=0)
    colors = colors[result.labels]
    plt.scatter(points[:,0], points[:,1], color=colors)
    path = os.path.join(PLOT_DIR, os.path.splitext(os.path.basename(in_file))[0],
                        f"{scanner_type}.png") 
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"saving plot to {path}")
    plt.savefig(path)
    
def main(args):
    print(vars(args)) 
    for in_file in args.input:
        for scanner_type in args.scannerTypes:
            plot(scanner_type, in_file, args.eps, args.minPts)
    

if __name__ == "__main__":
    args = get_args()
    main(args)
    
