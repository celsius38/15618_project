import time
import argparse
import os
from typing import Tuple
import logging 
from collections import namedtuple
import subprocess
import numpy as np
import pandas as pd


# constant definitions
EPS = 1
MIN_PTS = 10 
BINARY = "./src/dbscan-release"
LOGS_DIR = "logs"
# TODO: add more scanner types and input files
SCANNER_TYPES = ["seq", "gdbscan"]
IN_FILES = ["src/benchmark-files/" + x for x in 
    [f"{case}-{n}.in"  
        for case in ["random"]  
        for n in [1000, 10000, 100000]]
]


class ScanResult(object):
    def __init__(self, time=None, num_clusters=None, labels=None):
        self.time = time
        self.num_clusters = num_clusters
        self.labels = labels
    
    def to_file(self, path):
        directory = os.path.dirname(path)
        os.makedirs(directory, exist_ok=True)
        with open(path, 'w') as f:
            f.write(f"Taking {self.time} ms\n")
            f.write("====================\n")
            f.write(f"{self.num_clusters} clusters\n")
            self.labels = np.array(self.labels) 
            self.labels.tofile(f, sep='\n')

    def __str__(self):
        return f"{self.num_clusters} clusters in {self.time} ms"

# @deprecated
# def get_logger():
    # logger = logging.getLogger(__name__)
    # logger.setLevel(logging.DEBUG)
    # handler = logging.StreamHandler()
    # handler.setLevel(logging.DEBUG)
    # handler.setFormatter(logging.Formatter('%(name)s - %(levelname)s - %(message)s'))
    # logger.addHandler(handler)
    # return logger


def get_args():
    """
    specify custom eps, minPts through:
        `python3 checker.py --eps 0.5 --minPts 10`
    specify to only check for sequential (default is check for all):
        `python3 checker.py --seq`
    specify to only run for a single file:
        `python3 checker.py -in src/benchmark-files/random-1000.in`
    """
    parser = argparse.ArgumentParser("Checker.py arguments")
    parser.add_argument("--check", action="store_true",
                        help="Turn on correctness check")
    parser.add_argument("--eps", type=float, default=1.)
    parser.add_argument("--minPts", type=int, default=10)
    parser.add_argument("--input", type=str, help="Input file, default all IN_FILES", 
                        metavar="src/benchmark-files/random-1000.in",
                        default=IN_FILES)
    parser.add_argument("--scannerTypes", type=str, nargs='+',
                        choices=["seq"],
                        default=SCANNER_TYPES,
                        help="List of scanner types to test with")
    return parser.parse_args()


def timeit(fun):
    def wrapper(*args, **kwargs):
        start = time.time()
        res = fun(*args, **kwargs)
        print("taking {time.time() - start}s")
        return res
    return wrapper


def dbscan_ref(in_file:str, eps:float, min_pts:int) -> ScanResult:
    from sklearn.cluster import DBSCAN
    MS_PER_S = 1e3
    points = np.loadtxt(in_file)
    start = time.time()
    clustering = DBSCAN(eps=eps, min_samples=min_pts, n_jobs=-1).fit(points)
    labels = clustering.labels_
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    return ScanResult(
            time= (time.time()-start) * MS_PER_S,
            num_clusters=num_clusters, 
            labels=labels)


def dbscan_invoke(scanner_type:str, in_file:str, eps:float, min_pts:int) -> ScanResult:
    args = ' '.join([
        BINARY, "--input", in_file, "--eps", str(eps), "--minPts", str(min_pts),
        "--" + scanner_type
    ])
    result = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    if(result.returncode) != 0:
        return None
    stdout = result.stdout.decode("UTF-8")
    res = ScanResult()
    last_ret = -1
    while True:
        new_ret = stdout.find("\n", last_ret+1)
        line = stdout[last_ret+1:new_ret]
        last_ret = new_ret
        if line.startswith("Taking"):
            res.time = float(line[7:line.find("ms")])
        if line.endswith("clusters"):
            res.num_clusters = int(line[0: line.find(' ')])
            break
    res.labels = np.fromstring(stdout[new_ret+1:],dtype=int,sep='\n')
    return res


def check_label(gold: ScanResult, result: ScanResult) -> bool:
    """
    check if the label output match with that of gold
    notice that same cluster might have different label in different run
    """
    if len(gold.labels) != len(result.labels):
        return False
    if gold.num_clusters != result.num_clusters:
        return False
    mapping = {}  # gold_label -> label
    for gold, label in zip(gold.labels, result.labels):
        if gold == -1:
            if label != -1:
                return False
        else:
            if gold in mapping:
                if label != mapping[gold]:
                    return False
            else:
                mapping[gold] = label
    return True


def main():
    args = get_args()
    if "input" not in vars(args):
        args.input = IN_FILES
    if "scannerTypes" not in vars(args):
        args.scannerTypes = SCANNER_TYPES
    # list of result for each input
    record_list = []
    print(args)
    for input in args.input:
        print(f"===== case: {input} =====")
        logs_dir = os.path.join(LOGS_DIR, os.path.splitext(os.path.basename(input))[0])
        record = {}  
        gold = dbscan_ref(input, args.eps, args.minPts) 
        print(f"ref: {gold}")
        gold.to_file(os.path.join(logs_dir, "ref.out"))
        record["ref"] = gold.time
        for scanner_type in args.scannerTypes:
            res = dbscan_invoke(scanner_type, input, args.eps, args.minPts)
            res.to_file(os.path.join(logs_dir, f"{scanner_type}.out"))
            if args.check:
                if not check_label(gold, res):
                    print(f"{scanner_type} correctness check failed")
                    record[scanner_type] = np.nan 
                    continue
            record[scanner_type] = res.time 
            print(f"{scanner_type}: {res}")
        record_list.append(record) 
    basenames = [os.path.basename(input) for input in args.input]
    df = pd.DataFrame(record_list, index=basenames)
    print(df)


if __name__ == "__main__":
    main()
