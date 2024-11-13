# Copyright (C) 2024 Università degli Studi di Cagliari
# Licensed under BSD 3-Clause License (see LICENSE)
# Authors: Francesco Ratto, Federico Manca (<name>.<surname>@unica.it)

from frontend import frontend 
from backend import backend


def main():
    print("------------- QONNX TO MDC ---------")
    
    import sys
    # Using sys.argv
    path1 = sys.argv[1]
    path2 = sys.argv[2]

    # Using argparse
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('path1', type=str, help='Path for onnxmodel')
    parser.add_argument('path2', type=str, help='Path for results')
    args = parser.parse_args()

    path1 = args.path1
    path2 = args.path2
    
    
    print("------------- Frontend ----------")
    model = frontend(path1)

    print("------------ Backend -----------")
    backend(path2, model)

if __name__ == "__main__":
    main()




