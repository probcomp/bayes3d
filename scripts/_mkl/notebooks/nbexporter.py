"""

    Exports notebooks to `.py` files using `nbdev.nb_export`.

"""
import argparse
import glob
import os
from nbdev.export import nb_export
from pathlib import Path

NBS = "."
LIB = "../../../bayes3d/_mkl/"

class bcolors:
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    PURPLE = "\033[95m"
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def main():
    lib_path     = Path(__file__).parents[0]/LIB
    rel_lib_path = os.path.relpath(lib_path)

    rel_nbs_path = os.path.relpath(Path(__file__).parents[0]/NBS)
    file_pattern = f"{rel_nbs_path}/**/[a-zA-Z0-9]*.ipynb"

    print(f"{bcolors.BLUE}Trying to export the following files")

    for fname in glob.glob(file_pattern, recursive=True):

        print(f"\t{bcolors.PURPLE}{fname}{bcolors.ENDC}")
        nb_export(fname, lib_path=rel_lib_path)

    print(f"{bcolors.BLUE}to{bcolors.ENDC}")
    print(f"\t{bcolors.PURPLE}{bcolors.BOLD}{rel_lib_path}{bcolors.ENDC}")

if __name__ == "__main__":
    main()
