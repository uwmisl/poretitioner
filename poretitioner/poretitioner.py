# The following imports are simply a code-review proof-of-concept to illustrate that
# dependency management works.
import dask.bag as db  # noqa: F401
import jupyter  # noqa: F401
import numpy  # noqa: F401
import pandas  # noqa: F401
from dask.diagnostics import ProgressBar  # noqa: F401
from matplotlib import pyplot as plt


def main():
    # Stub for the command line
    print("This print statement is a placeholder.")

    plt.plot([0, 1], [1, 0])
    plt.title("This plot proves that matplotlib is working.")
    plt.show()


if __name__ == "__main__":
    main()
