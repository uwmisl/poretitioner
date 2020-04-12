def main():
    # The following imports are simply a code-review proof-of-concept to illustrate that
    # dependency management works.

    print("Importing 'numpy'...")
    import numpy  # noqa: F401

    print("\tImported 'numpy'!")

    print("Importing 'pandas'...")
    import pandas  # noqa: F401

    print("\tImported 'pandas'!")

    print("Importing 'h5py'...")
    import h5py  # noqa: F401

    print("\tImported 'h5py'!")

    print("Importing 'dask'...")
    import dask  # noqa: F401

    print("\tImported 'dask'!")

    print("Importing 'matplotlib'...")
    import matplotlib  # noqa: F401

    print("\tImported 'matplotlib'!")

    print("Importing 'seaborn'...")
    import seaborn  # noqa: F401

    print("\tImported 'seaborn'!")

    print("Importing 'iPython notebook'...")
    import notebook  # noqa: F401

    print("\tImported 'iPython notebook'!")

    print("Importing 'jupyter'...")
    import jupyter  # noqa: F401

    print("\tImported 'jupyter'!")

    from matplotlib import pyplot as plt

    print(
        "As proof that these libraries are properly packaged and callable, drawing a basic plot..."
    )
    plt.plot([0, 1], [1, 0])
    plt.title("matplotlib is properly packaged and importable.")
    plt.show()

    print("\tPlotted!")


if __name__ == "__main__":
    main()
