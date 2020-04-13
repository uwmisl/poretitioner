import torch  # noqa: F401
import torchvision  # noqa: F401
from matplotlib import pyplot as plt


def main():
    # The following imports are simply a code-review proof-of-concept to illustrate that
    # dependency management works.
    print(
        "As proof that these libraries are properly packaged and callable, drawing a basic plot..."
    )
    plt.plot([0, 1], [1, 0])
    plt.title("matplotlib is properly packaged and importable.")
    plt.show()

    print("\tPlotted!")


if __name__ == "__main__":
    main()
