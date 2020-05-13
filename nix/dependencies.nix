# ##########################################################################################
#
# dependencies.nix
#
###########################################################################################
#
# This Nix expression hosts the project's dependencies. There are three sections:
#
#   run   - Anything that needs to be available at runtime (e.g. you need to import it in python).
#
#   build - Packages you don't want packaged in the final build, but still need to build
#           the project (e.g. pre-commit hooks linters)
#
#   test  - Test-only dependencies (e.g code coverage packages)
#
# It takes in Nixpkgs and the Python package corresponding to a Python version.
#
###########################################################################################

{ pkgs, python, lib ? pkgs.lib, stdenv ? pkgs.stdenv, cudaSupport ? false }:
let precommit = (import ./pkgs/pre-commit/pre-commit.nix) { inherit python; };
    minknow = (pkgs.callPackage ./pkgs/minknow/minknow.nix) { inherit python; };
in with python.pkgs; rec {

  ###########################################################################################
  #
  # run - This list hosts the project's runtime dependencies (basically, anything you
  #       need to explicitly import in Python)
  #
  ###########################################################################################

  run = [
    # Numerical computation library
    numpy
    # Data manipulation and analysis
    pandas
    # Hierarchical Data Format utilities
    h5py
    # Parallel computing library
    dask
    # Charts and plotting library
    matplotlib
    tkinter
    # Data visualization
    seaborn
    # Interactive computing
    notebook
    # For interactive builds
    jupyter
    # Neural networks
    torchvision

    # MinKnow
    minknow

  ] ++ lib.optional (cudaSupport) pytorchWithCuda
    ++ lib.optional (!cudaSupport) pytorchWithoutCuda;

  ###########################################################################################
  #
  # build - This list hosts dependencies that shouldn't be packaged for distribution,
  #         but are still needed for developers. This includes testing frameworks, as well as
  #         tools like linters, git hooks, and static analyzers.
  #
  ###########################################################################################

  build = [
    # Git hooks
    precommit
    # Import sorter
    isort
    # Highly opinionated code-formatter
    black
    # Style-guide enforcer
    flake8
    # Docstring static analyzer
    pydocstyle
    # Serialization protocol (used for interfacing with MinKNOW/MinION)
    pkgs.protobuf
    protobuf
    grpcio
    grpcio-tools
  ];

  ###########################################################################################
  #
  # test- This list hosts the project's test-only dependencies (e.g. test runners).
  #       It should not include any packages that aren't part of the testing infrastructure.
  #
  ###########################################################################################

  test = [
    # Testing suite
    pytest
    # Test runner
    pytestrunner
    # Test code coverage generator
    pytestcov
  ];

  ###########################################################################################
  #
  # all - A list containing all of the above packages.
  #
  ###########################################################################################

  all = run ++ test ++ build;
}
