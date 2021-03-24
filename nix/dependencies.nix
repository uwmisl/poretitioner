###########################################################################################
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

{ pkgs ? import <nixpkgs> { config = (import ./config.nix); overlays = [ (import ./overlays.nix) ]; }
, python ? (pkgs.callPackage ./python.nix) { inherit pkgs; }
, lib ? pkgs.lib
, stdenv ? pkgs.stdenv
, cudaSupport ? false
}:
let
  # Takes in a python version, spits out run, build, test, and all packages for it.
  getDependenciesForPython = pythonPackages:
    with pythonPackages;
    let
      debugpy = pythonPackages.callPackage ./pkgs/debugpy/default.nix { };
      pytorch = pythonPackages.callPackage ./pkgs/pytorch/default.nix { python = pythonPackages.python; inherit cudaSupport; };
      fast5_research = pythonPackages.callPackage ./pkgs/fast5_research/default.nix { };
    in
    rec {
      run = [
        # Reading/writing TOML documents.
        toml
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
        # tkinter
        # Data visualization
        seaborn
        # Interactive computing
        notebook
        # For interactive builds
        jupyter
        # For validating config and dict inputs
        schema
        # Neural networks
        #torchvision
        # Colorful logs!
        colorlog

        # Pytorch
        pytorch
      ]
      # Oxford Nanopore Tools
      ++ [ fast5_research ]
      ;

      build = [
        # Import sorter
        isort
        # Highly opinionated code-formatter
        black
        # Style-guide enforcer
        flake8
        # Docstring static analyzer
        pydocstyle
      ];

      test = [
        # Testing suite
        pytest
        # Test runner
        pytestrunner
        # Test code coverage generator
        pytestcov
        # Debugpy (Used for debugging in VSCode: https://code.visualstudio.com/docs/python/debugging#_command-line-debugging)
        debugpy
      ];

      all = run ++ build ++ test;
    }
  ;

  pythonDeps = getDependenciesForPython python.pkgs;
in
rec {
  getDependenciesForPython = getDependenciesForPython;
  inherit pythonDeps;

  ###########################################################################################
  #
  # run - This list hosts the project's runtime dependencies (basically, anything you
  #       need to explicitly import in Python)
  #
  ###########################################################################################

  run = [ ]
    ++ pythonDeps.run
  ;

  ###########################################################################################
  #
  # build - This list hosts dependencies that shouldn't be packaged for distribution,
  #         but are still needed for developers. This includes testing frameworks, as well as
  #         tools like linters, git hooks, and static analyzers.
  #
  ###########################################################################################

  build = [
    # Git hooks
    pkgs.pre-commit
    # Nix file style enforcer
    pkgs.nixpkgs-fmt
  ]
  ++ pythonDeps.build
  ;

  ###########################################################################################
  #
  # test- This list hosts the project's test-only dependencies (e.g. test runners).
  #       It should not include any packages that aren't part of the testing infrastructure.
  #
  ###########################################################################################

  test = [
    pkgs.less
  ]
  ++ pythonDeps.test
  ;

  ###########################################################################################
  #
  # all - A list containing all of the above packages.
  #
  ###########################################################################################

  all = run ++ test ++ build;
}
