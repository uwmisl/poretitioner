###########################################################################################
#
# python.nix
#
###########################################################################################
#
# This expression customizes our Python environment, including package overrides.
#
#   - This is useful for things like deciding which backend graphics matplotlib should use.
#
###########################################################################################

{ pkgs }:
let
  python = pkgs.python37; # Specifies Python version. In our case, Python 3.7;

  # The following code customizes our Python packages, like Cuda vs. Non-Cuda Pytorch.
  pythonWithOverrides = (python.withPackages (pypkgs:
    let
      # Use QT as backend for matplotlib.
      matplotlib = (pypkgs.matplotlib.override { enableTk = true; });
    in [ matplotlib ]));
in pythonWithOverrides
