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
  python = pkgs.python38; # Specifies Python version. In our case, Python 3.8;
in python
