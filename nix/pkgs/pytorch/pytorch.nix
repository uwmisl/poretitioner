###########################################################################################
#
# pytorch.nix
#
###########################################################################################
#
# This provides a stable version of pytorch.
# Details on how to use it can be found here [1, 2].
#
# Why are we building it in manually instead of downloading it from the Nix store?
# Sadly, it's not in the Nix store yet.
#
# [1] - https://github.com/microsoft/debugpy
# [2] - https://code.visualstudio.com/docs/python/debugging#_command-line-debuggingd
#
###########################################################################################

{ pkgs ? import <nixpkgs> { config = (import ../../config.nix); overlays = [ (import ../../overlays.nix) ]; }
, lib ? pkgs.lib
, python ? pkgs.python39
, cudaSupport ? false
}:
with python.pkgs;
callPackage ./default.nix { inherit python cudaSupport; }
