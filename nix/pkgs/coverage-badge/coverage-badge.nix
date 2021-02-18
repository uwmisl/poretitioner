# ##########################################################################################
#
# coverage-badge.nix
#
###########################################################################################
#
# The `coverage-badge` generates a neat little coverage badge to show in our repo.
#
# Why are we building it in manually instead of downloading it from the Nix store?
# Sadly, it's not in the Nix store yet.
#
###########################################################################################

let
  pkgs = import <nixpkgs> { };
  defaultPython = pkgs.python37;
in { python ? defaultPython }: with python.pkgs; callPackage ./default.nix { }
