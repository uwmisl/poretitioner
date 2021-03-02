# ##########################################################################################
#
# pytype.nix
#
###########################################################################################
#
# The `ont-fast5-api` provides some APIs for handling Fast5 files.
#
# Why are we building it in manually instead of downloading it from the Nix store?
# Sadly, it's not in the Nix store yet.
#
###########################################################################################

{ pkgs ? import <nixpkgs> { }
, lib ? pkgs.lib
, python ? pkgs.python37
}:
with pkgs;
with python.pkgs;

# let pybind11 = callPackage ./pybind11.nix { inherit lib; };
#     #importlab = callPackage ../importlab { };
#     typed-ast = python.pkgs.typed-ast;
# in
callPackage ./default.nix { }
