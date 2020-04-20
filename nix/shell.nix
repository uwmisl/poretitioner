###########################################################################################
#
# shell.nix
#
###########################################################################################
#
# This Nix expression provides a shell environment developers to test.
# This is most useful for ongoing development, testing new dependencies, etc.
#
# Running `nix-shell` on this file will pop you into a nice shell environment where all our
# packages are available for you.
#
###########################################################################################

{ pkgs ? import <nixpkgs> { config = import ./config.nix; }, cudaSupport ? true }:
with pkgs;
let
  python = callPackage ./python.nix { inherit pkgs cudaSupport; };
  dependencies = callPackage ./dependencies.nix { inherit python; };
in mkShell { buildInputs = dependencies.all; }
