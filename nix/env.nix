###########################################################################################
#
# env.nix
#
###########################################################################################
#
# A collection of tools for developers to install locally.
# This configures things like linters, pre-commit git hooks.
#
# Install these by running `nix-env -i -f <<path to this file>>`
# replacing '<<path to this file>>' with the absolute path to this `env.nix` file.
#
###########################################################################################
{ pkgs ? import <nixpkgs> { config = import ./config.nix; }, cudaSupport ? false }:
with pkgs;
let
  python = callPackage ./python.nix { inherit pkgs cudaSupport; };
  dependencies = callPackage ./dependencies.nix { inherit python; };
  dev_pkgs = dependencies.build;
  testing_pkgs = dependencies.test;
in [ python ] ++ dev_pkgs ++ testing_pkgs
