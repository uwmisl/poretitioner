# ##########################################################################################
#                       
# pre-commit.nix 
#                      
###########################################################################################
#
# The `pre-commit` third-party package helps us automate several development processes [1].
#
# Why are we building in manually instead of downloading it from the Nix store?  
# Sadly, >=v2.0 version of pre-commit isn't available in the Nix package repository yet, 
# (Nix-pkgs only has v1.2.1, which is coupled to Python 2), so we're building it ourselves 
# for now.
#
# [1] - https://github.com/pre-commit/pre-commit 
#
###########################################################################################

let
  pkgs = import <nixpkgs> { };
  defaultPython = pkgs.python37;
in { python ? defaultPython }: with python.pkgs; callPackage ./default.nix { }
