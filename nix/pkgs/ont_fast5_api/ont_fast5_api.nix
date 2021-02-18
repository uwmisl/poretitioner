# ##########################################################################################
#
# ont_fast5_api.nix
#
###########################################################################################
#
# The `ont-fast5-api` provides some APIs for handling Fast5 files.
#
# Why are we building it in manually instead of downloading it from the Nix store?
# Sadly, it's not in the Nix store yet.
#
###########################################################################################

{ pkgs ? import <nixpkgs> { }, lib ? pkgs.lib
, python ?  pkgs.python37

}:
with python.pkgs;
callPackage ./default.nix { }
