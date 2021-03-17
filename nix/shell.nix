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

{ pkgs ? import <nixpkgs> { config = import ./config.nix; }
, python ? pkgs.callPackage ./python.nix { inherit pkgs; }, cudaSupport ? false
}:
with pkgs;
let
  dependencies = callPackage ./dependencies.nix { inherit python cudaSupport; };

  pythonEnv = python.withPackages (ps: (dependencies.getDependenciesForPython python).all );
in mkShell { 
  buildInputs = [
    pkgs.zsh
    pythonEnv
  ]
  ++  dependencies.all ;

  shellHook = ''
    exec env zsh
  '';

  MY_ENVIRONMENT_VARIABLE = "world";
  propagatedBuildInputs = dependencies.all ++ [ pythonEnv ]; 
}
