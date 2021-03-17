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
, python ? pkgs.callPackage ./python.nix { inherit pkgs; }
, cudaSupport ? false
, postShellHook ? ""
}:
with pkgs;
let
  dependencies = callPackage ./dependencies.nix { inherit python cudaSupport; };

  poretitionerPath="../src/poretitioner";

  pythonEnv = python.buildEnv.override {
    extraLibs=dependencies.pythonDeps.all
     ++ [ poretitionerPath ];
  };


in mkShell {
  shellHook = ''
    PYTHONPATH="${poretitionerPath}:$PYTHONPATH";
  ''
  + postShellHook
  ;

  buildInputs = [
    pkgs.bash
    pkgs.bashInteractive
    pkgs.locale
    pkgs.xtermcontrol
    pkgs.xterm
    pkgs.zsh
    pythonEnv
    pythonEnv.pkgs.bpython
  ]
  ++  dependencies.all ;


  propagatedBuildInputs = dependencies.all ++ [ pythonEnv ];
}
