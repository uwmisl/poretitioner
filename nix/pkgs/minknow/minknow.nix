let
  pkgs = import <nixpkgs> { };
  stdenv = pkgs.stdenv;
  defaultPython = pkgs.python37;
in { python ? defaultPython }: with python.pkgs; callPackage ./default.nix { inherit python; }
