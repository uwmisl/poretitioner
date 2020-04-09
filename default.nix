###########################################################################################
#
# default.nix
#
###########################################################################################
#
# This expression builds the Poretitioner application.
# It runs the test suite before building and will fail if any tests fail.
#
###########################################################################################

{ pkgs ? import <nixpkgs> { }, python ? pkgs.python37 }:
with pkgs;
let
  run_pkgs = (callPackage ./nix/dependencies.nix { inherit python; }).run;

  # To understand how `buildPythonApplication` works, check out https://github.com/NixOS/nixpkgs/blob/master/pkgs/development/interpreters/python/mk-python-derivation.nix
  poretitioner = python.pkgs.buildPythonApplication {
    pname = "poretitioner";
    version = "0.0.1";

    src = ./.;

    # Tests are done separately
    doCheck = false;

    # Run-time dependencies
    propagatedBuildInputs = run_pkgs;
  };
in poretitioner
