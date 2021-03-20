###########################################################################################
#
# playground.nix
#
###########################################################################################
#
# This Nix expression provides a python environemnt shell environment developers to test.
# It's great for when you want to test some new approaches quickly
# This is most useful for ongoing development, testing new dependencies, etc.
#
# Running `nix-shell` on this file will pop you into a nice shell environment where all our
# packages are available for you.
#
###########################################################################################

{ pkgs ? import <nixpkgs> { config = (import ./nix/config.nix); overlays = [ (import ./nix/overlays.nix) ];
  config = import ./config.nix;
  }
, python ? pkgs.callPackage ./python.nix { inherit pkgs; }
, cudaSupport ? false
}:
with pkgs;
with python.pkgs;
let
  # This file contains any python code you'd like
  # For convenience, we load some of the application's most important
  # classes and constants at the top, so you can jump right in and
  # use them in in bpython immediately.
  playground_py = ''./nix/playground.py'';
  dependencies = callPackage ./dependencies.nix { inherit python cudaSupport; };
  shell = callPackage ./shell.nix {
    inherit python cudaSupport;
    postShellHook = ''
      ${python.pkgs.bpython.pname} --interactive ${playground_py}
    '';
  };
in
shell
#{ pkgs ? import <nixpkgs> { config = (import ./nix/config.nix); } , python ? (pkgs.callPackage ./nix/python.nix) { inherit pkgs; } }:
