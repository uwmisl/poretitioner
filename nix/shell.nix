# ##########################################################################################
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

with import <nixpkgs> { };

let
  python = pkgs.python37;
  run_pkgs = callPackage ./runDependencies.nix { inherit python; };
  dev_pkgs = callPackage ./devDependencies.nix { inherit python; };
  test_pkgs = callPackage ./testingDependencies.nix { inherit python; };
in mkShell { buildInputs = run_pkgs ++ dev_pkgs ++ test_pkgs; }
