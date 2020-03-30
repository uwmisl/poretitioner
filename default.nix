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

with import <nixpkgs> { };

let
  python = python37;
  run_pkgs = callPackage ./nix/runDependencies.nix { inherit python; };
  dev_pkgs = callPackage ./nix/devDependencies.nix { inherit python; };
  test_pkgs = callPackage ./nix/testingDependencies.nix { inherit python; };

  # To understand how `buildPythonApplication` works, check out https://github.com/NixOS/nixpkgs/blob/master/pkgs/development/interpreters/python/mk-python-derivation.nix
in python.pkgs.buildPythonApplication {
  pname = "poretitioner";
  version = "0.0.1";

  src = ./.;

  # Build-time exclusive dependencies
  nativeBuildInputs = dev_pkgs;

  # Test Dependencies
  doCheck = true;
  checkInputs = test_pkgs;
  checkPhase = ''
    py.test tests
  '';

  # Run-time dependencies
  buildInputs = run_pkgs;
  propagatedBuildInputs = run_pkgs;
}
