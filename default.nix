###########################################################################################
#
# default.nix
#
###########################################################################################
#
# This expression builds the Poretitioner application.
#   - To see how this application is built, see the `App` section.
#   - To see how this application is packaged for Docker, see the `Docker` section.
#
###########################################################################################

{ pkgs ? import <nixpkgs> { }, python ? pkgs.python37 }:
with pkgs;
let
  name = "poretitioner";

  ###########################################################################################
  #
  # App - Builds the actual poretitioner application.
  #
  ###########################################################################################
  dependencies = (callPackage ./nix/dependencies.nix { inherit python; });
  run_pkgs = dependencies.run;
  test_pkgs = dependencies.test;

  # To understand how `buildPythonApplication` works, check out https://github.com/NixOS/nixpkgs/blob/master/pkgs/development/interpreters/python/mk-python-derivation.nix
  poretitioner = python.pkgs.buildPythonApplication {
    pname = name;
    version = "0.0.1";

    src = ./.;

    checkInputs = test_pkgs;
    doCheck = true;
    checkPhase = ''
      pytest tests
    '';


    # Run-time dependencies
    propagatedBuildInputs = run_pkgs;
  };

  ###########################################################################################
  #
  # Docker - Builds a Docker image for the poretitioner application.
  #
  ###########################################################################################

  # The docker image we distribute.
  binPath = builtins.concatStringsSep ''/'' [ poretitioner.outPath ''bin'' poretitioner.pname ];
  dockerImage = dockerTools.buildImage {
    name = name;

    config = {
      # Runs 'poretitioner' by default.
      Cmd = [ ''${binPath}'' ];
    };
  };

in {
  app = poretitioner;
  docker = dockerImage;
}
