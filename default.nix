###########################################################################################
#
# default.nix
#
###########################################################################################
#
# This expression builds the Poretitioner application.
#   - To see how this application is built, see the `App` section.
#       Build with "nix-build -A app"
#       To build without testing (only recommended for local builds and rapid-prototyping)
#   - To see how this application is packaged for Docker, see the `Docker` section.
#
# Nix Resources:
#   - https://nix.dev/tutorials/
#
###########################################################################################

{ pkgs ? import <nixpkgs> { config = (import ./nix/config.nix); overlays = [ (import ./nix/overlays.nix) ]; }
, cudaSupport ? false
, python ? (pkgs.callPackage ./nix/python.nix) { inherit pkgs; }
}:
with pkgs;
let
  appInfo =
    builtins.fromJSON (builtins.readFile ./src/poretitioner/APPLICATION_INFO.json);
  name = appInfo.name;
  version = appInfo.version;

  ############################################################
  #
  # App - Builds the actual poretitioner application.
  #
  ############################################################
  dependencies =
    callPackage ./nix/dependencies.nix { inherit python cudaSupport; };
  tests = callPackage ./nix/test.nix {
    coverage = python.pkgs.coverage;
    pytest = python.pkgs.pytest;
  };
  run_pkgs = dependencies.run;
  all_pkgs = dependencies.all;
  test_pkgs = dependencies.test;
  run_tests_and_coverage = "echo Running tests:  ${tests.coverage};"
    + tests.coverage;

  shell = callPackage ./nix/shell.nix {
    inherit python cudaSupport;
    postShellHook = ''
    '';
  };


  src = ./.;

  # How to develop/release python packages with Nix:
  # https://github.com/NixOS/nixpkgs/blob/master/doc/languages-frameworks/python.section.md
  #
  # doCheck - Whether to run the test suite as part of the build, defaults to true.
  # To understand how `buildPythonPackage` works, check out https://github.com/NixOS/nixpkgs/blob/master/pkgs/development/interpreters/python/mk-python-derivation.nix
  poretitioner = { doCheck ? true }:
    python.pkgs.buildPythonPackage {
      pname = name;
      version = version;
      format = "pyproject";
      src = src;
      checkInputs = test_pkgs;
      inherit doCheck;
      checkPhase = run_tests_and_coverage;

      # Run-time dependencies
      propagatedBuildInputs = run_pkgs;
    };

  app = { doCheck ? true }: python.pkgs.toPythonApplication (poretitioner { inherit doCheck; });

  ####################################################################
  #
  # Docker - Builds the Docker image for the poretitioner application.
  #
  ####################################################################

  supportsDocker = pkgs.stdenv.isLinux;
  # Currently can't build docker images on Mac OS (Darwin): https://github.com/NixOS/nixpkgs/blob/f5a90a7aab126857e9cac4f048930ddabc720c55/pkgs/build-support/docker/default.nix#L620
  dockerImage = { app }: dockerTools.buildImage {
    name = "${name}";
    tag = "latest";

    # Setting 'created' to 'now' will correctly set the file's creation date
    # (instead of setting it to Unix epoch + 1). This is impure, but fine for our purposes.
    created = "now";
    config = {
      Cmd = [  ];
      # Runs 'poretitioner' by default.
      Entrypoint = [ "${app.outPath}/bin/${app.pname}" ];
    };
  };

in
{
  app-no-test = app { doCheck = false; };
  test = poretitioner { doCheck = true; };
  app = app { doCheck = true; };
  lib = poretitioner { doCheck = false; };

  # Note: Shell can only be run by using "nix-shell" (i.e. "nix-shell -A shell ./default.nix").
  # Here's an awesome, easy-to-read overview of nix shells: https://ghedam.at/15978/an-introduction-to-nix-shell
  shell = shell;
}
//
# Docker support
pkgs.lib.optionalAttrs (supportsDocker) {
  docker = dockerImage { app = (app { doCheck = false; }); };
}
