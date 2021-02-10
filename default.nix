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
###########################################################################################

{ pkgs ? import <nixpkgs> { config = (import ./nix/config.nix); }
, cudaSupport ? false
, python ? (pkgs.callPackage ./nix/python.nix) { inherit pkgs; }
}:
with pkgs;
let
  appInfo = builtins.fromJSON (builtins.readFile ./poretitioner/APPLICATION_INFO.json);
  name = appInfo.name;
  version = appInfo.version;

  ############################################################
  #
  # App - Builds the actual poretitioner application.
  #
  ############################################################
  dependencies = callPackage ./nix/dependencies.nix { inherit python cudaSupport; };
  run_pkgs = dependencies.run;
  all_pkgs = dependencies.all;
  test_pkgs = dependencies.test;
  run_tests = "coverage run -m pytest -c ./pytest.ini";
  src = ./.;

  # How to develop/release python packages with Nix:
  # https://github.com/NixOS/nixpkgs/blob/master/doc/languages-frameworks/python.section.md
  #
  # doCheck - Whether to run the test suite as part of the build, defaults to true.
  # To understand how `buildPythonPackage` works, check out https://github.com/NixOS/nixpkgs/blob/master/pkgs/development/interpreters/python/mk-python-derivation.nix
  poretitioner = {doCheck ? true } : python.pkgs.buildPythonPackage {
    pname = name;
    version = version;

    src = src;
    checkInputs = test_pkgs;
    inherit doCheck;
    checkPhase = run_tests;

    # Run-time dependencies
    propagatedBuildInputs = run_pkgs ++ [ src ];
  };

  app = {doCheck ? true }: python.pkgs.toPythonApplication (poretitioner { inherit doCheck; });

  ####################################################################
  #
  # Docker - Builds the Docker image for the poretitioner application.
  #
  ####################################################################

  binPath = builtins.concatStringsSep "/" [
    poretitioner.outPath
    "bin"
    poretitioner.pname
  ];

  # Currently can't build docker images on Mac OS (Darwin): https://github.com/NixOS/nixpkgs/blob/f5a90a7aab126857e9cac4f048930ddabc720c55/pkgs/build-support/docker/default.nix#L620
  dockerImage = lib.optionals stdenv.isLinux (dockerTools.buildImage {
    name = "${name}_v${version}";
    tag = "latest";

    # Setting 'created' to 'now' will correctly set the file's creation date
    # (instead of setting it to Unix epoch + 1). This is impure, but fine for our purposes.
    created = "now";
    config = {
      # Runs 'poretitioner' by default.
      Cmd = [ "${binPath}" ];
    };
  });

in {
  app-no-test = app { doCheck = false; };
  test = poretitioner { doCheck = true; };
  app = app { doCheck = true; };
  lib = poretitioner { doCheck = true; };
  docker = dockerImage;
  shell = mkShell { buildInputs = [ (poretitioner  { doCheck = false; }) ] ++ all_pkgs ; };
}
