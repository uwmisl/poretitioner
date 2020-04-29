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

  ############################################################
  #
  # App - Builds the actual poretitioner application.
  #
  ############################################################
  dependencies = callPackage ./nix/dependencies.nix { inherit python cudaSupport; };
  run_pkgs = dependencies.run;
  test_pkgs = dependencies.test;

  # doCheck - Whether to run the test suite as part of the build, defaults to true.
  # To understand how `buildPythonApplication` works, check out https://github.com/NixOS/nixpkgs/blob/master/pkgs/development/interpreters/python/mk-python-derivation.nix
  poretitioner = {doCheck ? true } : python.pkgs.buildPythonApplication {
    pname = name;
    version = appInfo.version;

    src = ./.;

    checkInputs = test_pkgs;
    inherit doCheck;
    checkPhase = "pytest tests";

    # Run-time dependencies
    propagatedBuildInputs = run_pkgs;
  };

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

  dockerImage = dockerTools.buildImage {
    name = name;
    tag = "latest";

    # Setting 'created' to 'now' will correctly set the file's creation date
    # (instead of setting it to Unix epoch + 1). This is impure, but fine for our purposes.
    created = "now";
    config = {
      # Runs 'poretitioner' by default.
      Cmd = [ "${binPath}" ];
    };
  };

in {
  app-no-test = poretitioner { doCheck = false; };
  app = poretitioner;
  docker = dockerImage;
}
