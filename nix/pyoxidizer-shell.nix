{ pkgs ? import <nixpkgs> {} }:

let
  # picked a conservative python version because of errors regarding `cf_feature_version`
  python = pkgs.python37;
  run_dependencies = import ./runDependencies.nix { inherit python; };
  python_deps = with pkgs; [openssl pkgconfig python] ++ run_dependencies ;

  pyoxidizer = pkgs.rustPlatform.buildRustPackage rec {
    pname = "pyoxidizer";
    version = "0.6.0";
    src = pkgs.fetchFromGitHub {
      owner = "indygreg";
      repo = "PyOxidizer";
      rev = "v0.6.0";
      sha256 = "1n4976wdcdpn4hwm8n7rwwx3pcvlkksad3y44n3q5mmh2lbr21g0";
    };
    cargoSha256 = "0fx0h9wlqbwlfmgz2kq6s3jhlyv32xxrhnmhipsmqb6vfpzm2pbq";
    nativeBuildInputs = python_deps;
    doCheck = false;
  };

  # fhs = pkgs.buildFHSUserEnv {
  #   name = "python-env";

  #   targetPkgs = pkgs: with pkgs; [
  #     sudo
  #     wget
  #     git
  #     curl
  #     less
  #     lesspipe
  #     emacs
  #     which
  #     cargo
  #     rustc
  #     pyoxidizer
  #   ] ++ python_deps;
  #   runScript = "bash";
  #   profile = ''
  #     export CARGO_TARGET_DIR=target
  #   '';
  # };

in pkgs.mkShell {
  name = "python-env-shell";

  buildInputs = with pkgs; [ cargo git rustc pyoxidizer ];
  # shellHook = ''
  #   exec env SSL_CERT_FILE=$SSL_CERT_FILE python-env
  # '';
}
