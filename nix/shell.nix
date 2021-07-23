###########################################################################################
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

{ pkgs ? import <nixpkgs> { config = (import ./config.nix); overlays = [ (import ./overlays.nix) ]; }
, python ? pkgs.callPackage ./python.nix { inherit pkgs; }
, cudaSupport ? false
, postShellHook ? ""
}:
with pkgs;
let
  dependencies = callPackage ./dependencies.nix { inherit python cudaSupport; };

  #poretitionerPath = ../src/poretitioner;

  #myPython = python.withPackages (_: dependencies.pythonDeps.all ++ [ poretitionerPath ]);

in
mkShell {
  shellHook = ''
    PYTHONPATH="${poretitionerPath}:$PYTHONPATH";
    # Patch `packages.json` so that nix's *python* is used as default value for `python.pythonPath`.
    if [ -e "../.vscode/settings.json" ]; then
      substituteInPlace ../.vscode/settings.json \
        --replace \"python\"  \"${myPython}/bin/python\"
    fi
  ''
  # --replace "\"python.pythonPath\": .*" "\"python.pythonPath\": \"${myPython}/bin/python\""
  + postShellHook
  ;

  buildInputs = [
    pkgs.bash
    pkgs.bashInteractive
    pkgs.locale
    pkgs.xtermcontrol
    pkgs.xterm
    pkgs.zsh
    python
  ]
  ++ dependencies.all;

  propagatedBuildInputs = dependencies.all ++ [ python ];
}
