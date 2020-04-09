{ pkgs ? import <nixpkgs> {}, python ? pkgs.python37 }:

with pkgs;
let poretitioner = callPackage ../default.nix { inherit pkgs python; };
    binPath = builtins.concatStringsSep ''/'' [ poretitioner.outPath ''bin'' poretitioner.pname ];
in
dockerTools.buildImage {
  name = "poretitioner";

  config = {
    Cmd = [ ''${binPath}'' ];
  };
}
