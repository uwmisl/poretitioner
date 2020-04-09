{ pkgs ? import <nixpkgs> {}, python ? pkgs.python37 }:

with pkgs;
let poretitioner = callPackage ../default.nix { inherit pkgs python; };
    binPath = builtins.concatStringsSep ''/'' [poretitioner.outPath python.sitePackages ''bin'' poretitioner.pname ];
in
dockerTools.buildImage {
  name = "poretitioner";
  runAsRoot = ''
    echo ${binPath}
  '';

  config = {
    Cmd = [ ''${binPath}'' ];
  };
}
