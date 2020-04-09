{ pkgs ? import <nixpkgs> {} }:

with pkgs;
let poretitioner = callPackage ../default.nix { inherit pkgs; };
    binPath = builtins.concatStringsSep ''/'' [pore.outPath python.sitePackages ''bin'' poretitioner.pname ];
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
