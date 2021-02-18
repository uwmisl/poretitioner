NIXPKGS_ALLOW_UNFREE=1 # This lets us build non-Free software. https://nixos.org/manual/nixpkgs/stable/#sec-allow-unfree
#! /usr/bin/env nix-shell
#! nix-shell -i python -p "python37.withPackages (ps: [ ps.django ])"
#! nix-shell -p (import ./nix/dependencies.nix { inherit pkgs  python; } ).all

# nix-shell  -p '(let RUN_ME={ pkgs ? import <nixpkgs> { config = (import ./nix/config.nix); } , python ? (pkgs.callPackage ./nix/python.nix) { inherit pkgs; } }:  (import ./nix/dependencies.nix { inherit pkgs  python; } ).all; in RUN_ME {})'
#! nix-shell --pure -i bash

echo "poop"
echo "Python path: $PYTHONPATH"
echo $@
