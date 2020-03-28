###########################################################################################
#                       
# env.nix 
#                      
###########################################################################################
#
# A collection of tools for developers to install locally.
# This configures things like linters, pre-commit git hooks
#
# Install these by running `nix-env -i -f <<path to this file>>`
# replacing '<<path to this file>>' with the absolute path to this `env.nix` file.
#
###########################################################################################


let
    pkgs = import <nixpkgs> {};
    python = pkgs.python37;

    dev_pkgs = (import ./devDependencies.nix) { inherit python; };
    testing_pkgs = (import ./testingDependencies.nix) { inherit python; }; 
in
dev_pkgs ++ testing_pkgs