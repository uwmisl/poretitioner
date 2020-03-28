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
    inherit (pkgs) buildEnv;
    python37pkgs = pkgs.python37.pkgs;

    get_dev_pkgs = (import ./devDependencies.nix);
    get_testing_pkgs = (import ./testingDependencies.nix);
    dev_pkgs = get_dev_pkgs python37pkgs
                 ++ [python37pkgs.virtualenv]; # VirtualEnv is needed for pre-commit hooks
in
dev_pkgs
