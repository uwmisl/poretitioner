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


with import <nixpkgs> {};

let get_run_pkgs = (import ./runDependencies.nix);
    get_dev_pkgs = (import ./devDependencies.nix);
    get_test_pkgs = (import ./testingDependencies.nix);
    pythonEnv = python37.withPackages (python37Packages:
        get_run_pkgs python37Packages
     ++ get_dev_pkgs python37Packages
     ++ get_test_pkgs python37Packages
    ); 

in mkShell {
    buildInputs = [
        pythonEnv
    ];
}

