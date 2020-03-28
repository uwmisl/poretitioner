###########################################################################################
#                       
# default.nix 
#                      
###########################################################################################
#
# This expression hosts dependencies that shouldn't be packaged for distribution, but 
# are still needed for developers. This includes testing frameworks, as well as tools like
# linters, git hooks, and static analyzers.
#
# It takes in `pythonPackages`, which is intended to be provided by `python.withPackages`.
#
###########################################################################################

let nixpkgs = import <nixpkgs> {};
    packages = nixpkgs.python37.pkgs;

    run_pkgs = (import ./nix/runDependencies.nix) packages;
    dev_pkgs = (import ./nix/devDependencies.nix) packages;
    test_pkgs = (import ./nix/testingDependencies.nix) packages;

    # Reference https://github.com/NixOS/nixpkgs/blob/master/pkgs/development/interpreters/python/mk-python-derivation.nix
    # Base python package definition, which we'll then 
    poretitioner = packages.buildPythonApplication {
        pname = "poretitioner";
        version = "0.0.1"; 

        src = ./.;

        doCheck = true;

        # Build-time exclusive dependencies
        nativeBuildInputs = dev_pkgs;
    
        # Test Dependencies 
        checkInputs = test_pkgs;
        checkPhase = ''
            py.test tests
        '';

        # Run-time dependencies
        buildInputs = run_pkgs;
        propagatedBuildInputs = run_pkgs;
    };

in poretitioner