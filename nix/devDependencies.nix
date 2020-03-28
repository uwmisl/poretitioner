###########################################################################################
#                       
# devDependencies.nix 
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


pythonPackages: 
    let precommit = import ./pre-commit.nix;
        test_pkgs = (import ./testingDependencies.nix) pythonPackages;
    in [
        #####################################
        #      Development Automation       #
        #####################################
        # Git hooks
        precommit  

        #####################################
        #          Code Formatting          #
        #####################################

        # Import sorter 
        pythonPackages.isort
        # Highly opinionated code-formatter
        pythonPackages.black
        # Style-guide enforcer 
        pythonPackages.flake8
        # Docstring static analyzer
        pythonPackages.pydocstyle
        # Static type analyzer
        #pythonPackages.pytype
    ] 
    ++ test_pkgs
