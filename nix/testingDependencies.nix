###########################################################################################
#                       
# testingDependencies.nix
#                      
###########################################################################################
#
# This expression hosts the project's test-only dependencies (e.g. test runners). 
# It should not include any packages that aren't part of the testing infrastructure.
#
# It takes in a `pythonPackage`, which is intended to be provided by `python.withPackages`.
#
###########################################################################################


pythonPackages: [
        # Testing suite
        pythonPackages.pytest
        # Test runner 
        pythonPackages.pytestrunner
        # Test code coverage generator
        pythonPackages.pytestcov
]
