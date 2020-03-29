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

{ python }:
with python.pkgs; [
  # Testing suite
  pytest
  # Test runner 
  pytestrunner
  # Test code coverage generator
  pytestcov
]
