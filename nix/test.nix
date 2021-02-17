# ##########################################################################################
#
# test.nix
#
###########################################################################################
#
# This Nix expression holds expressions for testing.
# This is most useful for ongoing development, testing new dependencies, etc.
#
# Running `nix-shell` on this file will pop you into a nice shell environment where all our
# packages are available for you.
#
###########################################################################################

{ coverage
, pytest
}:
let
  # Make sure 'pytest' and 'coverage' python dependencies are installed.
  #run_tests = "coverage run -m pytest -c ./pytest.ini";
  coverage_command = "${coverage.pname}";
  pytest_command = "${pytest.pname}";
  pytest_config_filepath = "./pytest.ini";
  run_pytest_command = "${pytest_command} -c ${pytest_config_filepath}";
  # Where to store the output of the coverage.
  coverage_directory = "tests/coverage/";
  generate_html_report = "${coverage_command} html --directory=${coverage_directory}";
in
{
    # Command to run tests and generate coverage
    #coverage = "${coverage_command} run --source='./poretitioner/' ${run_pytest_command} ; ${generate_html_report}";
    coverage = "${coverage_command} run --source='./poretitioner/' -m ${run_pytest_command}; ${generate_html_report}";
    tests = run_pytest_command;
}
