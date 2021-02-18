{ lib, python, buildPythonApplication, coverage }:
buildPythonApplication rec {

  pname = "coverage-badge";
  version = "v1.0.1";
  src = fetchGit {
    url = "https://github.com/dbrgn/coverage-badge.git";
    rev = "59098c030555b45f8498956fb39c79bc82afae8d";
  };

  propagatedBuildInputs = [ coverage ];

  # Found here: https://github.com/pre-commit/pre-commit/blob/master/setup.cfg
#   propagatedBuildInputs =
#     [ cfgv identify pyyaml toml nodeenv virtualenv importlib-metadata ]
#     ++ lib.optional (pythonOlder "3.7") importlib-resources
#     ++ lib.optional (pythonOlder "3.2") futures;

#   # slow and impure
#   doCheck = false;

  # hook-tmpl uses a hard-coded python shebang, so we need to replace it.
#   preFixup = ''
#     substituteInPlace $out/${python.sitePackages}/pre_commit/resources/hook-tmpl \
#            --subst-var-by pre-commit $out
#        '';

  meta = with lib; {
    description =
      "A small script to generate coverage badges using Coverage.py.";
    homepage = "https://github.com/dbrgn/coverage-badge";
    license = licenses.mit;
  };
}
