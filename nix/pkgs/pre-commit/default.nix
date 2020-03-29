###########################################################################################
#
# default.nix
#
###########################################################################################
#
# The `pre-commit` third-party package helps us automate several development processes [1].
#
# Why are we building in manually instead of downloading it from the Nix store?
# Sadly, >=v2.0 version of pre-commit isn't available in the Nix package repository yet,
# (Nix-pkgs only has v1.2.1, which is coupled to Python 2), so we're building it ourselves
# for now.
#
# [1] - https://github.com/pre-commit/pre-commit
#
###########################################################################################

{ lib, python, buildPythonApplication, pythonOlder, cfgv, identify, pyyaml, toml
, nodeenv, virtualenv, futures, importlib-metadata, importlib-resources }:
buildPythonApplication rec {

  pname = "pre-commit";
  version = "2.2.0";
  src = fetchGit {
    url = "https://github.com/pre-commit/pre-commit.git";
    rev = "30d3bb29900cf7caa6624edbaf3faf37a11b07f3";
  };

  patches = [ ./hook-tmpl-use-the-hardcoded-path-to-pre-commit.patch ];

  # Found here: https://github.com/pre-commit/pre-commit/blob/master/setup.cfg
  propagatedBuildInputs =
    [ cfgv identify pyyaml toml nodeenv virtualenv importlib-metadata ]
    ++ lib.optional (pythonOlder "3.7") importlib-resources
    ++ lib.optional (pythonOlder "3.2") futures;

  # slow and impure
  doCheck = false;

  # hook-tmpl uses a hard-coded python shebang, so we need to replace it.
  preFixup = ''
    substituteInPlace $out/${python.sitePackages}/pre_commit/resources/hook-tmpl \
           --subst-var-by pre-commit $out
       '';

  meta = with lib; {
    description =
      "A framework for managing and maintaining multi-language pre-commit hooks";
    homepage = "https://pre-commit.com/";
    license = licenses.mit;
    maintainers = [ maintainers.borisbabic ];
  };
}
