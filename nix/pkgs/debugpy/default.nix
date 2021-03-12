{ lib, buildPythonApplication, fetchFromGitHub }:
let
  repo_name = "debugpy";
  pname = repo_name;
  revision_tag = "v1.2.1";
  version = revision_tag;
in buildPythonApplication rec {

  inherit pname version;

  src = fetchFromGitHub {
    owner = "microsoft";
    repo = repo_name;
    rev = revision_tag;
    sha256 = "1dgjbbhy228w2zbfq5pf0hkai7742zw8mmybnzjdc9l6pw7360rq";
  };

  doCheck = false;

  meta = with lib; {
    description =
      "A Visual Studio code debugger for Python.";
    homepage = "https://github.com/microsoft/debugpy";
    license = licenses.mit;
  };
}
