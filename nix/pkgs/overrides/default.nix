{ lib, buildPythonPackage, fetchFromGitHub }:
let
  repo_name = "overrides";
  pname = repo_name;
  revision_tag = "3.1.0";
  version = revision_tag;
in
buildPythonPackage rec {

  inherit pname version;

  src = fetchFromGitHub {
    owner = "mkorpela";
    repo = repo_name;
    rev = revision_tag;
    sha256 = "1x19r0983v3jj0ca9r0yy1cxn4zh59c978nycrab5n9akwikbnpv";
  };

  doCheck = false;

  meta = with lib; {
    description =
      "A decorator to automatically detect mismatch when overriding a method.";
    homepage = "https://github.com/mkorpela/overrides";
    license = licenses.asl20;
  };
}
