{ lib, buildPythonPackage, fetchFromGitHub, setuptools, flex, cmake, attrs, ninja, bison, pybind11, pylint, pyyaml, six, toml, typed-ast}:
let
  pname = "pytype";
  owner = "google";
  repo = "pytype";
  revision_tag = "2021.02.19";
  version = revision_tag;

in buildPythonPackage rec {

  inherit pname version;

  src = fetchFromGitHub {
    inherit owner repo;
    rev = revision_tag;
    sha256 = "0srcyc8mmc8q2w2p103gpifphavwkbmk7siz01z49qaqbp0bswhf";
  };

  doCheck = true;

  patches = [ ./dep.patch ];
  nativeBuildInputs = [ bison flex cmake ninja setuptools ];
  #nativeBuildInputs = [ pybind11 flex bison cmake];

  propagatedBuildInputs = [ pybind11 pylint pyyaml six toml typed-ast];

  meta = with lib; {
    description =
      "Pytype checks and infers types for your Python code - without requiring type annotations.";
    homepage = "https://github.com/google/pytype";
  };
}
