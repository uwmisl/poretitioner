{ lib
, ninja
, cmake
, python
, wheel
, setuptools
, buildPythonPackage
, fetchFromGitHub
}:

buildPythonPackage rec {
  pname = "pybind11";
  version = "v2.6.2";

  #format = "other";

  disabled = python.pythonOlder "3.5"; # requires python version !=3.0,!=3.1,!=3.2,!=3.3,!=3.4,>=2.7

  src = fetchFromGitHub {
    owner = "pybind";
    repo = "pybind11";
    rev = "v2.6.2";
    sha256 = "1lsacpawl2gb5qlh0cawj9swsyfbwhzhwiv6553a7lsigdbadqpy";
  };

  nativeBuildInputs = [ ninja cmake ];

  propagatedBuildInputs = [ setuptools wheel ];

  doCheck = true;

  # # Extra packages (may not be necessary)
  # pybind11_global==2.6.2 # global
  meta = with lib; {
    description = "Seamless operability between C++11 and Python";
    homepage = https://github.com/pybind/pybind11;
    license = licenses.bsd3;
  };
}