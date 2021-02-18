{ lib, buildPythonPackage, fetchFromGitHub, h5py, numpy, six, packaging, progressbar33 }:
let
  repo_name = "ont_fast5_api";
  pname = lib.strings.replaceChars [ "_" ] [ "-" ] repo_name;
  revision_tag = "release_3.2.0";
  version = revision_tag;
in buildPythonPackage rec {

  inherit pname version;

  src = fetchFromGitHub {
    owner = "nanoporetech";
    repo = repo_name;
    rev = revision_tag;
    sha256 = "1z7ki16c9m9z24w5bwh978c05g1l6lh9n3vgb0mrjq7n5san90j6";
  };

  doCheck = true;

  propagatedBuildInputs = [ h5py numpy six packaging progressbar33 ];

  meta = with lib; {
    description =
      "A simple interface to HDF5 files of the Oxford Nanopore .fast5 file format.";
    homepage = "https://github.com/nanoporetech/ont_fast5_api";
    license = licenses.mpl20;
  };
}
