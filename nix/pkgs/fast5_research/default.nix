{ lib, fetchFromGitHub, buildPythonPackage, toPythonApplication, h5py, numpy, futures, progressbar2, pysam }:
let
  name = "fast5_research";
  pname = name;
  version = "v1.2.22";
  fast5_research = buildPythonPackage {

    inherit pname version;

    src = fetchFromGitHub {
      owner = "nanoporetech";
      repo = name;
      rev = version;
      sha256 = "15zw1xaa0dhh8b3qkmsr2kwcmagyd44ddq07j7mc9qwpdck6r2zr";
    };

    doCheck = false;

    propagatedBuildInputs = [
      h5py
      numpy
      futures
      progressbar2
      pysam
    ];

    meta = with lib; {
      description =
        "Python fast5 reading and writing functionality provided by ONT Research.";
      homepage = "https://github.com/nanoporetech/fast5_research/";
      license = licenses.mpl20;
    };
  };

in
fast5_research