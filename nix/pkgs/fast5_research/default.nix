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
      sha256 = "0rp3yrvlf9bgij4g967hh7lqx0kkil4n0l657v60d4k8kkdzxifg";
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
