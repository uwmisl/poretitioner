{ lib, fetchFromGitHub, buildPythonPackage, h5py, numpy, six, progressbar33, pyyaml, packaging }:
let
  api_name = "ont_fast5_api";
  api_version = "release_3.3.0";
  
  ont_fast5_api = buildPythonPackage {
    pname = api_name;
    version = api_version;

    src = fetchFromGitHub {
      owner = "nanoporetech";
      repo = api_name;
      rev = api_version;
      sha256 = "1hlxz6ac1jr5pincq206ssfz7z3zwj8a4iz8c38id1i2s2rx5idd";
    };

    propagatedBuildInputs = [
      h5py
      numpy
      six
      progressbar33
      packaging
    ];

    doCheck = false;

    meta = with lib; {
      description =
        "A simple interface to HDF5 files of the Oxford Nanopore .fast5 file format.";
      homepage = "https://github.com/nanoporetech/ont_fast5_api/";
      license = licenses.mpl20;
    };
  };
in 
ont_fast5_api

#   validator_name = "ont_h5_validator";
#   validator_version = "release_2.0.1";
#   ont_h5_validator = buildPythonPackage {
#     pname = validator_name;
#     version = validator_version;

#     src = fetchFromGitHub {
#       owner = "nanoporetech";
#       repo = validator_name;
#       rev = validator_version;
#       sha256 = "0rp3yrvlf9bgij4g967hh7lqx0kkil4n0l657v60d4k8kkdzxifg";
#     };

#     doCheck = false;

#     propagatedBuildInputs = [
#       h5py
#       numpy
#       six
#       pyyaml
#     ];

#     meta = with lib; {
#       description =
#         "A simple tool for validating the structure of fast5 files against the file schema maintained by Oxford Nanopore Technologies.";
#       homepage = "https://github.com/nanoporetech/ont_fast5_api/";
#       license = licenses.mpl20;
#     };
#   };
  
# in [
#   ont_fast5_api
#   ont_h5_validator
# ]
