{ stdenv, fetchFromGitHub, networkx, six }:

pythonPackages.buildPythonPackage rec {
  name = "importlab";

  src = fetchFromGitHub {
    owner = "google";
    repo = "importlab";
    rev = "171d0b5687dd6a9d2c5b01b4b2c3ecce2d79dddb";
    sha256 = "1qlz2b7271na0gpr6qgr43rqrmgg2g9gf1whbsg29d346578bbgv";
  };

  propagatedBuildInputs = [ networkx six ];

  checkPhase = "true";
}

# { lib
# , python
# , buildPythonPackage
# , fetchFromGitHub
# , cmake
# , networkx
# }:

# buildPythonPackage rec {
#   pname = "importlab";
#   version = "0.6.1";

#   disabled = python.pythonOlder "3.6.0"; # requires python version >=3.6.0

#   src = fetchFromGitHub {
#     owner = "google";
#     repo = "importlab";
#     rev = "0.1.1";
#     sha256 = "0byfs49qakvwyvllk8v77bi2if9vfg9mg1gdfhp98r580zsiiyy2";
#   };


#   doCheck = false;
  
#   nativeBuildInputs = [ cmake ];
#   propagatedBuildInputs = [ networkx ];

#   meta = with lib; {
#     description = "A library to calculate python dependency graphs";
#     homepage = https://github.com/google/importlab;
#     license = licenses.asl20;
#     # maintainers = [ maintainers. ];
#   };
# }