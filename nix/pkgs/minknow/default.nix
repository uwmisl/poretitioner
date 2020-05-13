{ lib
, python
, fetchFromGitHub
, buildPythonPackage
, stdenv
}:
let basename = "minknow";
    buildInputs = with python.pkgs; [ protobuf grpcio grpcio-tools ];

minknow = buildPythonPackage rec {
  name = "${basename}";
  version = "v3.6.0";

  src = fetchFromGitHub {
    owner  = "nanoporetech";
    repo   = "minknow_api";
    rev    = "${version}";
    sha256 = "12fbp8qjjbl9azl5cfq6aagx5298b0lq8lq0bi26rkdpa498j66q";
  };

  format = "other";

  nativeBuildInputs = buildInputs;
  propagatedBuildInputs = buildInputs;

  buildPhase = ''
    mkdir -p $TMPDIR/minknow/generated/
    ${python}/bin/python -m grpc_tools.protoc \
      --python_out=$TMPDIR/minknow/generated/ \
      --grpc_python_out=$TMPDIR/minknow/generated/ \
      -I $src/. $src/minknow/rpc/*.proto
  '';

  installPhase = ''
    mkdir -p "$out/${python.sitePackages}"
    cp -r $TMPDIR/minknow/generated/** "$out/${python.sitePackages}/"
    export PYTHONPATH="$out/${python.sitePackages}:$PYTHONPATH"
  '';

};

in minknow
