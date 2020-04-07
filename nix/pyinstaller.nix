{ pkgs ? import <nixpkgs> {} }:

let
  # picked a conservative python version because of errors regarding `cf_feature_version`
  python = pkgs.python37;
  run_dependencies = import ./runDependencies.nix { inherit python; };

  pypkgs = python.pkgs;
  dask = pypkgs.dask;
  distributed = pypkgs.distributed;
  daskPath = dask.outPath  + "/" +  python.sitePackages + "/" + dask.pname;
  distributedPath = distributed.outPath  + "/" +  python.sitePackages + "/" + distributed.pname;

  python_deps = with pkgs; [
    openssl
    pkgconfig
    pkgs.tk
    pkgs.tkgate
    pkgs.tk-8_6
    pkgs.tk-8_5
    python
    python.pkgs.tkinter
   ] ++ run_dependencies ;

    altgraph = python.pkgs.buildPythonPackage rec {
        pname = "altgraph";
        version = "v0.17";
        src = builtins.fetchGit {
            url = "https://github.com/ronaldoussoren/altgraph.git";
            ref = "refs/tags/v0.17";
        };
    };

    macholib = python.pkgs.buildPythonPackage rec {
        pname = "macholib";
        version = "v1.14";
        src = builtins.fetchGit {
            url = "https://github.com/ronaldoussoren/macholib.git";
            ref = "refs/tags/v1.14";
        };

        propagatedBuildInputs = [
            altgraph
        ];
    };

  pyinstaller = python.pkgs.buildPythonApplication rec {
    pname = "pyinstaller";
    version = "v3.6";
    src = builtins.fetchGit {
      url = "https://github.com/pyinstaller/pyinstaller.git";
      ref = "refs/tags/v3.6";
    };

    buildInputs = [
      pkgs.libgit2
      pkgs.tk

      python.pkgs.setuptools
      macholib

    ];

    # TODO: Update Apple-only imports with stdenv.isDarwin checks before adding
    propagatedBuildInputs = [
    #   pkgs.darwin.apple_sdk.frameworks.Security
    #   pkgs.darwin.apple_sdk.frameworks.AppKit
    #   pkgs.darwin.apple_sdk.frameworks.Tcl
    #   pkgs.darwin.apple_sdk.libs.Xplugin
    #   pkgs.darwin.apple_sdk.frameworks.Quartz
    #   pkgs.darwin.apple_sdk.frameworks.QuartzCore
    #   pkgs.tk
    #   pkgs.tkgate
    #   pkgs.tk-8_6
    #   pkgs.tk-8_5
        python.pkgs.setuptools
        macholib
    ];
    nativeBuildInputs = python_deps;
    doCheck = false;
  };

    poretitioner = (import ../default.nix);
    poretitionerPath = poretitioner.outPath  + "/" +  python.sitePackages + "/" + poretitioner.pname + "/poretitioner.py";

in pkgs.mkShell rec {
  name = "python-env-shell";

  buildInputs = with pkgs; [ tk python git pyinstaller poretitioner ] ++ run_dependencies;
    propagatedBuildInputs = [
      pkgs.darwin.apple_sdk.frameworks.Security
      pkgs.darwin.apple_sdk.frameworks.AppKit
      pkgs.darwin.apple_sdk.frameworks.Tcl
      pkgs.darwin.apple_sdk.libs.Xplugin
      pkgs.darwin.apple_sdk.frameworks.Quartz
      pkgs.darwin.apple_sdk.frameworks.QuartzCore
    ] ++ [pyinstaller poretitioner ] ++ run_dependencies;

  MY_ENV_VAR = "foo:bar";
  PYINSTALLER_DASK_PATH=daskPath;
  PYINSTALLER_DISTRIBUTED_PATH=distributedPath;
  PYINSTALLER_PORETITIONER_PATH=poretitionerPath;

  PYINSTALLER_OUT=''./app'';
  # shellHook = ''
  #   exec env SSL_CERT_FILE=$SSL_CERT_FILE python-env
  # '';
  shellHook = ''
    pyinstaller --add-data="${PYINSTALLER_DASK_PATH}/dask.yaml:./dask" --add-data=${PYINSTALLER_DISTRIBUTED_PATH}/distributed.yaml:./distributed --distpath="${PYINSTALLER_OUT}" -D -c --hidden-import="pkg_resources.py2_warn" "${PYINSTALLER_PORETITIONER_PATH}"
  '';
}
