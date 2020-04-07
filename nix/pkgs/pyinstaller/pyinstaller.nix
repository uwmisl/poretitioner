{ lib , stdenv, python } :
let
  pypkgs = python.pkgs;

  # We need to do some manual work for the dask and distributed dependencies
  # to ensure that PyInstaller picks up on their non-Python files.
  # https://stackoverflow.com/questions/57057336/dask-pyinstaller-fails
  dask = pypkgs.dask;
  distributed = pypkgs.distributed;
  daskPath = builtins.toPath
    (dask.outPath + "/" + python.sitePackages + "/" + dask.pname);
  distributedPath = builtins.toPath
    (distributed.outPath + "/" + python.sitePackages + "/" + distributed.pname);

  # Dependency of pyinstaller.
  # Not currently available in Nixpkgs (As of 6 April 2020)
  altgraph = pypkgs.buildPythonPackage rec {
    pname = "altgraph";
    version = "v0.17";
    src = builtins.fetchGit {
      url = "https://github.com/ronaldoussoren/altgraph.git";
      ref = "refs/tags/v0.17";
    };
  };

  # TO SUPPORT WINDOWS: We must add the Win32 required packages as well.
  # https://github.com/pyinstaller/pyinstaller/blob/develop/requirements.txt

  # (Darwin) Dependency of pyinstaller.
  macholib = pypkgs.buildPythonPackage rec {
    pname = "macholib";
    version = "v1.14";
    src = builtins.fetchGit {
      url = "https://github.com/ronaldoussoren/macholib.git";
      ref = "refs/tags/v1.14";
    };

    propagatedBuildInputs = [ altgraph ];
  };

  # Not currently available in Nixpkgs (as of 6 April 2020).
  pyinstaller = pypkgs.buildPythonApplication rec {
    pname = "pyinstaller";
    version = "v3.6";
    src = builtins.fetchGit {
      url = "https://github.com/pyinstaller/pyinstaller.git";
      ref = "refs/tags/v3.6";
    };

    # TODO: Update Darwin-only imports with stdenv.isDarwin checks before adding
    propagatedBuildInputs = [ pypkgs.setuptools ]
      ++ lib.optional (stdenv.isDarwin) macholib;
    doCheck = false;
  };

  in pyinstaller
