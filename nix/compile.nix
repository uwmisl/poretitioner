###########################################################################################
#
# compile.nix
#
###########################################################################################
#
# This expression runs PyInstaller to package our python code into as few resources
# as possible. It creates an "app" directory in the working directory, then puts the
# final executable at: app/poretitioner/poretitioner
# linters, git hooks, and static analyzers.
#
# Run this compiler by running "nix-shell ./nix/compile.nix"
#
###########################################################################################

{ pkgs ? import <nixpkgs> { } }:

let
  # picked a conservative python version because of errors regarding `cf_feature_version`
  python = pkgs.python37;
  pyinstaller =
    (pkgs.callPackage ./pkgs/pyinstaller/pyinstaller.nix) { inherit python; };

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

  poretitioner = import ../default.nix;
  poretitionerPath = builtins.toPath (poretitioner.outPath + "/"
    + python.sitePackages + "/" + poretitioner.pname + "/poretitioner.py");

in pkgs.mkShell rec {
  name = "install-poretitioner";

  buildInputs = with pkgs; [ pyinstaller poretitioner ];

  # Environement variables.
  PYINSTALLER_DASK_PATH = daskPath;
  PYINSTALLER_DISTRIBUTED_PATH = distributedPath;
  PYINSTALLER_PORETITIONER_PATH = poretitionerPath;

  PYINSTALLER_OUT = "./app";

  shellHook = ''
    pyinstaller --add-data="${PYINSTALLER_DASK_PATH}/dask.yaml:./dask" --add-data=${PYINSTALLER_DISTRIBUTED_PATH}/distributed.yaml:./distributed --distpath="${PYINSTALLER_OUT}" -D -c --hidden-import="pkg_resources.py2_warn" "${PYINSTALLER_PORETITIONER_PATH}"
  '';
}
