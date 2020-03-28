###########################################################################################
#                       
# pre-commit.nix 
#                      
###########################################################################################
#
# The `pre-commit` third-party package helps us automate several development processes [1].
#
# Why are we building in manually instead of downloading it from the Nix store?  
# Sadly, >=v2.0 version of pre-commit isn't available in the Nix package repository yet, 
# (Nix-pkgs only has v1.2.1, which is coupled to Python 2), so we're building it ourselves 
# for now.
#
# [1] - https://github.com/pre-commit/pre-commit 
#
###########################################################################################


let nixpkgs = import <nixpkgs> {};
  packages = nixpkgs.python37.pkgs;

  precommit = packages.buildPythonApplication rec {
    pname = "pre-commit";
    version = "v2.2.0";

    src = fetchGit {
      url    = https://github.com/pre-commit/pre-commit.git;
      rev    = "30d3bb29900cf7caa6624edbaf3faf37a11b07f3";
    };

    doCheck = false;

    # Found here: https://github.com/pre-commit/pre-commit/blob/master/setup.cfg
    nativeBuildInputs = [ 
      packages.cfgv
      packages.identify
      packages.pyyaml
      packages.toml 
      packages.nodeenv
      packages.virtualenv
      packages.importlib-metadata
    ];
};
in precommit