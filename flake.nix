{
  description = "Nanopore signal analysis software.";

  inputs = {
    nixpkgs = {
      url = "github:NixOS/nixpkgs";
    };

    flake-utils = {
      url = "github:numtide/flake-utils";
    };

    pypi-deps-db = {
      url = github:DavHau/pypi-deps-db;
      flake = false;
    };

    mach-nix = {
      url = github:DavHau/mach-nix;
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.flake-utils.follows = "flake-utils";
      inputs.pypi-deps-db.follows = "pypi-deps-db";
    };
  };

  outputs = { self, nixpkgs, flake-utils, pypi-deps-db, mach-nix }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        buildOptions = {
            # Whether to build for release mode.
            isRelease = false;
        };


        pname = "poretitioner";
        version = "0.5";
        pkgs = nixpkgs.legacyPackages.${system};

        mach-nix-utils = import mach-nix {
          inherit pkgs;
        #   pypiDataRev = "90885674ac9a4005ec88b05904dc25b653363ab4";
        #   pypiDataSha256 = "08q5ii3k1p1azix6mwc3dkqc0l4p4md5zbjrdf9f6p3fyxwqk36g";
        };

        python = mach-nix-utils.mkPython {
          python = "python38";
          requirements = builtins.readFile ./requirements.txt;
          packagesExtra = [
          ]
          
          ++ pkgs.lib.optional (buildOptions.isRelease) [ "bpython" ];
        };

        pythonShell = mach-nix-utils.mkPythonShell {
          python = "python38";
          requirements = builtins.readFile ./requirements.txt;
          packagesExtra = [
          ]
          
          ++ pkgs.lib.optional (buildOptions.isRelease) [ "bpython" ];
        };

        customOverrides = self: super: {
          # Overrides go here
        };

        app = mach-nix-utils.buildPythonApplication {
          inherit pname version;

          python = "python38";
          requirements = builtins.readFile ./requirements.txt;

          # add missing dependencies whenever necessary.
          packagesExtra = [
            python
          ];
          src = ./.;
        };

        packageName = "poretitioner";
      in {
          packages.${packageName} = app;

          defaultPackage = self.packages.${system}.${packageName};

          devShell = pkgs.mkShell {
            shellHook = ''
                # Patch Visual Studio Code's workspace `settings.json` so that nix's python is used as default value for `python.pythonPath`.
                # That way, the debugger will know where all your dependencies are, etc.
                #
                if [ -e "./.vscode/settings.json" ]; then
                  echo "Setting VSCode Workspace's Python path for Nix:"
                  cat .vscode/settings.json  | jq '. + {"python.defaultInterpreterPath": "${python}/bin/python"}' | tee .vscode/settings.json | grep "python.pythonPath"
                fi
              '';

              buildInputs = [
                pkgs.bash
                pkgs.bashInteractive
                pkgs.locale
                pkgs.xtermcontrol
                pkgs.xterm
                pkgs.zsh
                pkgs.jq

                python
              ];
          };


          # devShell = import ./nix/shell.nix { inherit pkgs python; };
      });
}
