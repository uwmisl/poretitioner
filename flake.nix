{
  description = "Nanopore signal analysis software.";

  inputs = {
    nixpkgs = {
      url = "github:NixOS/nixpkgs/22.05-pre";
    };

    flake-utils = {
      url = "github:numtide/flake-utils";
    };

    pypi-deps-db = {
      url = github:DavHau/pypi-deps-db;
      flake = false;
    };

    mach-nix = {
      url = github:DavHau/mach-nix/3.4.0;
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

          # Add in freeze_support() for Dask, allowing us to use process-based clusters [1] (note: this is not always optimal over thread-based approaches, depending on the data and computation. For a discussion, please see [2]).
          # This must be done in the __name__ == '__main__' block, which is created by the Nix python wrapper.
          #
          # [1] - https://github.com/dask/distributed/issues/2422
          # [2] - https://iotespresso.com/numpy-releases-gil-what-does-that-mean/
          #
          postFixup = ''
            sed -i "s/if __name__ == '__main__':/if __name__ == '__main__':\n    import dask.multiprocessing as dask_mp\n    dask_mp.multiprocessing.freeze_support()\n    from dask.distributed import Client, LocalCluster\n    cluster = LocalCluster()\n    client = Client(cluster)/g" $out/bin/.${pname}-wrapped;
          '';
        };

        packageName = "poretitioner";


        # Create docker package
        #buildDocker = { app }:
      in {
          #packages.${packageName} = app;

          packages = flake-utils.lib.flattenTree {
            inherit app;

            defaultPackage = app;

            docker = pkgs.dockerTools.buildImage {
              name = "${app.pname}";
              tag = "latest";

              # Setting 'created' to 'now' will correctly set the file's creation date
              # (instead of setting it to Unix epoch + 1). This is impure, but fine for our purposes.
              created = "now";
              config = {
                Cmd = [ ];
                # Runs 'poretitioner' by default.
                Entrypoint = [ "${app.outPath}/bin/${app.pname}" ];
              };
            };
            #gitAndTools = pkgs.gitAndTools;
          };

          defaultPackage = app;

          #packages.${system}.docker =

          devShell =
          let
          # Adds a function so that vsdebugpy runs debugpy on localhost port 5678.
          # In VSCode, you can connect to the Python Remote Debugger on port 5678
          debugpyHook = ''vsdebugpy() { python -m debugpy --wait-for-client --listen localhost:5678 "$0"; }'';

          # Patch Visual Studio Code's workspace `settings.json` so that nix's python is used as default value for `python.pythonPath`.
          # That way, the debugger will know where all your dependencies are, etc.
          #
          vscodePythonHook = ''
                if [ -e "./.vscode/settings.json" ]; then
                  echo "Setting VSCode Workspace's Python path for Nix:"
                  cat .vscode/settings.json  | jq '. + {"python.defaultInterpreterPath": "${python}/bin/python"}' | tee .vscode/settings.json | grep "python.pythonPath"
                fi
              '';

          allHooks = [ vscodePythonHook debugpyHook ];
          in
          pkgs.mkShell {
            shellHook = pkgs.lib.concatStringsSep "\n" allHooks;

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
      });
}
