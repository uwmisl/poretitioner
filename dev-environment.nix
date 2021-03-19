{ pkgs ?  import <nixpkgs> { config = (import ./nix/config.nix); overlays = [ (import ./nix/overlays.nix) ]; }
, python ? pkgs.callPackage ./python.nix { inherit pkgs; }
, cudaSupport ? false
, poretitioner ? pkgs.callPackage ./default.nix { inherit pkgs python cudaSupport; }
, postShellHook ? ""
}:

let devEnvironmentImage = dockerTools.buildImage {
    name = "misl-dev";
    tag = "latest";

    fromImageName = "nixos/nix";

    dockerImage = ;
    name = "${name}";
    tag = "latest";

    # Note: we're only running as root inside the container ;)
    runAsRoot = ''
      #!${pkgs.runtimeShell}
      mkdir -p /data
    '';

    # Setting 'created' to 'now' will correctly set the file's creation date
    # (instead of setting it to Unix epoch + 1). This is impure, but fine for our purposes.
    created = "now";
    config = {
      Cmd = [ "${pkgs.zsh}" ];
      # Runs 'poretitioner' by default.
      Entrypoint = [ "${pkgs.zsh}/bin/zsh" "nix-shell -A shell" ];
    };
  };
in 
devEnvironmentImage