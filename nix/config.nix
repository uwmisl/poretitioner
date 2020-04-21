###########################################################################################
#
# config.nix
#
###########################################################################################
#
# This expression hosts our Nix package configuration [1].
#   - Namely, we have to explicitly allow unfree packages.
#     as Nix will throw an error for unfree packages by default
#
# [1] - https://nixos.org/nixos/nix-pills/nixpkgs-parameters.html#idm140737319754800
# [2] - https://github.com/NixOS/nixpkgs/issues/62230
#
###########################################################################################
pkgs:
{
  allowUnfreePredicate = pkg: builtins.elem pkg.pname or (builtins.parseDrvName pkg.name) [
    "cudatoolkit"
    # The Intel Math Kernel Libary (mkl) is needed for Pytorch, but uses the ISSL license. [2]
    "mkl"
  ];
}
