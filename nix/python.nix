###########################################################################################
#
# python.nix
#
###########################################################################################
#
# This expression customizes our Python environment, including package overrides.
#
#   - This is useful for things like specifying Cuda vs. Non-Cuda Pytorch
#     and which backend graphics matplotlib should use.
#
###########################################################################################


{ pkgs, cudaSupport ? true }:
let
  python = pkgs.python37; # Specifies Python version. In our case, Python 3.7;

  # The following overrides customize our Python packages, like Cuda vs. Non-Cuda Pytorch.
  pythonWithOverrides = (python.withPackages (ps:
    with ps;
    let
      # Build Pytorch with or without Cuda support.
      # Pytorch with CUDA example: https://github.com/NixOS/nixpkgs/issues/62230
      newPytorch = pytorch.override {
        inherit cudaSupport;
        cudatoolkit = pkgs.cudatoolkit_10;
        cudnn = pkgs.cudnn_cudatoolkit_10;
      };

      # Build TorchVision with properly Cuda'ed Pytorch.
      newTorchvision = torchvision.override { pytorch = mypytorch; };
    in [ newPytorch newTorchvision ]));
in pythonWithOverrides
