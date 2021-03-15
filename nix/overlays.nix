self: super:

{
  httplib2 = super.httplib2.override {
    # Fixes "ConnectionResetError: [Errno 54] Connection reset by peer"
    # which causes check phase to fail on unit test "tests/test_other.py:134: in test_timeout_subsequent"
    # https://github.com/NixOS/nixpkgs/pull/116162
    __darwinAllowLocalNetworking = true;
  };
#   rr = super.callPackage ./pkgs/rr {
#     stdenv = self.stdenv_32bit;
#   };
}
