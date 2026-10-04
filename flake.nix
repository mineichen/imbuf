{
  description = "Deterministic Rust";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    fenix.url = "github:nix-community/fenix";
  };

  outputs = { self, nixpkgs, flake-utils, fenix, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        rust = with fenix.packages.${system}; combine [
          stable.toolchain
        ];
        rustNightlyWithMiri = with fenix.packages.${system}; combine [
          (latest.withComponents [
            "rustc"
            "cargo"
            "miri"
          ])
        ];
        # Required to build and run the imbuf-opencv test suite.
        opencvBuildInputs = [
          pkgs.stdenv.cc
          pkgs.llvmPackages.libclang
          pkgs.cmake
          pkgs.opencv
          pkgs.pkg-config
        ];
        opencvEnvVars = {
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
            pkgs.llvmPackages.libclang.lib
            pkgs.opencv
            pkgs.stdenv.cc.cc.lib
            pkgs.openssl.out
          ];
          LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";
          PKG_CONFIG_PATH = pkgs.lib.makeSearchPath "lib/pkgconfig" [
            pkgs.glib.dev
            pkgs.opencv
          ];
        };
        envVarAssignments = pkgs.lib.concatMapStringsSep " " (
          name: value: "${name}=\"${value}\""
        ) opencvEnvVars;
      in
      {
        devShells.default = pkgs.mkShell ({
          buildInputs = [
            rust
            pkgs.bashInteractive
            pkgs.cargo-tarpaulin
          ] ++ opencvBuildInputs;

          shellHook = ''
            echo "===================================="
            echo " Welcome to the deterministic dev shell! "
            echo "===================================="
            echo "Rust toolchain:"
            rustc --version
            echo "Cargo version:"
            cargo --version
            echo "OpenCV version:"
            pkg-config --modversion opencv4 || echo "opencv not found via pkg-config"
            echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
            echo "LIBCLANG_PATH: $LIBCLANG_PATH"
            echo "PKG_CONFIG_PATH: $PKG_CONFIG_PATH"
            echo "===================================="
            echo "Ready to develop! 🦀"
          '';
        } // opencvEnvVars);

        apps.miri = {
          type = "app";
          # Miri cannot execute the OpenCV FFI calls of imbuf-opencv, so it only
          # covers the pure Rust imbuf crate.
          program = toString (pkgs.writeShellScript "miri" ''
            export PATH="${rustNightlyWithMiri}/bin:${pkgs.openssl.out}/bin:$PATH"
            export LD_LIBRARY_PATH="${pkgs.openssl.out}/lib"
            exec ${rustNightlyWithMiri}/bin/cargo miri test -p imbuf
          '');
        };

        apps.coverage = {
          type = "app";
          program = toString (pkgs.writeShellScript "coverage" ''
            export ${pkgs.lib.concatStringsSep " " envVarAssignments}
            set -e

            # Create coverage directory
            mkdir -p coverage

            # Run tests with coverage
            echo "Running tests with coverage..."
            ${pkgs.cargo-tarpaulin}/bin/cargo-tarpaulin \
              --out Html \
              --out Xml \
              --output-dir target/coverage \
              --exclude-files 'tests/*' \
              --exclude-files 'target/*' \
              --timeout 120 \
              --all-features

            echo ""
            echo "Coverage report generated in coverage/tarpaulin-report.html"
            echo "XML report generated in coverage/cobertura.xml"
            echo ""
            echo "Open coverage/tarpaulin-report.html in your browser to view the report."
          '');
        };
      });
}
