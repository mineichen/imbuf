# NOTE for agents: you are inside `nix run .#isolated`, so the environment is built from the flake.nix.
# Edits to this file are NOT visible until the sandbox is restarted externally, you are expected not be able to run "nix run .#isolated" yourself
{
  description = "Deterministic Rust";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
    fenix.url = "github:nix-community/fenix";
    devtools.url = "github:mineichen/nix-devtools";
  };

  outputs = inputs@{ flake-parts, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [ "x86_64-linux" "aarch64-linux" "aarch64-darwin" "x86_64-darwin" ];
      perSystem = { system, ... }:
        let
          pkgs = import inputs.nixpkgs {
            inherit system;
            config.allowUnfreePredicate = pkg: builtins.elem (inputs.nixpkgs.lib.getName pkg) [ "claude-code" ];
          };
          # Latest stable for development (e.g. a current rust-analyzer)
          rust = with inputs.fenix.packages.${system}; combine [
            stable.toolchain
          ];
          # Pinned to the MSRV (`rust-version` in Cargo.toml), used by `nix run .#test`.
          # Update both together.
          rustMsrv = with inputs.fenix.packages.${system}; combine [
            (toolchainOf {
              channel = "1.89.0";
              sha256 = "sha256-+9FmLhAOezBZCOziO0Qct1NOrfpjNsXxc/8I0c7BdKE=";
            }).toolchain
          ];
          rustNightlyWithMiri = with inputs.fenix.packages.${system}; combine [
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
          commonBuildInputs = [
            rust
            pkgs.git
          ] ++ opencvBuildInputs;
          # Shared by `nix develop` and `nix run .#isolated`
          greet = ''
            echo "Deterministic dev shell: $(rustc --version), $(cargo --version)"
          '';
          libraryDirs = map (pkg: "${pkg}/lib") [
            pkgs.llvmPackages.libclang.lib
            pkgs.opencv
            pkgs.stdenv.cc.cc.lib
            pkgs.openssl.out
          ];
          rustRpathFlags = pkgs.lib.concatMapStringsSep " " (dir: "-C link-arg=-Wl,-rpath,${dir}") libraryDirs;
          # Binaries find their libraries via rpath, so no LD_LIBRARY_PATH is needed.
          # Shared by the dev shell, the isolated shell and the apps, so they share cargo's
          # build cache (different RUSTFLAGS would rebuild everything)
          devEnv = {
            # Don't let the host's libraries shadow the rpath
            LD_LIBRARY_PATH = "";
            RUSTFLAGS = rustRpathFlags;
            RUSTDOCFLAGS = rustRpathFlags;
            LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";
            # The unwrapped libclang used by the opencv binding generator doesn't know
            # the include paths of the nix toolchain ('type_traits' file not found).
            OPENCV_CLANG_ARGS = pkgs.lib.concatStringsSep " " (
              let
                gcc = pkgs.stdenv.cc.cc;
                cxxInclude = "${gcc}/include/c++/${gcc.version}";
                clangMajor = pkgs.lib.versions.major pkgs.llvmPackages.libclang.version;
              in
              pkgs.lib.optionals pkgs.stdenv.cc.isGNU [
                "-isystem ${cxxInclude}"
                "-isystem ${cxxInclude}/${pkgs.stdenv.hostPlatform.config}"
              ]
              ++ [
                "-isystem ${pkgs.llvmPackages.libclang.lib}/lib/clang/${clangMajor}/include"
              ]
              ++ pkgs.lib.optionals pkgs.stdenv.hostPlatform.isLinux [
                "-idirafter ${pkgs.stdenv.cc.libc.dev}/include"
              ]
            );
            PKG_CONFIG_PATH = pkgs.lib.makeSearchPath "lib/pkgconfig" [
              pkgs.glib.dev
              pkgs.opencv
            ];
          };
          # Miri cannot execute the OpenCV FFI calls of imbuf-opencv, so it only
          # covers the pure Rust imbuf crate.
          miriScript = pkgs.writeShellScript "miri" ''
            export PATH="${rustNightlyWithMiri}/bin:${pkgs.stdenv.cc}/bin:${pkgs.openssl.out}/bin:$PATH"
            export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc.lib pkgs.openssl.out ]}"
            exec ${rustNightlyWithMiri}/bin/cargo miri test -p imbuf-image -p imbuf
          '';
          testMsrvScript = pkgs.writeShellScript "test-msrv" ''
            set -euo pipefail
            export PATH="${rustMsrv}/bin:${pkgs.lib.makeBinPath (opencvBuildInputs ++ [ pkgs.git ])}:$PATH"
            # Don't invalidate the build cache of the development toolchain
            export CARGO_TARGET_DIR="''${CARGO_TARGET_DIR:-target/msrv}"
            rustc --version
            exec cargo test --workspace --all-features "$@"
          '';
          coverageScript = inputs.devtools.lib.cargoTarpaulin {
            inherit pkgs rust;
            path = opencvBuildInputs ++ [ pkgs.git ];
            env = devEnv;
          };
          isolated = inputs.devtools.lib.isolated {
            inherit pkgs inputs;
            name = "imbuf-isolated-dev";
            packages = commonBuildInputs ++ [
              pkgs.coreutils
              pkgs.curl
              pkgs.findutils
              pkgs.diffutils
              pkgs.gnugrep
              pkgs.gnused
              pkgs.gawk
              pkgs.gnutar
              pkgs.gzip
              pkgs.which
              pkgs.less
              pkgs.procps
              pkgs.ripgrep
              pkgs.opencode
            ];
            claude.pluginDirs = [
              (inputs.devtools.lib.claudeRustAnalyzer {
                inherit pkgs;
                # The one of the development toolchain
                rust-analyzer = rust;
              })
            ];
            env = devEnv;
            # `nix run .#test`, `.#miri` and `.#coverage` without downloading the toolchains
            extraStorePaths = [ testMsrvScript miriScript coverageScript ];
            shellHook = greet;
          };
        in
        {
          devShells.default = pkgs.mkShell ({
            buildInputs = commonBuildInputs ++ [
              pkgs.bashInteractive
            ];

            shellHook = greet;
          } // devEnv);

          apps.miri = {
            type = "app";
            program = toString miriScript;
          };

          # Runs the whole test suite with the MSRV toolchain. Arguments are passed to `cargo test`.
          apps.test = {
            type = "app";
            program = toString testMsrvScript;
          };

          apps.coverage = {
            type = "app";
            program = toString coverageScript;
          };

          # nix run .#isolated [dir], see isolated.nix of nix-devtools
          apps.isolated = {
            type = "app";
            program = pkgs.lib.getExe isolated;
          };
        };
    };
}
