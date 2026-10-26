{
  description = "Deterministic Rust";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
    fenix.url = "github:nix-community/fenix";
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
          rust = with inputs.fenix.packages.${system}; combine [
            stable.toolchain
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
          greet = ''
            echo "===================================="
            echo " Welcome to the deterministic dev shell! "
            echo "===================================="
            cargo --version
          '';
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
          rustRpathFlags = pkgs.lib.concatStringsSep " " (
            map (dir: "-C link-arg=-Wl,-rpath,${dir}")
              (pkgs.lib.splitString ":" opencvEnvVars.LD_LIBRARY_PATH)
          );
          policy = pkgs.writeText "policy.json" ''{"default":[{"type":"insecureAcceptAnything"}]}'';
          containername = "imbuf-isolated-dev";
          podmanRuntimePath = pkgs.lib.makeBinPath [
            pkgs.slirp4netns
          ];
          podmanRun = "${pkgs.podman}/bin/podman run --rm -it "
            + "--network=slirp4netns "
            + "--tmpfs /tmp "
            + "-v .:/workspace/imbuf:z "
            + "-e HOME=/root "
            + "${containername}:latest /bin/entrypoint.sh";
        in
        {
          devShells.default = pkgs.mkShell ({
            buildInputs = commonBuildInputs ++ [
              pkgs.bashInteractive
              pkgs.cargo-tarpaulin
            ];

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
          } // opencvEnvVars // {
            LD_LIBRARY_PATH = "";
            RUSTFLAGS = rustRpathFlags;
            RUSTDOCFLAGS = rustRpathFlags;
          });

          apps.miri = {
            type = "app";
            # Miri cannot execute the OpenCV FFI calls of imbuf-opencv, so it only
            # covers the pure Rust imbuf crate.
            program = toString (pkgs.writeShellScript "miri" ''
              export PATH="${rustNightlyWithMiri}/bin:${pkgs.stdenv.cc}/bin:${pkgs.openssl.out}/bin:$PATH"
              export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc.lib pkgs.openssl.out ]}"
              exec ${rustNightlyWithMiri}/bin/cargo miri test -p imbuf-image -p imbuf
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

          packages.isolated-build = pkgs.dockerTools.buildImage {
            name = containername;
            tag = "latest";
            copyToRoot = pkgs.buildEnv {
              name = containername;
              paths = commonBuildInputs ++ [
                pkgs.bashInteractive
                pkgs.ripgrep
                pkgs.git
                pkgs.opencode
                pkgs.claude-code
                pkgs.busybox
                pkgs.cargo-tarpaulin
                (pkgs.writeScriptBin "entrypoint.sh" ''
                  #!${pkgs.bashInteractive}/bin/bash
                  ${greet}
                  exec ${pkgs.bashInteractive}/bin/bash
                '')
              ];
              pathsToLink = [ "/bin" "/lib" "/include" "/share" ];
            };
            config = {
              Env = [
                "SSL_CERT_FILE=${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt"
                "HOME=/root"
              ] ++ pkgs.lib.mapAttrsToList (name: value: "${name}=${value}") opencvEnvVars;
              Cmd = [ "/bin/entrypoint.sh" ];
              WorkingDir = "/workspace/imbuf";
            };
          };

          apps.isolated-build = {
            type = "app";
            program = toString (pkgs.writeShellScript containername ''
              export PATH="${podmanRuntimePath}:$PATH"
              ${pkgs.podman}/bin/podman rmi ${containername} || true
              ${pkgs.podman}/bin/podman load \
                --signature-policy ${policy} \
                --input ${inputs.self.packages.${system}.isolated-build}
              ${podmanRun}
            '');
          };

          apps.isolated-nobuild = {
            type = "app";
            program = toString (pkgs.writeShellScript "run-isolated" ''
              set -euo pipefail
              export PATH="${podmanRuntimePath}:$PATH"
              ${podmanRun}
            '');
          };
        };
    };
}
