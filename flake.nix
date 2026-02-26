{
  description = "SRT LLM Processor - Post-processes subtitle files using LLM";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";

    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows = "uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      uv2nix,
      pyproject-nix,
      pyproject-build-systems,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        inherit (pkgs) lib;

        python = pkgs.python311;

        workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./.; };

        overlay = workspace.mkPyprojectOverlay {
          # Prefer prebuilt wheels for faster builds
          sourcePreference = "wheel";
        };

        # Build fixups for packages that don't declare their build systems properly.
        # srt 3.5.3 is sdist-only and uses setuptools without declaring it.
        pyprojectOverrides = final: _prev: {
          srt = _prev.srt.overrideAttrs (old: {
            nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [ final.setuptools ];
          });
        };

        pythonSet =
          (pkgs.callPackage pyproject-nix.build.packages {
            inherit python;
          }).overrideScope
            (
              lib.composeManyExtensions [
                pyproject-build-systems.overlays.wheel
                overlay
                pyprojectOverrides
              ]
            );

        # Build a virtualenv with only the runtime dependencies.
        # The project itself is a script, not an installable package, so we
        # enumerate the direct deps from pyproject.toml explicitly.
        virtualenv = pythonSet.mkVirtualEnv "srt-llm-processor-env" {
          httpx = [ ];
          openai = [ ];
          rich = [ ];
          srt = [ ];
        };

        # Copy only the source files needed at runtime into the Nix store
        appSrc = pkgs.stdenv.mkDerivation {
          name = "srt-llm-processor-src";
          src = lib.cleanSource ./.;
          dontBuild = true;
          installPhase = ''
            mkdir -p $out
            cp main.py $out/
            cp -r src $out/src
          '';
        };
      in
      {
        packages.default = pkgs.writeShellApplication {
          name = "srt-llm-processor";
          text = ''
            PYTHONPATH="${appSrc}" exec "${virtualenv}/bin/python" "${appSrc}/main.py" "$@"
          '';
        };

        apps.default = {
          type = "app";
          program = "${self.packages.${system}.default}/bin/srt-llm-processor";
        };

        devShells.default = pkgs.mkShell {
          packages = [
            pkgs.uv
            python
          ];
          shellHook = ''
            unset PYTHONPATH
            export UV_PYTHON_DOWNLOADS=never
          '';
        };
      }
    );
}
