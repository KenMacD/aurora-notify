{
  description = "Aurora Forecast Coordinate Finder";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        
        python = pkgs.python313;
        
        auroraApp = pkgs.python3Packages.buildPythonApplication {
          pname = "aurora";
          version = "0.1.0";
          
          src = ./.;
          
          pyproject = true;

          build-system = with pkgs.python3Packages; [ uv-build ];

          propagatedBuildInputs = with pkgs.python3Packages; [
            python-dotenv
            requests
          ];
        };
        
        # Development environment
        devShells = {
          default = pkgs.mkShell {
            buildInputs = with pkgs; [
              python
              python313Packages.pip
              python313Packages.python-dotenv
              python313Packages.requests
              python313Packages.uv-build
              python313Packages.pytest
              python313Packages.ruff
            ];
            
            # Set up environment for development
            shellHook = ''
              export PYTHONPATH="$PWD/src:$PYTHONPATH"
              echo "Development environment ready!"
              echo "Run 'python -m aurora <lat> <lon>' to execute the application"
            '';
          };
        };
      in
      {
        packages = {
          default = auroraApp;
          aurora = auroraApp;
        };
        
        apps = {
          default = flake-utils.lib.mkApp {
            drv = auroraApp;
          };
        };
        
        devShells = devShells;
      });
}
