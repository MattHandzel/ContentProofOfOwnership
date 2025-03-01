{pkgs ? import <nixpkgs> {}}:
with pkgs;
  mkShell {
    buildInputs = [
      python311
      python311Packages.soundfile
      python311Packages.numpy
      python311Packages.matplotlib
      python311Packages.librosa
      python311Packages.torch
      python311Packages.torchaudio

      python311Packages.scipy
      python311Packages.fastapi
      python311Packages.uvicorn
      python311Packages.python-multipart
      python311Packages.scikit-learn
      python311Packages.virtualenv

      # (pkgs.glibc.overrideAttrs (oldAttrs: {
      #   version = "2.39";
      # }))
    ];

    shellHook = ''
      if ! [-e .venv]; then
        virtualenv .venv
        source .venv/bin/activate
        pip install -r requirements.txt
      else
        source .venv/bin/activate
      fi

    '';

    NIX_LD_LIBRARY_PATH = lib.makeLibraryPath [
      stdenv.cc.cc
      zlib
    ];
  }
