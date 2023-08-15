# die-e ("dayı")
An AlphaZero implementation on backgammon to beat our dad.


Install libtorch under ./libtorch for rust-analyzer to work
https://pytorch.org/get-started/locally/

Followed https://github.com/ssoudan/tch-m1/tree/main for libtorch setup

- install miniforge with homebrew -- See https://naolin.medium.com/conda-on-m1-mac-with-miniforge-bbc4e3924f2b
- create a new conda environment: `conda env create -f environment.yml`
- activate the new environment: `conda activate tch-rs-py`
- create a symlink in this repo: `ln -sf /opt/homebrew/Caskroom/miniforge/base/envs/tch-rs-py/lib/python3.10/site-packages/torch/ torch`
- run: `cargo run`