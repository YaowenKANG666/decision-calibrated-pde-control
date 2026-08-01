# Official NS2D dataset

The optional two-dimensional benchmark uses the public NeuralOperator
Navier--Stokes dataset:

- source: `https://zenodo.org/records/12825163`;
- DOI: `10.5281/zenodo.12825163`;
- official code: `neuraloperator/neuraloperator`;
- physical system: 2D incompressible Navier--Stokes, Reynolds number 500;
- contents: input/output PyTorch tensors representing fluid time evolution;
- original numerical resolution: 1024 x 1024;
- distributed archives: 128 x 128 (about 1.5 GB) and 1024 x 1024
  (about 15.4 GB).

The open-source repository does not redistribute the archive. Download and
verify the practical 128 x 128 version with:

```bash
python scripts/download_ns2d.py --root /path/to/ns2d --resolution 128
```

Expected archive MD5:

```text
70a389207ac93935d5ff3d4289d43581  nsforcing_128.tgz
```

The official NeuralOperator loader can also download the same archive:

```python
from neuralop.data.datasets import NavierStokesDataset

dataset = NavierStokesDataset(
    root_dir="/path/to/ns2d",
    n_train=1200,
    n_tests=[400],
    batch_size=8,
    test_batch_sizes=[8],
    train_resolution=128,
    test_resolutions=[128],
    encode_input=False,
    encode_output=False,
    download=True,
)
```

The downloaded `.pt` tensors and compressed archive are ignored by Git. Only
download scripts, checksums, metadata, and small derived result tables belong
in the repository.
