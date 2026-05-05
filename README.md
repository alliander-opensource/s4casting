<!--
SPDX-FileCopyrightText: Contributors to the s4casting project

SPDX-License-Identifier: MPL-2.0
-->
# S4Casting
S4Casting is a forecasting toolkit built around several deep learning models:
- **State Space Models (SSMs)** such as S4 and selective SSMs (a.k.a. S6 / Mamba-style models)
- **Transformer variants**

It is designed for medium voltage power forecasting tasks, including:
- **Short-term forecasting** – covering time horizons of days, typically used for forecasts up to 2 days ahead.
- **Mid- and long-term forecasting** – covering time horizons of months to years, typically used for 1-year ahead planning.

The repo contains:
- Training loop (distributed / torchrun-ready)
- Evaluation and benchmarking pipeline
- Config-driven experiments (CPU / CUDA configs)
- Inference scripts for running trained models on new data

## Installation
You can install the repo following the installation guide:
- [Installation](docs/INSTALLATION.md)

## Documentation

You can find detailed documentation for each component in the following links:

- [Usage](docs/USAGE.md)
- [Data Sources](docs/DATA_SOURCES.md)
- [Model settings](docs/CONFIGURATION.md)
- [Evaluation](docs/EVALUATION.md)
- [Repository](docs/REPOSITORY.md)
- [Contributing](docs/CONTRIBUTING.md)

For more technical details, please also see our CIRED 2025 paper: [data/CIRED_2025_paper.pdf](data/CIRED_2025_paper.pdf)

## Example Notebooks

You can find example notebooks in the `notebooks/` directory, demonstrating how to use the S4Casting toolkit for forecasting tasks.

## LF Energy OpenSTEF Support
Future s4casting models will be integrated into the [LF Energy OpenSTEF project](https://github.com/OpenSTEF/openstef) in the near future. OpenSTEF provides reusable, automated machine‑learning pipelines for generating accurate and explainable short‑term grid load forecasts (up to 48 hours ahead). By integrating s4casting models into OpenSTEF, we can leverage shared infrastructure, align on common data and ML patterns, and reduce model‑specific complexity. At the same time, this integration expands the set of available forecasting models within OpenSTEF, improving overall maintainability, consistency, and reuse across implementations.

## License
This project is licensed under the Mozilla Public License, version 2.0 - see [LICENSE](Add Link) for details.

## Licenses third-party libraries

This project includes third-party libraries,
which are licensed under their own respective Open-Source licenses.
SPDX-License-Identifier headers are used to show which license is applicable.
The concerning license files can be found in the
[LICENSES](Add Link) directory.

## Contributing

Please read [CODE_OF_CONDUCT](docs/CODE_OF_CONDUCT.md),
[CONTRIBUTING](docs/CONTRIBUTING.md),
[GOVERNANCE](docs/GOVERNANCE.md) and
[RELEASE](docs/CONTRIBUTING.md) for details on the process for submitting pull
requests to us.

## Citation
If you find our code or models useful in your work, please cite our paper:

```bibtex
@article{doi:10.1049/icp.2025.1968,
  author = {Michael Mesarcik and Jessica Loke and Jochem Wildeboer and Bob Lucassen},
  title = {Probabilistic day-ahead power forecasting in the medium-voltage grid using state space models},
  journal = {IET Conference Proceedings},
  volume = {2025},
  issue = {14},
  pages = {1947-1951},
  year = {2025},
  doi = {10.1049/icp.2025.1968},
  URL = {https://digital-library.theiet.org/doi/abs/10.1049/icp.2025.1968},
  eprint = {https://digital-library.theiet.org/doi/pdf/10.1049/icp.2025.1968}
}
```

## Contact
Please read [SUPPORT](DOCS/SUPPORT.md) for how to connect and get into
contact with the S4Casting project.

