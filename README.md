# Hazardous Orienteering Problem

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6381846.svg)](https://doi.org/10.5281/zenodo.6381846)

Code, instance generator, instance files and results for the Hazardous Orienteering Problem.

The accompanying paper is [available on-line](https://santini.in/files/papers/santini-archetti-2022.pdf) and can be cited as follows:

```bib
@article{Santini_Archetti_2022,
    author={Santini, Alberto and Archetti, Claudia},
    title={The Hazardous Orienteering Problem},
    year=2022,
    journal={Networks},
    doi={10.1002/net.22129},
    url={https://santini.in/files/papers/santini-archetti-2022.pdf}
}
```

You can cite this repository itself via Zenodo:

```bib
@misc{hop_2022,
    title={Code and instances for the Hazardous Orienteering Problem},
    author={Santini, Alberto},
    date=2022-08-30,
    year=2022,
    doi={10.5281/zenodo.6381846},
    url={https://github.com/alberto-santini/hazardous-orienteering-problem},
    howpublished={Github repository},
}
```

## Structure

The main bulk of the code is in folder `hop`, while folder `run` contains Python scripts to run the various solvers from the command line or on a Slurm cluster.
You can call `run/run_hop.py --help` to have an idea of how to use the solvers.

Folder `data` contains the instances.
The original Orienteering Problem instances are in `data/op-tsiligirides`.
The Hazardous Orienteering Problem instances we generated are in `data/hop-tsiligirides`.

Folder `results/run` contains all results files, in comma-separated values (csv) format.
The Jupyter notebooks used to analyse results and produce tables and figures are in folder `analysis`.

## License

Distributed under the GPLv3.0 License. See the `LICENSE` file.
