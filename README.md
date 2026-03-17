# Real-Time Loss Tools

Tools for the assessment of earthquake damage and loss in real-time, developed within the
[RISE project](http://rise-eu.org/home/).

The Real-Time Loss Tools carry out rapid loss assessments (RLA) and event-based operational
earthquake loss forecasts (OELF) incorporating probabilities of damage states based on
structural health monitoring (SHM) methods, calculating cumulative damage, expected economic
losses and human casualties (injuries and deaths) and updating the number of occupants in the
buildings, by taking into account the time of the day of the earthquake as well as whether
people are allowed back (due to inspection and repair times) and are able to do so (due to their
own health status). The tools recursively call the
[OpenQuake engine](https://github.com/gem/oq-engine) and update the exposure model and other
relevant input files.

These tools are not an operational deployment but a research tool made openly available to the
community.

## Installation

It is recommended that a Python >= 3.12 virtual environment be created so as to install and run
the software from within it.

There are two main steps to this installation: (1) installation of the OpenQuake engine and (2)
installation of the Real-Time Loss Tools themselves. Detailed steps are provided in what
follows.

### Installing the OpenQuake Engine

1. Create a new Python >= 3.12 virtual environment and activate it (it is strongly recommended
to upgrade pip: `pip install --upgrade pip`).
2. Clone OpenQuake v3.25.1 from [https://github.com/gem/oq-engine.git](https://github.com/gem/oq-engine.git)
(e.g., `git clone --single-branch --branch v3.25.1 https://github.com/gem/oq-engine.git`).
3. Navigate to the local OpenQuake repository and run
`pip install -r requirements-py312-osname.txt`, where `osname` stands for the name of your OS
(e.g., `requirements-py312-linux64.txt` for Linux).
4. From the same local OpenQuake repository run `pip3 install -e .`.
5. If older versions of OpenQuake had been installed, it might be necessary to update the
database by doing `oq engine --upgrade-db`.


### Installing the Real-Time Loss Tools
1. Activate the virtual environment created in Step 1 above (if not already activated).
1. Clone the `real-time-loss-tools` repository
(`git clone https://git.gfz-potsdam.de/real-time-loss-tools/real-time-loss-tools.git`).
2. Navigate to the local `real-time-loss-tools` repository and run `pip3 install -e .`.

### Software dependencies

- Python 3.12 or above
- OpenQuake engine 3.25.1 or above

### Python libraries

- `pyyaml` (v6.0.3 or above)
- `numpy` (v2.2.6 or above)
- `pandas` (v2.2.3 or above)
- `shapely` (v2.1.0 or above)
- `pyproj` (v3.7.2 or above)
- `rtree` (v1.4.1 or above)
- `geopandas` (v1.1.1 or above)
- `pytz` (v2023.3 or above)

## Running

### Preparation

A series of [input files](docs/03_Input.md) and a specific
[file structure](docs/03_Input.md#assumed-file-structure) are needed to run the
`Real Time Loss Tools`. Please click on the corresponding links for details and follow the
instructions.

Once the input files and the file structure have been created, create a `config.yml` file
following the example contained in this repository as
[config_example.yml](./config_example.yml) and the corresponding
[documentation](docs/02_Configuration.md). The `config.yml` file can be located anywhere, but
the `Real Time Loss Tools` need to be run from the directory where `config.yml` is located. It
can be placed, for example, within `main_path`.

### Execution

Having activated the virtual environment where the `Real Time Loss Tools` are installed,
navigate to the directory where you placed `config.yml`. Then type:

```bash
(YourPreferredName) $ rtlt
```

The program will start to run.

### Execution in debug mode

In order to get debug information in the log, set `debug_logging: True` in `config.yml`.

### Example files

A full set of example input and output files can be found
[here](https://git.gfz-potsdam.de/real-time-loss-tools/rise-d6-1-data-files). These are the
input files used for a proof of concept developed as part of Task 6.1 of the
[RISE project](http://rise-eu.org/home/), as well as the associated main outputs.

## Acknowledgements

These tools have been developed within the [RISE project](http://rise-eu.org/home/), which has
received funding from the European Union's Horizon 2020 research and innovation programme under
grant agreement No. 821115.

## Citation

Please cite:

Nievas CI, Crowley H, Weatherill G, Cotton F (2025) Real-Time Loss Tools: Open-Source Software
for Time- and State-Dependent Seismic Damage and Loss Calculations - Features and Application to
the 2023 Turkiye-Syria Sequence. Seismica, 4(1).
[https://doi.org/10.26443/seismica.v4i1.1238](https://doi.org/10.26443/seismica.v4i1.1238)

## Documentation

Software documentation: click [here](docs/README.md).

Scientific documentation:

Nievas CI, Crowley H, Weatherill G, Cotton F (2025) Real-Time Loss Tools: Open-Source Software
for Time- and State-Dependent Seismic Damage and Loss Calculations - Features and Application to
the 2023 Turkiye-Syria Sequence. Seismica, 4(1).
[https://doi.org/10.26443/seismica.v4i1.1238](https://doi.org/10.26443/seismica.v4i1.1238)

Additional documentation (more details, input formats, etc):

Nievas CI, Crowley H, Reuland Y, Weatherill G, Baltzopoulos G, Bayliss K, Chatzi E, Chioccarelli
E, Gueguen P, Iervolino I, Marzocchi W, Naylor M, Orlacchio M, Pejovic J, Popovic N, Serafini F,
Serdar N (2023) Integration of RISE innovations in the fields of OELF, RLA and SHM.
RISE Project Deliverable 6.1. Available at
[http://static.seismo.ethz.ch/rise/deliverables/Deliverable_6.1.pdf](http://static.seismo.ethz.ch/rise/deliverables/Deliverable_6.1.pdf).

## Software DOI

Nievas CI, Crowley H, Weatherill G (2023) Real-Time Loss Tools. Zenodo.
[https://doi.org/10.5281/zenodo.7948699](https://doi.org/10.5281/zenodo.7948699)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7948699.svg)](https://doi.org/10.5281/zenodo.7948699)

## Copyright and copyleft

Copyright (C) 2022-2026 Cecilia Nievas: cecilia.nievas@gfz-potsdam.de

This program is free software: you can redistribute it and/or modify it
under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or (at
your option) any later version.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero
General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program. If not, see http://www.gnu.org/licenses/.

Also add information on how to contact you by electronic and paper mail.

If your software can interact with users remotely through a computer
network, you should also make sure that it provides a way for users to
get its source. For example, if your program is a web application, its
interface could display a "Source" link that leads users to an archive
of the code. There are many ways you could offer source, and different
solutions will be better for different programs; see section 13 for the
specific requirements.

See the [LICENSE](./LICENSE) for the full license text.
