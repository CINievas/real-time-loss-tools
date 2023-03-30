# Real Time Loss Tools

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

It is recommended that a Python 3.8 virtual environment be created so as to install and run the
software from within it. To do so, type:

```bash
$ python3.8 -m venv YourPreferredName
```

`YourPreferredName` will be the name of the virtual environment. Activate it by doing:

```bash
$ source YourPreferredName/bin/activate
```

Before doing anything else, upgrade `pip` to its latest version (not doing so might result in
errors during installation that will not indicate that the problem lies in the version of
`pip`):

```bash
(YourPreferredName) $ pip install --upgrade pip
```

Move within your directory structure to a location where you would like to clone the present
repository. From there, do:

```bash
(YourPreferredName) $ git clone https://git.gfz-potsdam.de/cnievas/real-time-loss-tools.git
(YourPreferredName) $ cd real-time-loss-tools
(YourPreferredName) $ pip3 install -e .
```

The last command will install the `Real Time Loss Tools` and all its dependencies, including the
[OpenQuake engine](https://github.com/gem/oq-engine). If you already have a version of OpenQuake
installed in your computer, even if it is within a virtual environment, the different versions
or installations will share the same underlying database and it might be necessary to stop and
re-start it to be able to run the `Real Time Loss Tools`. The error that might flag up if this
is the case might not be fully self-explanatory, and so it is recommended to stop and re-start
the database as a precautionary measure by doing:

```bash
(YourPreferredName) $ oq dbserver stop
(YourPreferredName) $ oq dbserver start
```

Please note that mixing different versions of OpenQuake can lead to the database becoming
unusable with older versions.

The virtual environment can be deactivated by typing:

```bash
(YourPreferredName) $ deactivate
```

### Software dependencies

- Python 3.8 or above (tested only with Python 3.8 and 3.9)

### Python libraries

- `pyyaml`
- `numpy`
- `pandas`
- `shapely`
- `pyproj`
- `rtree`
- `geopandas`
- `pytz`
- `openquake.engine 3.15`

## Documentation

For scientific documentation, please see [Citation](#citation) down below.

Software documentation: click [here](docs/README.md).

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

Please cite as:

Nievas CI, Crowley H, Reuland Y, Weatherill G, Baltzopoulos G, Bayliss K, Chatzi E, Chioccarelli
E, Guéguen P, Iervolino I, Marzocchi W, Naylor M, Orlacchio M, Pejovic J, Popovic N, Serafini F,
Serdar N (2023) Integration of RISE innovations in the fields of OELF, RLA and SHM.
RISE Project Deliverable 6.1. Available at
[http://static.seismo.ethz.ch/rise/deliverables/Deliverable_6.1.pdf](http://static.seismo.ethz.ch/rise/deliverables/Deliverable_6.1.pdf).

## Copyright and copyleft

Copyright (C) 2022-2023 Cecilia Nievas: cecilia.nievas@gfz-potsdam.de

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
