# Real Time Loss Tools

Tools for the assessment of earthquake damage and loss in real-time, developed within the
[RISE project](http://rise-eu.org/home/).

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

Software documentation: coming soon.

## Running

### Preparation

A series of input files and a specific file structure are needed to run the
`Real Time Loss Tools`. Please refer to [Assumed file structure](#assumed-file-structure) down
below for details and follow the instructions.

Once the input files and the file structure have been created, create a `config.yml` file
following the example contained in this repository as
[config_example.yml](./config_example.yml). The `config.yml` file can be located anywhere, but
the `Real Time Loss Tools` need to be run from the directory where `config.yml` is located. It
can be placed, for example, within `main_path`.

### Execution

Having activated the virtual environment where the `Real Time Loss Tools` are installed,
navigate to the directory where you placed `config.yml`. Then type:

```bash
(YourPreferredName) $ rtlt
```

The program will start to run.

### Assumed file structure

The configuration file will point at a `main_path` that will be used for running this software.
It is assumed that `main_path` will have the following structure and pre-existing files:

- `triggering.csv`: CSV file that simulates the triggering of the computations by listing a
series of catalogue CSV files (see description of `catalogues` directory) and specifying which
sort of analysis is to be run for each of them, namely `RLA` (Rapid Loss Assessment) or `OELF`
(Operation Earthquake Loss Forecasting). Each row of `triggering.csv` simulates the triggering
of the code to run that particular earthquake or catalogue of earthquakes.
- `catalogues`: Directory with a series of CSV files, each of which contains parameters
associated with one (RLA) or more (OELF) earthquakes. The names of these files need to be listed in
`triggering.csv` (see description above).
- `current`: Directory with files that represent the current status of the running process:
  - `job.ini`: INI file required to run OpenQuake (see OpenQuake's user manual for in-depth
  explanations). It needs to exist before putting the software to run, though the general
  description, time of the day and name of the rupture XML file are updated during the run. The
  rest of the parameters will NOT be updated (i.e. values in the existing `job.ini` will be
  kept) and need to be carefully selected by the user before the run. The parameters needed are
  (see `job_ini_example.ini`):
    - `general:description`: General description to be used in OpenQuake.
    - `general:calculation_mode`: `scenario_damage`.
    - `general:ses_seed`: A positive integer to ensure reproducibility of the results.
    - `exposure:exposure_file`: Relative path to the XML file of the exposure model.
    - `exposure:taxonomy_mapping_csv`: Relative path to the CSV file with the
    exposure-to-vulnerability mapping (if desired).
    - `exposure:time_event`: `day`, `night` or `transit`.
    - `fragility:structural_fragility_file`: Relative path to the XML file of the fragility
    model.
    - `Rupture information:rupture_model_file`: Relative path to the XML file of the rupture.
    - `Rupture information:rupture_mesh_spacing`: Mesh spacing for the discretisation of the
    earthquake rupture, in km. Recommended: 0.5.
    - `Site conditions:site_model_file`: Relative path to the CSV file of the site model.
    - `Calculation parameters:gsim_logic_tree_file`: Relative path to the XML file of the ground
    motion logic tree.
    - `Calculation parameters:truncation_level`: Maximum number of standard deviations of the
    ground motion model to use. Usual value: 3.
    - `Calculation parameters:maximum_distance`: Distance (in km) between source and site above
    which it is assumed that no output of relevance can be produced and thus no calculation is
    attempted for those sites. Usual value: 200.0.
    - `Calculation parameters:number_of_ground_motion_fields`: Number of ground motion fields to
    be generated to capture the aleatory uncertainty in the ground motions. Recommended: 1000.
    - `Calculation parameters:minimum_intensity`: E.g. "{"AvgSA": 1E-5}".
  - `exposure_model.xml`: XML file defining the exposure input for OpenQuake. It needs to exist
  before putting the software to run, though the time of the day of the earthquake will be
  updated during the run. The following parameters will NOT be updated (i.e. values in the
  existing `exposure_model.xml` will be kept): 
    - `exposureModel`: `category`, `id` and `taxonomySource`
    - `description`
    - `costType`: `name`, `type` and `unit`
    - `tagNames`
    - `assets`
  - `exposure_model_current.csv`: CSV file defining the exposure input for OpenQuake. It needs
  to NOT exist before putting the software to run, otherwise it will not run. The software will
  initialise this file by copying `exposure_models/exposure_model_undamaged.csv` (see
  description) at the beginning of the run. This file will be updated each time a rapid loss
  assessment is run.
- `exposure_models`: Directory with CSV files of the exposure model after each earthquake run,
subdivided into `oelf` and `rla`, as well as the original undamaged exposure model, named
`exposure_model_undamaged.csv`, which needs to exist before putting the software to run. This
file is not modified during the run. The `oelf` sub-directory will be further subdivided as per
the different catalogue CSV files (see description of `triggering.csv` and `catalogues` above).
`exposure_model_undamaged.csv` must contain the following fields:
  - `id`: Asset ID, unique identifier per row.
  - `lon`: Longitude of the asset, in degrees.
  - `lat`: Latitude of the asset, in degrees.
  - `taxonomy`: String describing the building class. These strings are sought for in
  `static/fragility_model.xml`.
  - `number`: Number of buildings associated with the asset.
  - `structural`: Total replacement cost of the asset.
  - `census`: Number of occupants in the asset irrespective of the time of the day and the
  damage state of the building.
  - `occupancy`: "residential", "commercial", or "industrial" (needs to match the keys used for
  `time_of_day_occupancy` in `config.yml`).
  - `id_X`, `name_X` (can be several values of X): ID(s) and name(s) of the administrative
  unit(s) to which the asset belongs. "X" is the administrative level.
  - `building_id`: ID of the building. One `building_id `can be associated with different
  values of asset IDs (`id`).
  - `original_asset_id`: ID of the asset in the initial undamaged version of the exposure model.
  It can be the value of `id` in the undamaged version or any other unique ID per row that
  refers to a combination of a building ID and a building class with no initial damage.
- `openquake_output`: Directory to which OpenQuake results will be exported. OELF results will
be saved organised into sub-directories as per the different catalogue CSV files (see
description of `triggering.csv` and `catalogues` above).
- `output`: Directory to which the software will export results. It should be empty when
starting the run.
- `ruptures`: Directory with:
  - XML files of the earthquake ruptures, subdivided into `oelf` and `rla`. No pre-existing XML
  files are needed here (the directories will be populated during the run). The `oelf`
  sub-directory will be further subdivided as per the different catalogue CSV files (see
  description of `triggering.csv` and `catalogues` above).
  - An XML with the earthquake source model to be used to stochastically generate rupture
  properties for Operational Earthquake Loss Forecasting (OELF), in the OpenQuake format. The
  name of this XML file can be arbitrary and must be indicated in the configuration file.
- `shm`: Directory with a CSV file with damage classification determined by means of Structural
Health Monitoring (SHM) techniques, named `damage_results_shm.csv`. If RLA is indicated in
`triggering.csv`, the file needs to exist for the software to run. It needs to contain the
following fields:
  - `building_id`: ID of the building, as in `exposure_model_undamaged.csv`.
  - `dmg_state`: Name of the damage state as output by OpenQuake.
  - columns whose names are the IDs of individual earthquakes referenced to in `catalogues`:
  These fields contain the probability of each `building_id` resulting in a specific damage
  state (`dmg_state`) due to the action of the earthquake with the indicated event ID. All rows
  associated with a particular `building_id` and event ID must add up to unity.
- `static`: Directory with input files that need to be defined by the user before the run and
remain the same all throughout. These files are:
  - `fragility_model.xml`: XML file of the fragility model, in OpenQuake input format.
  - `exposure_vulnerability_mapping.csv`: CSV file with the exposure-to-vulnerability mapping,
  if desired and specified in the `job.ini` file that it needs to be used.
  - `gmpe_logic_tree.xml`: XML file of the ground motion logic tree, in OpenQuake input format.
  - `site_model.csv`: CSV file of the site model, in OpenQuake input format.
  - `consequences_economic.csv`: CSV file with economic loss ratios (as percentages) per
  building class (row) and damage state (including "DS0", i.e. the no damage case).
  - `consequences_injuries_severity_X.csv`: CSV file with human loss ratios (as percentages) per
  building class (row) and damage state (including "DS0", i.e. the no damage case), for every
  injury severity level `X` listed in the `config.yml` file under `injuries_scale`.
  - `recovery_damage.csv`: CSV file with the number of days that it takes to inspect (column
  `N_inspection` and to repair or replace (column `N_repair`) a building in each damage state
  (column `dmg_state`). Each row corresponds to a damage state.
  - `recovery_injuries.csv`: CSV file with the number of days that it takes for a person with
  different severities of injury (column `injuries_scale`) to be able to return to their
  building/s due only to their health condition (column `N_discharged`). If the injuries scale
  covers death, use a very large number for `N_discharged` under this category.

Before running the software, the user needs to set up the structure under `main_path` as
follows:
- `triggering.csv` needs to exist (see description above). This file is not modified during the
run.
- `catalogues` needs to exist and contain the CSV files described above. These files are not
modified during the run.
- `current` needs to contain `job.ini` and `exposure_model.xml`, while
`exposure_model_current.csv` must NOT exist in the directory.
- `exposure_models` must contain the `oelf` and `rla` sub-directories, each of which must be
empty, as well as `exposure_model_undamaged.csv`, whose existence is a requirement for the
software to run.
- `openquake_output` needs to exist and be empty.
- `output` needs to exist and be empty.
- `ruptures` must contain the `oelf` and `rla` sub-directories, each of which must be empty, and
an XML with the earthquake source model in the OpenQuake format (see description above).
- `shm` must contain `damage_results_shm.csv`, only if RLA is indicated at least in one case
under `triggering.csv`.
- `static` needs to contain `fragility_model.xml`, `gmpe_logic_tree.xml`, `site_model.csv`,
`consequences_economic.csv`, `consequences_injuries_severity_X.csv` (for every injury
severity level `X` listed in the `config.yml` file under `injuries_scale`),
`recovery_damage.csv`, and `recovery_injuries.csv`. 

### Effect of `store_intermediate` on output stored

The section [Assumed file structure](#assumed-file-structure) above describes the file structure
needed to run this software and the kinds of outputs that are generated. However, the user can
control whether intermediate results are stored or not by setting the option
`store_intermediate` in the `config.yml` file to True or False, respectively. The effect of
`store_intermediate` on the files that are stored is shown in the following table, which
specifies if each kind of file is stored ("yes") or not ("no"). Only directories that are used
for storing output are listed:

| Path                 | File                                                   | `store_intermediate` = True | `store_intermediate` = False |
|----------------------|--------------------------------------------------------|-----------------------------|------------------------------|
| current              | exposure_model_current.csv                             | yes                         | yes                          |
| exposure_models/oelf | exposure_model_current_oelf_SES.csv                    | yes                         | yes                          |
| exposure_models/oelf | exposure_model_after_SES_EQ.csv                        | yes                         | no                           |
| exposure_models/rla  | exposure_model_after_EQ.csv                            | yes                         | no                           |
| openquake_output     | (SES)\_EQ_damages_OQ.csv                               | yes                         | no                           |
| openquake_output     | (SES)\_EQ_damages_OQ_raw.csv                           | yes                         | no                           |
| oputput              | damage_states_after_RLA_trig_name.csv                  | yes                         | yes                          |
| oputput              | damage_states_after_OELF_trig_name.csv                 | yes                         | yes                          |
| oputput              | damage_states_after_OELF_trig_name_realisation_SES.csv | yes                         | no                           |
| ruptures             | rupture_EQ.xml                                         | yes                         | yes                          |
| ruptures             | RUP_SES-EQ.xml                                         | yes                         | yes                          |

Notes on nomenclature:

1. `SES` refers to "stochastic event set", that is, each individual realisation of seismicity
for a given period of time in the seismicity forecast. As part of a file name, it represents the
ID of the SES.
2. Parentheses `( )` imply that there are two cases, one in which the file name contains what
is between the parentheses and one in which it does not.
3. `EQ` refers to the earthquake ID.
4. `trig_name` refers to each of the CSV files contained in `catalogues` and listed in the
`triggering.csv` file.

### Log for quick checks

The code creates a file named `quick_input_check.txt` under `main_path`. The purpose of this
file is to allow the user to quickly check certain parameters of the run. This is useful, for
example, when running the `Real Time Loss Tools` in a large batch in which different jobs
correspond to different combinations of input files.

Example of `quick_input_check.txt`:

```
LOG FILE
Real-Time Loss Tools has started
General description: Run 03 a
Running with state-dependent fragility models
/my/local/path/run_03_a is path in config file
/my/local/path/run_03_a is current path
State dependent: True
First filename in triggering.csv is 'EQ_01.csv'
With update of occupants in 'recovery_damage'
With update of occupants in 'recovery_injuries'
Real-Time Loss Tools has finished
```

The contents of `quick_input_check.txt` should be interpreted as follows (referring to the
example above):

- `General description: ...`: The `description_general` entered by the user in the
`config.yml` file.
- `Running with state-dependent fragility models` or `Running with state-independent fragility
models`: The `state-dependent` message is shown when `state_dependent_fragilities` in the
`config.yml` file is set to True by the user, while the `state-independent` counterpart is shown
when `state_dependent_fragilities` in the `config.yml` file is set to False. The code does not
actually check if the fragility model inside `fragility_model.xml` is state-dependent or state-
independent.
- `/my/local/path/run_03_a is path in config file`: The `main_path` entered by the user in the
`config.yml` file.
- `/my/local/path/run_03_a is current path`: Path from which the job is being run. This path
should match the `path in config file`; if not, the file structure should be checked.
- `State dependent: True/False`: It is True if the string "state_dependent" exists in the file
`main_path/current/job.ini` and False otherwise. The code does not actually check if the
fragility model inside `fragility_model.xml` is state-dependent or state-independent. This check
assumes that, when running an analysis with a state-dependent fragility model, the string
"state-dependent" exists in `main_path/current/job.ini`.
- `First filename in triggering.csv is 'EQ_01.csv'`: The first file name in the `triggering.csv`
file. This is useful if the file names contain a reference to the overall case being run within
a batch job, to make sure the correct files have been passed as input.
- `With update of occupants in 'recovery_damage'`: The code adds up the number of days input by
the user in `recovery_damage.csv`. If the sum is larger than zero, the user sees the message
`With update of occupants in 'recovery_damage'`, while if the sum is zero, the user sees the
message `No update of occupants in 'recovery_damage'`. When all zeros are input to
`recovery_damage.csv` and `recovery_injuries.csv`, the code does not carry out the updating of
occupants as per previous damage/human losses.
- `With update of occupants in 'recovery_injuries'`: The code adds up the number of days input by
the user in `recovery_injuries.csv`. See explanation above.

## Acknowledgements

These tools have been developed within the [RISE project](http://rise-eu.org/home/), which has
received funding from the European Union's Horizon 2020 research and innovation programme under
grant agreement No. 821115.

## Citation

Please cite as:

Nievas CI, Crowley H, Reuland Y, Weatherill G, Baltzopoulos G, Bayliss K, Chatzi E, Chioccarelli
E, Guéguen P, Iervolino I, Marzocchi W, Naylor M, Orlacchio M, Pejovic J, Popovic N, Serafini F,
Serdar N (2023) Integration of RISE innovations in the fields of OELF, RLA and SHM.
RISE Project Deliverable 6.1.

## Copyright and copyleft

Copyright (C) 2022 Cecilia Nievas: cecilia.nievas@gfz-potsdam.de

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
