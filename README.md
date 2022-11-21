# Real Time Loss Tools

Tools for the assessment of earthquake damage and loss in real-time, developed within the
[RISE project](http://rise-eu.org/home/).

## Installation

```bash
git clone https://git.gfz-potsdam.de/cnievas/real-time-loss-tools.git
cd real-time-loss-tools
pip3 install -e .
```

### Software dependencies

- Python 3.8 or above

### Python libraries

- `pyyaml`
- `numpy`
- `pandas`
- `shapely`
- `pyproj`
- `rtree`
- `geopandas`
- `openquake.engine 3.15`

## Running

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
It must contain the following fields:
  - `id`: Asset ID, unique identifier per row.
  - `lon`: Longitude of the asset, in degrees.
  - `lat`: Latitude of the asset, in degrees.
  - `taxonomy`: String describing the building class. These strings are sought for in
  `static/fragility_model.xml`.
  - `number`: Number of buildings associated with the asset.
  - `structural`: Total replacement cost of the asset.
  - `night`: Number of occupants in the asset during the night time.
  - `day`: Number of occupants in the asset during the day time.
  - `transit`: Number of occupants in the asset during the transit time.
  - `occupancy`: "Res", "Com", or "Ind"
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
- `static`: Directory with input files for OpenQuake that need to be defined by the user before
the run and remain the same all throughout. These files are:
  - `fragility_model.xml`: XML file of the fragility model, in OpenQuake input format.
  - `gmpe_logic_tree.xml`: XML file of the ground motion logic tree, in OpenQuake input format.
  - `site_model.csv`: CSV file of the site model, in OpenQuake input format.

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
- `static` needs to contain `fragility_model.xml`, `gmpe_logic_tree.xml`, and `site_model.csv`.

## Acknowledgements

These tools have been developed within the [RISE project](http://rise-eu.org/home/), which has
received funding from the European Union's Horizon 2020 research and innovation programme under
grant agreement No. 821115.

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
