# Input files

Full sets of example input files can be found
[here](https://git.gfz-potsdam.de/real-time-loss-tools/rise-d6-1-data-files).

## Assumed file structure

The [configuration file](02_Configuration.md#configuration-file) points at a `main_path` that
is used for running this software. It is assumed that `main_path` will have the following
structure and pre-existing files:

- [config.yml](02_Configuration.md#configuration-file)

- [triggering.csv](02_Configuration.md#triggering-file): CSV file that simulates the triggering
of the computations by listing a series of catalogue CSV files (see description of `catalogues`
directory) and specifying which sort of analysis is to be run for each of them, namely `RLA`
(Rapid Loss Assessment) or `OELF` (Operation Earthquake Loss Forecasting). Each row of
`triggering.csv` simulates the triggering of the code to run that particular earthquake or
catalogue of earthquakes. Additionally, it may indicate which rupture XML file(s) to use for one
or more of the RLA triggers.

- [catalogues](#earthquake-catalogues): Directory with a series of CSV files, each of which
contains parameters associated with one (RLA) or more (OELF) earthquakes. The names of these
files need to be listed in `triggering.csv`.

- `current`: Directory with files that represent the current status of the running process:

  - [job.ini](02_Configuration.md#jobini-for-openquake): file required to run OpenQuake (see
  OpenQuake's user manual for in-depth explanations). It needs to exist before putting the
  software to run, though the general description, time of the day and name of the rupture XML
  file are updated during the run. The rest of the parameters will NOT be updated (i.e. values
  in the existing `job.ini` will be kept) and need to be carefully selected by the user before
  the run. The needed parameters are explained [here](02_Configuration.md#jobini-for-openquake).

  - [exposure_model.xml](#exposure-model): XML file defining the exposure input for OpenQuake.
  It needs to exist before putting the software to run. 
  
  - `exposure_model_current.csv`: CSV file defining the exposure input for OpenQuake. It needs
  to **NOT** exist before putting the software to run, otherwise it will not run. The software
  will initialise this file by copying `exposure_models/exposure_model_undamaged.csv` (see
  description) at the beginning of the run. This file will be updated each time a rapid loss
  assessment is run.

- `exposure_models`: Directory with CSV files of the exposure model after each earthquake run,
subdivided into `oelf` and `rla`, as well as the original undamaged exposure model, named
`exposure_model_undamaged.csv`, which needs to exist before putting the software to run. This
file is not modified during the run. The `oelf` sub-directory will be further subdivided as per
the different catalogue CSV files (see description of `triggering.csv` and `catalogues` above).
See details on `exposure_model_undamaged.csv` [here](#exposure-model).

- `openquake_output`: Directory to which OpenQuake results will be exported. OELF results will
be saved organised into sub-directories as per the different catalogue CSV files (see
description of `triggering.csv` and `catalogues` above).

- `output`: Directory to which the software will export results. It should be empty when
starting the run.

- `ruptures`: Directory with:

  - XML files of the earthquake ruptures, subdivided into `oelf` and `rla`. No pre-existing XML
  files are needed under the `oelf` sub-directory (the directory will be populated during the
  run). The `oelf` sub-directory will be further subdivided as per the different catalogue CSV
  files (see description of `triggering.csv` and `catalogues` above).
  
  - Rupture XML files in the OpenQuake format under the `ruptures/rla`, if desired and as
  specified in the [triggering.csv](02_Configuration.md#triggering-file).
  
  - A CSV file with rupture parameters to be used for RLA, under
  `ruptures/rla/`[source_parameters.csv](#rupture-parameters-for-rla), if desired and as
  specified in the [triggering.csv](02_Configuration.md#triggering-file).
  
  - An XML with the earthquake source model to be used to stochastically generate rupture
  properties for Operational Earthquake Loss Forecasting (OELF), in the
  [OpenQuake format](https://docs.openquake.org/oq-engine/master/manual/hazard.html#source-typologies).
  The name of this XML file can be arbitrary and must be indicated in the configuration file
  under [oelf_source_model_filename](02_Configuration.md#oelf_source_model_filename).

- `shm`: Directory with a CSV file with damage classification determined by means of Structural
Health Monitoring (SHM) techniques, named
[damage_results_shm.csv](#damage-states-from-structural-health-monitoring). If RLA is indicated
in `triggering.csv`, the file needs to exist for the software to run.

- `static`: Directory with input files that need to be defined by the user before the run and
remain the same all throughout. These files are:

  - [fragility_model.xml](#fragility-models): XML file of the fragility model, in OpenQuake
  input format.
  
  - [exposure_vulnerability_mapping.csv](#fragility-models): CSV file with the
  exposure-to-vulnerability mapping, if desired and specified in the `job.ini` file that it
  needs to be used.
  
  - [gmpe_logic_tree.xml](#ground-motion-models): XML file of the ground motion logic tree, in
  [OpenQuake input format](https://docs.openquake.org/oq-engine/master/manual/hazard.html#defining-logic-trees).
  
  - [site_model.csv](#ground-motion-models): CSV file of the site model, in OpenQuake input
  format.
  
  - [consequences_economic.csv](#economic-and-human-consequence-models): CSV file with economic
  loss ratios (as percentages) per building class (row) and damage state (including the no
  damage case).
  
  - [consequences_injuries_severity_X.csv](#economic-and-human-consequence-models): CSV file
  with human loss ratios (as percentages) per building class (row) and damage state (including
  the no damage case), for every injury severity level `X` listed in the `config.yml` file under
  [injuries_scale](02_Configuration.md#injuries_scale).
  
  - [recovery_damage.csv](#timelines-for-damage-inspection-and-hospitalisations): CSV file with
  the number of days that it takes to inspect (column `N_inspection` and to repair or replace
  (column `N_repair`) a building in each damage state (column `dmg_state`). Each row corresponds
  to a damage state.
  
  - [recovery_injuries.csv](#timelines-for-damage-inspection-and-hospitalisations): CSV file
  with the number of days that it takes for a person with different severities of injury (column
  `injuries_scale`) to be able to return to their building/s due only to their health condition
  (column `N_discharged`). If the injuries scale covers death, use a very large number for
  `N_discharged` under this category.

Before running the software, the user needs to set up the structure under `main_path` as
follows:

- `triggering.csv` needs to exist (see description above). This file is not modified during the
run.

- `catalogues` needs to exist and contain the CSV files described above. These files are not
modified during the run.

- `current` needs to contain `job.ini` and `exposure_model.xml`, while
`exposure_model_current.csv` must **NOT** exist in the directory.

- `exposure_models` must contain the `oelf` and `rla` sub-directories, each of which must be
empty, as well as `exposure_model_undamaged.csv`, whose existence is a requirement for the
software to run.

- `openquake_output` needs to exist and be empty.

- `output` needs to exist and be empty.

- `ruptures` must contain the `oelf` and `rla` sub-directories, and an XML with the earthquake
source model in the OpenQuake format (see description above). The sub-directory `oelf` must be
empty, while `rla` must contain a `source_parameters.csv` file.

- `shm` must contain `damage_results_shm.csv`, only if RLA is indicated at least in one case
under `triggering.csv`.

- `static` needs to contain `fragility_model.xml`, `gmpe_logic_tree.xml`, `site_model.csv`,
`consequences_economic.csv`, `consequences_injuries_severity_X.csv` (for every injury
severity level `X` listed in the `config.yml` file under `injuries_scale`),
`recovery_damage.csv`, and `recovery_injuries.csv`. 

## Earthquake Catalogues

One CSV file must exist in the `catalogues` directory for each RLA and OELF calculation listed
in `triggering.csv`.

In the case of RLAs, the catalogue CSV files must contain the following fields:

- `longitude`: longitude of the hypocentre;
- `latitude`: latitude of the hypocentre;
- `magnitude`: moment magnitude;
- `datetime`: UTC date and time of occurrence, in standard ISO 8601 format (i.e.,
YYYY-MM-DDTHH:MM:SS);
- `depth`: hypocentral depth, in km;
- `catalog_id`: an identifier for internal purposes (if desired);
- `event_id`: identifier of the event, to be used to retrieve finite-fault rupture parameters
from `ruptures/rla/`[source_parameters.csv](#rupture-parameters-for-rla)

The current version of the Real-Time Loss Tools can only generate simple planar ruptures for
normal faults with parameters specified as in the Italian Accelerometric Archive (ITACA)
[website](https://itaca.mi.ingv.it) (Russo et al., 2022). It will search in
`ruptures/rla/`[source_parameters.csv](#rupture-parameters-for-rla) for the `event_id` specified
in the catalogue file.

In the case of OELFs, the catalogue CSV files must contain the following fields:

- `longitude` or `Lon`: longitude of the hypocentre;
- `latitude` or `Lat`: latitude of the hypocentre;
- `magnitude` or `Mag`: moment magnitude;
- `datetime` or `Time`: UTC date and time of occurrence, in standard ISO 8601 format (i.e.,
YYYY-MM-DDTHH:MM:SS);
- `catalog_id` or `Idx.cat`: ID of the realisation of seismicity (stochastic event set) that
this earthquake belongs to. Please note the relevance of parameters `continuous_ses_numbering`
and `ses_range` of the [config.yml](02_Configuration.md#configuration-file) on how this field is
interpreted.

Optional fields for the OELF catalogue files are:

- `depth`: hypocentral depth, in km;
- `event_id`: unique identifier of the earthquake within a stochastic event set.

When no `depth` is specified, the depth is sampled by the stochastic rupture generator when
building each earthquake rupture.

When no `event_id` is specified, the Real-Time Loss Tools automatically generates one.

## Rupture Parameters for RLA

For each RLA trigger, the user can either provide a rupture XML file and indicate it in the
[triggering.csv](02_Configuration.md#triggering-file) file, or provide a series of parameters
by means of the `source_parameters.csv` file, which must exist under `main_path/ruptures/rla/`.
In the latter case, the current version of the Real-Time Loss Tools can only generate ruptures
for normal faults in the form of simple planar ruptures.

Following the nomenclature used by the Italian Accelerometric Archive
([ITACA](https://itaca.mi.ingv.it); Russo et al., 2022), `source_parameters.csv` must contain
the following fields (one row for each of the RLA earthquakes named in `triggering.csv` for
which no rupture XML files are indicated/provided):

- `event_id`: identifier of the earthquake, as in the corresponding catalogue CSV file.
- `Mw`: moment magnitude;
- `nucleation_lon`, `nucleation_lat`, `nucleation_depth`: longitude, latitude and depth (in km)
of the hypocentre;
- `LL_lon`, `LL_lat`: longitude and latitude of the lower-left corner of the rupture plane;
- `UR_lon`, `UR_lat`: longitude and latitude of the upper-right corner of the rupture plane;
- `LR_lon`, `LR_lat`: longitude and latitude of the lower-right corner of the rupture plane;
- `UL_lon`, `UL_lat`: longitude and latitude of the upper-left corner of the rupture plane;
- `Z_top`: depth to the top of the rupture (in km);
- `Strike`: strike of the rupture plane;
- `Dip`: dip of the rupture plane;
- `Rake`: rake of the rupture plane.

Once these parameters are retrieved, the Real-Time Loss Tools create the associated rupture XML
file and place it under `main_path/ruptures/rla/`.

The connection between each RLA trigger and a set of parameters in the `source_parameters.csv`
file is provided by the `event_id`, which is indicated in the corresponding catalogue file of
the trigger.

For any RLA trigger, if a rupture XML file is indicated in the `triggering.csv` file, it will be
used and any parameters that may exist in the `source_parameters.csv` file for this earthquake
will be ignored. Note that the sole existence of a rupture XML file under `ruptures/rla` does
not imply that it will be used: its name must be listed in the `rupture_xml` column of the
`triggering.csv` file.

Any provided rupture XML files must follow the OpenQuake format.

## Exposure Model

The initial exposure model is input by means of two files that follow the OpenQuake format:

- `exposure_model_undamaged.csv` (within the `exposure_models` directory)
- `exposure_model.xml` (within the `current` directory)

While OpenQuake can take as input several CSV files, as long as they are all named in the XML
file, the Real-Time Loss Tools are designed to work with just one exposure CSV file.

Each row of `exposure_model_undamaged.csv` corresponds to a building or set of buildings
associated with a particular location and building class (as in any model to be used with
OpenQuake). Apart from the standard fields needed for OpenQuake, the Real-Time Loss Tools
use the concept of a `building_id` and `original_asset_id`, both of which need to be defined.
A `building_id` is associated with the resolution of the exposure model, and can refer to an
individual building or an aggregated group of buildings. An `original_asset_id` corresponds to
a specific combination of a `building_id` and a building class (i.e., type of structure). The
intended use of these concepts is the following:

- If a `building_id` refers to an individual building, each `original_asset_id` refers to a
building class with a specific probability of being that of the building (specified in the field
`number`, see below).

- If the `building_id` refers to an aggregated group of buildings, each `original_asset_id`
refers to a building class with an associated number of buildings (specified in the field
`number` as well).

Once a RLA is run and buildings get assigned damage states, each specific combination of
`original_asset_id` and a damage state becomes (internally) an `asset_id`. Each `asset_id`
occupies one row of the exposure CSV file and corresponds to what OpenQuake usually calls simply
an "asset" with a specific `id` (see below).

The columns required for the exposure CSV file are the following:

- `id`: unique identifier to be used by OpenQuake (OpenQuake requires the label “id”; the
Real-Time Loss Tools treats this field internally as the `asset_id`);
- `lon`, `lat`: longitude and latitude of the asset (e.g. centroid of an individual building or
an aggregated geometry);
- `taxonomy`: building class, specifying the damage state as the last parameter, using a slash
to concatenate it to the rest of the building class string (e.g. `class_A/DS0`, `class_A/DS1`);
- `number`: number of buildings of this `original_asset_id` (i.e., of this `building_id` and
building class), or probability of this building class corresponding to this
`original_asset_id`;
- `structural`: total replacement cost of all buildings (or fractions of buildings) specified in
`number`;
- `census`: number of occupants in all buildings (or fractions of buildings) specified in
`number`, irrespective of the time of the day;
- `occupancy`: occupancy associated with certain parameters to assign the number of occupants at
different times of the day (the names need to coincide with those specified under
[time_of_day_occupancy](02_Configuration.md#time_of_day_occupancy) in the `config.yml` file;
- `building_id`: unique identifier for this individual building or aggregation unit;
- `original_asset_id`: unique identifier for this combination of `building_id` and building
class.

The `building_id` is used by the Real-Time Loss Tools to output final results to the
`main_path/output` folder. Intermediate OpenQuake results are output in terms of `asset_id` to
the `main_path/openquake_output` folder. It is noted that `building_id` and `original_asset_id`
consistently refer to the same building/s and building classes along the whole run, but the
values of `asset_id` may change.

The `exposure_model.xml` file follows the
[OpenQuake format](https://docs.openquake.org/oq-engine/master/manual/risk.html#exposure-models).
Example:

```
<?xml version="1.0" encoding="UTF-8"?>
<nrml xmlns="http://openquake.org/xmlns/nrml/0.4" xmlns:gml="http://www.opengis.net/gml">
  <exposureModel category="buildings" id="exposure" taxonomySource="GEM taxonomy">
    <description>exposure model</description>
    <conversions>
      <costTypes>
        <costType name="structural" type="aggregated" unit="EUR"/>
      </costTypes>
    </conversions>
    <occupancyPeriods>time</occupancyPeriods>
    <tagNames>occupancy id_3 name_3 building_id</tagNames>
    <assets>exposure_model_current.csv</assets>
  </exposureModel>
</nrml>
```

Some notes of relevance on the `exposure_model.xml` file:

- The time of the day of the earthquake (under `occupancyPeriods`) does not need to be specified
by the user, as the Real-Time Loss Tools update this line for running each earthquake.

- The Real-Time Loss Tools assume that replacement costs and census occupants are input for the
total number of buildings (or fractions of buildings) in an asset (row of the CSV file). For
this reason, `type="aggregated"` must be specified in the `exposure_model.xml` file.

- The name of the CSV file must be `exposure_model_current.csv`.

- The parameters `exposureModel`, `category`, `id`, `taxonomySource`, `description`, `costType`,
`name`, `type`, `unit`, `tagNames` and `assets` are **not** updated during the run.

## Damage States from Structural Health Monitoring

The Real-Time Loss Tools incorporate to the RLAs damage assessments carried out by means of
Structural Health Monitoring (SHM) techniques, or any other external source. If RLA is indicated
in `triggering.csv`, the file `damage_results_shm.csv` needs to exist for the software to run.
It needs to contain the following fields:

  - `building_id`: ID of the building, as in [exposure_model_undamaged.csv](#exposure-model).
  - `dmg_state`: Name of the damage state as output by OpenQuake.
  - columns whose names are the IDs of individual earthquakes referenced to in `catalogues`:
  These fields contain the probability of each `building_id` resulting in a specific damage
  state (`dmg_state`) due to the action of the earthquake with the indicated event ID. All rows
  associated with a particular `building_id` and event ID must add up to unity.
  
Example:

```
building_id,dmg_state,EQ_01,EQ_02
building_01,no_damage,0.40,0.27
building_01,dmg_1,0.30,0.20
building_01,dmg_2,0.20,0.40
building_01,dmg_3,0.08,0.10
building_01,dmg_4,0.02,0.03
```

If no SHM-based damage results are available, the `damage_results_shm.csv` must contain one line
with the heading, including the IDs of the earthquakes, and nothing else:

```
building_id,dmg_state,EQ_01,EQ_02
```

Damage results specified in `damage_results_shm.csv` take prevalence over those calculated with
ground motion-based fragility models.

## Fragility Models

The fragility models need to be input in the
[OpenQuake format](https://docs.openquake.org/oq-engine/master/manual/risk.html#fragility-models),
as a single XML file (`fragility_model.xml`) under `main_path/static`. OpenQuake supports both
continuous and discrete fragility models.

When working with state-independent fragility models, one entry in the XML file per building
class (i.e.,unique string defining the type of structure) is sufficient to run OpenQuake.
However, as the Real-Time Loss Tools keep track of damage by concatenating the damage state to
the building class, one entry per building class and initial damage state is required in the XML
file instead.

State-dependent fragility models are only defined for damage states more severe than the initial
damage state, but OpenQuake requires that all damage states be defined for each entry. By
definition, the probability of exceedance of the initial damage state is 1, irrespective of the
seismic intensity measure, and so is the probability of exceedance of any damage state that is
less severe than the initial one (they have all been exceeded already). This is very simple to
input when working with discrete fragility models, but it raises the need to simulate this
behaviour with a continuous curve, when working with continuous fragility models instead. In
tested applications a very small value of mean and standard deviation (1E-10) used with a
lognormal CDF function have yielded good results (please note that the selection of appropriate
parameters is the responsibility of the user and these values are to be interpreted only as
suggestions).

Special attention should be paid to OpenQuake parameters that specify lower bounds of ground
motions below which the buildings are assumed to be undamaged. These are `minIML` and
`noDamageLimit`, within the fragility XML file, and `minimum_intensity`, within the `job.ini`
configuration file (see [here](02_Configuration.md#calculation-parameters)). These parameters
are very useful for state-independent fragility models, but are tricky when working with
state-dependent fragility models instead. If, for example, `minIML` and `noDamageLimit` are set
to 0.01 g, OpenQuake assumes that ground motions below 0.01 g lead to that building class being
undamaged, which is not the same as not causing any _additional_ damage. Values of `minIML` and
`noDamageLimit` of 1E-15 have been successfully used in applications (please note that the
selection of appropriate parameters is the responsibility of the user and these values are to be
interpreted only as suggestions).

The use of an exposure-vulnerability mapping CSV file (`exposure_vulnerability_mapping.csv`) is
optional when running the Real-Time Loss Tools using state-dependent fragility models. If
desired, its name and location needs to be specified in the `job.ini` file (under
[taxonomy_mapping_csv](02_Configuration.md#jobini-for-openquake)). This file links building
classes as defined in the exposure model to building classes as defined in the fragility model.
If not used, strings of building classes need to be the same in the exposure and fragility
models. Otherwise, any relevant mapping can be used. Please note that this mapping needs to
include the damage state. Trivial mapping files such as in the example below can be used:

```
taxonomy,conversion,weight
CR/LFINF+CDL+LFC:10.0/H:1/DS0,CR/LFINF+CDL+LFC:10.0/H:1/DS0,1.0
CR/LFINF+CDL+LFC:10.0/H:1/DS1,CR/LFINF+CDL+LFC:10.0/H:1/DS1,1.0
CR/LFINF+CDL+LFC:10.0/H:1/DS2,CR/LFINF+CDL+LFC:10.0/H:1/DS2,1.0
CR/LFINF+CDL+LFC:10.0/H:1/DS3,CR/LFINF+CDL+LFC:10.0/H:1/DS3,1.0
CR/LFINF+CDL+LFC:10.0/H:1/DS4,CR/LFINF+CDL+LFC:10.0/H:1/DS4,1.0
CR/LFINF+CDL+LFC:10.0/H:2/DS0,CR/LFINF+CDL+LFC:10.0/H:2/DS0,1.0
CR/LFINF+CDL+LFC:10.0/H:2/DS1,CR/LFINF+CDL+LFC:10.0/H:2/DS1,1.0
...
```

When running the Real-Time Loss Tools using state-independent fragility models, the use of an
exposure-vulnerability mapping CSV file is fundamental to indicate to OpenQuake to keep on using
the same state-independent fragility model irrespective oft he damage state of the buildings in
the exposure model. This is the case because the Real-Time Loss Tools still update the exposure
model and keep track of damage states, even if state-independent fragility models are input.
Following the example above, the exposure-vulnerability mapping that would make the program
run using state-independent fragilities (without changing the input `fragility_model.xml` file)
would be:

```
taxonomy,conversion,weight
CR/LFINF+CDL+LFC:10.0/H:1/DS0,CR/LFINF+CDL+LFC:10.0/H:1/DS0,1.0
CR/LFINF+CDL+LFC:10.0/H:1/DS1,CR/LFINF+CDL+LFC:10.0/H:1/DS0,1.0
CR/LFINF+CDL+LFC:10.0/H:1/DS2,CR/LFINF+CDL+LFC:10.0/H:1/DS0,1.0
CR/LFINF+CDL+LFC:10.0/H:1/DS3,CR/LFINF+CDL+LFC:10.0/H:1/DS0,1.0
CR/LFINF+CDL+LFC:10.0/H:1/DS4,CR/LFINF+CDL+LFC:10.0/H:1/DS0,1.0
CR/LFINF+CDL+LFC:10.0/H:2/DS0,CR/LFINF+CDL+LFC:10.0/H:2/DS0,1.0
CR/LFINF+CDL+LFC:10.0/H:2/DS1,CR/LFINF+CDL+LFC:10.0/H:2/DS0,1.0
...
```

Alternatively, the input `fragility_model.xml` file can be changed so that the definition of
curves conditioned on different initial damage states all correspond to the same initial
undamaged condition.

## Ground Motion Models

The ground motion logic tree is input as an XML file (`gmpe_logic_tree.xml`, placed under
`main_path/static`) following the
[OpenQuake format](https://docs.openquake.org/oq-engine/master/manual/hazard.html#defining-logic-trees).

The site model is input as a CSV file (`site_model.csv`, placed under `main_path/static`)
following the OpenQuake format as well. The parameters required might vary depending on the
ground motion models specified in the ground motion logic tree.

## Economic and Human Consequence Models

The `consequences_economic.csv` file (placed under `main_path/static`) must contain loss ratios
(i.e., ratio of cost of repair associated with a specific damage state to the replacement cost
of the whole building) as percentages for each building class (row) and damage state (column).
The no damage case must be included.

Example (using economic loss ratios and damage state definitions from the European Seismic Risk
Model 2020; ESRM20, Crowley et al., 2021):

```
Taxonomy,DS0,DS1,DS2,DS3,DS4
CR/LFINF+CDN+LFC:0.0/H:1,0,5,15,60,100
CR/LFINF+CDN+LFC:0.0/H:2,0,5,15,60,100
...
```

The `consequences_injuries_severity_X.csv` (placed under `main_path/static`) file must contain
the ratios (as percentages) of people injured with severity level `X` to the total number of
occupants of a building at the time of the earthquake, per building class (row) and damage state
(column) (the file structure is the same as for `consequences_economic.csv`). The no damage case
must be included. One file is needed for each injury severity level `X` listed in the
`config.yml` file under [injuries_scale](02_Configuration.md#injuries_scale).

## Timelines for Damage Inspection and Hospitalisations

The `recovery_damage.csv` file (placed under `main_path/static`) must contain three columns. The
first one, labelled `dmg_state` must indicate each damage state, while the other two specify the
(mean) number of days needed to inspect (column `N_inspection`) and to repair or replace (column
`N_repair`) a building in each damage state.

Example (from Deliverable 6.1 of the RISE project,
[Nievas et al., 2023](http://static.seismo.ethz.ch/rise/deliverables/Deliverable_6.1.pdf)):

```
dmg_state,N_inspection,N_repair
DS0,7,0
DS1,7,15
DS2,45,365
DS3,45,1095
DS4,45,1095
```

The `recovery_injuries.csv` file (placed under `main_path/static`) must contain the (mean)
number of days spent in hospital by a person with different severities of injury, until they are
allowed to return to the building/s they usually occupy (in terms only of their health
condition). Each row must correspond to each of the injury severity levels listed in the
`config.yml` file under [injuries_scale](02_Configuration.md#injuries_scale). If the injuries
scale covers death, use a very large number for `N_discharged` under this category.

Example (from Deliverable 6.1 of the RISE project, see
[Nievas et al., 2023](http://static.seismo.ethz.ch/rise/deliverables/Deliverable_6.1.pdf);
severity of injuries as defined in HAZUS (FEMA, 2003)):

```
injuries_scale,N_discharged
1,0
2,3
3,8
4,36500
```

In the example shown above, injuries of severity level 4 are instant deaths or mortal injuries,
and the 36,500 days (around 100 years) are used to simulate this.

If all values in `recovery_damage.csv` and `recovery_injuries.csv` are set to zero, then no
update of occupants is carried out, which means that all occupants are assumed to be able and
allowed to occupy the buildings during all earthquakes, even if they are injured or the building
is fully damaged).

Return to [documentation index](README.md).
