# Configuration and running controls

The overall running of the Real-Time Loss Tools is controlled by three files, whose contents are
described in what follows: [config.yml](#configuration-file),
[triggering.csv](#triggering-file), and the [job.ini](#jobini-for-openquake) file for
OpenQuake.

## Configuration file

The configuration file named `config.yml` needs to contain the parameters explained below. An
example configuration file can be found [here](../config_example.yml).

### `description_general`

General description, used by OpenQuake to describe each job run. The Real-Time Loss Tools
will use this description as `description` within the `job.ini` file.

### `main_path`

Path to the main running directory, assumed to have the required
[file structure](03_Input.md#assumed-file-structure).

### `debug_logging`

Optional parameter. Set to `True` to get debug level logs. Skip or set to `False` to get info
level logs only (no debug messages).

### `oelf_source_model_filename`

Name of the XML file that contains the source model needed to create the earthquake ruptures to
run OELF calculations. This file is assumed to be located under `main_path/ruptures`.

### `state_dependent_fragilities`

Set to `True` if the fragility models used to run OpenQuake are state-dependent, set to `False`
otherwise. See details on the behaviour of the software with each option
[here](01_Overview.md#cumulative-damage-state-dependent-vs-state-independent-fragility-models).

### `mapping_damage_states`

These parameters link the names of damage states as they are output by OpenQuake and as they are
labelled in the exposure and fragility models. The strings before the colons are the former
while the strings to the right of the colons are the latter. The software assumes that the
damage states are input in order of severity, from least severe (no damage) to most severe.

In the following example, the strings `DS0`, `DS1`, etc., would be those used in the exposure
and fragility models, while `no_damage`, `dmg_1`, etc., are the standard labelling used by
OpenQuake.

```
mapping_damage_states:
  no_damage: DS0
  dmg_1: DS1
  dmg_2: DS2
  dmg_3: DS3
  dmg_4: DS4
```

### `oelf`

Parameters used to run the Operational Earthquake Loss Forecasting (OELF) calculations:

**min_magnitude**

Minimum magnitude to carry out a damage and loss assessment while running OELF. Earthquakes in
the OELF catalogues with magnitudes below `min_magnitude` are assumed to produce no additional
damage.

**max_distance**

Maximum epicentral distance between the earthquake source and the exposure sites to run the
earthquake. Earthquakes in the OELF catalogues located farther away from the exposure sites than
`max_distance` are assumed to produce no additional damage. 

**continuous_ses_numbering**

If set to `True`, the software assumes there are as many stochastic event sets as indicated by
`ses_range` (see below), with an increment of 1. If set to `False`, the Real-Time Loss Tools
simply read the IDs of the stochastic event sets from the OELF catalogue. This parameter is used
to take into account the fact that the input OELF CSV catalogue may not contain entries for all
realisations of seismicity if a minimum magnitude threshold was used to write the file. For
example, if a threshold of 4.0 was used to write the OELF CSV catalogue and there were
stochastic event sets that produced all earthquakes with magnitude smaller than 4.0, such event
sets would not be represented in the OELF CSV catalogue, but they are still a realisation of
seismicity that needs to be considered in the final damage and loss statistics of the forecast.
This is what setting `continuous_ses_numbering` to `True` allows to do: consider that these
stochastic event sets exist but cause no additional damage.

**ses_range**

Start and end (integer) number of the ID of the stochastic event sets to be considered, given as
a list separated by comma and space (", "). It is only used if `continuous_ses_numbering` is set
to `True`. For example,

```
continuous_ses_numbering: True
ses_range: 1, 10000
```

results in the Real-Time Loss Tools considering the existence of 10,000 stochastic event sets,
irrespective of how many of them contain eartquakes in the OELF catalogues CSV files.

**rupture_generator_seed**

Optional seed (positive non-zero integer) to set for the random number generator that controls
the stochastic rupture simulations, so that results are reproducible.

**rupture_region_properties**

Optional set of properties to control the generation and scaling of OELF ruptures according to
the tectonic region. One set with the following parameters can be specified for each tectonic
region in the OELF source model (`oelf_source_model_filename`):

- **msr**: Choice of magnitude-scaling relation (click
[here](https://docs.openquake.org/oq-engine/master/manual/hazard.html#magnitude-scaling-relationships)
for details on magnitude-scaling relations supported by OpenQuake).
- **area_mmax**: Maximum earthquake magnitude to cap the scaling of the rupture area. Magnitudes
larger than this will have rupture areas fixed to that corresponding to this `area_mmax`
magnitude. This parameter is used to avoid unrealistic ruptures from being generated when the
input OELF seismicity catalogues contain very large magnitudes.
- **aspect_limits**: Lower and upper limits used for randomly sampling the aspect ratio of the
ruptures.
- **default_usd**: Default upper seismogenic depth (km), if not specified in the source model.
- **default_lsd**: Default lower seismogenic depth (km), if not specified in the source model.

### `injuries_scale`

Scale of severity of human casualties (injuries, deaths), given as a list separated by comma and
space (", "). Example:

```
injuries_scale = 1, 2, 3, 4
```

### `injuries_longest_time`

Maximum number of days after an earthquake has occurred that will be used to build the timeline
of occupants for the future, irrespective of the number of days indicated in
`recovery_damage.csv` or `recovery_injuries.csv`.

### `time_of_day_occupancy`

Factors to be used to multiply the number of census occupants to obtain the number of people in
a building at a certain time of the day. They need to be specified for each occupancy case
covered by the occupancy field of the exposure CSV file. For each occupancy, factors for day,
night and transit times are needed. The following example shows the factors that correspond to
Italy in the European Seismic Risk Model 2020 (ESRM20; Crowley et al., 2020, 2021).

```
time_of_day_occupancy:
  residential:
    day: 0.242853  # approx. 10 am to 6 pm
    night: 0.9517285  # approx. 10 pm to 6 am
    transit: 0.532079  # approx. 6 am to 10 am and 6 pm to 10 pm
  commercial:
    day: 0.4982155
    night: 0.0436495
    transit: 0.090751
  industrial:
    day: 0.4982155
    night: 0.0436495
    transit: 0.090751
```

Click [here](https://gitlab.seismo.ethz.ch/efehr/esrm20_exposure/-/blob/master/social_indicators/population_distribution_PAGER.xlsx)
for ESRM20 factors for other European countries.

### `timezone`

Local time zone in the format of the [IANA Time Zone Database](https://www.iana.org/time-zones),
used to convert from UTC time (in which the earthquakes are specified) to local time, for the
purpose of determining the time of the day of the earthquake and, as a consequence, the number
of occupants in the buildings. Example:

```
timezone = "Europe/Rome"
```

### `store_intermediate`

If `True`, intermediate results/calculations including intermediate exposure models and damage
states are stored. Benchmarking has shown that setting this parameter to `False` does not result
in a significant reduction in running time.

The effect of `store_intermediate` on the files that are stored is shown in the following table,
which specifies if each kind of file is stored ("yes") or not ("no"). Only directories that are
used for storing output are listed:

| Path                 | File                                                               | `store_intermediate` = True | `store_intermediate` = False |
|----------------------|--------------------------------------------------------------------|-----------------------------|------------------------------|
| current              | exposure_model_current.csv                                         | yes                         | yes                          |
| current/occupants    | injured_still_away_after_RLA_trig_name.csv                         | yes                         | yes                          |
| current/occupants    | occupancy_factors_after_RLA_trig_name.csv                          | yes                         | yes                          |
| exposure_models/oelf | forecast/exposure_model_current_oelf_SES.csv                       | yes                         | yes                          |
| exposure_models/oelf | forecast/exposure_model_after_SES_EQ.csv                           | yes                         | no                           |
| exposure_models/rla  | exposure_model_after_EQ.csv                                        | yes                         | no                           |
| openquake_output     | EQ_damages_OQ.csv                                                  | yes                         | no                           |
| openquake_output     | EQ_damages_OQ_raw.csv                                              | yes                         | no                           |
| openquake_output     | forecast/SES_EQ_damages_OQ.csv                                     | yes                         | no                           |
| openquake_output     | forecast/SES_EQ_damages_OQ_raw.csv                                 | yes                         | no                           |
| oputput              | damage_states_after_RLA_trig_name.csv                              | yes                         | yes                          |
| oputput              | damage_states_after_OELF_trig_name.csv                             | yes                         | yes                          |
| oputput              | losses_economic_after_RLA_trig_name.csv                            | yes                         | yes                          |
| oputput              | losses_economic_after_OELF_trig_name.csv                           | yes                         | yes                          |
| oputput              | losses_human_after_RLA_trig_name.csv                               | yes                         | yes                          |
| oputput              | losses_human_after_OELF_trig_name.csv                              | yes                         | yes                          |
| oputput              | forecast/damage_states_after_OELF_trig_name_realisation_SES.csv    | yes                         | no                           |
| oputput              | forecast/losses_economic_after_OELF_trig_name_realisation_SES.csv  | yes                         | no                           |
| oputput              | forecast/losses_human_after_OELF_trig_name_realisation_SES.csv     | yes                         | no                           |
| oputput              | all_damage_states_after_RLA_cumulative.csv                         | (Note 5)                    | (Note 5)                     |
| oputput              | all_damage_states_after_RLA_incremental.csv                        | (Note 5)                    | (Note 5)                     |
| oputput              | all_damage_states_after_OELF_cumulative.csv                        | (Note 5)                    | (Note 5)                     |
| oputput              | all_losses_economic_RLA_cumulative_absolute.csv                    | (Note 5)                    | (Note 5)                     |
| oputput              | all_losses_economic_RLA_cumulative_ratio.csv                       | (Note 5)                    | (Note 5)                     |
| oputput              | all_losses_economic_RLA_incremental_absolute.csv                   | (Note 5)                    | (Note 5)                     |
| oputput              | all_losses_economic_RLA_incremental_ratio.csv                      | (Note 5)                    | (Note 5)                     |
| oputput              | all_losses_economic_OELF_cumulative_absolute.csv                   | (Note 5)                    | (Note 5)                     |
| oputput              | all_losses_economic_OELF_cumulative_ratio.csv                      | (Note 5)                    | (Note 5)                     |
| oputput              | all_losses_human_severity_X_RLA_cumulative_absolute.csv            | (Note 5)                    | (Note 5)                     |
| oputput              | all_losses_human_severity_X_RLA_cumulative_ratio.csv               | (Note 5)                    | (Note 5)                     |
| oputput              | all_losses_human_severity_X_RLA_incremental_absolute.csv           | (Note 5)                    | (Note 5)                     |
| oputput              | all_losses_human_severity_X_RLA_incremental_ratio.csv              | (Note 5)                    | (Note 5)                     |
| oputput              | all_losses_human_severity_X_OELF_incremental_absolute.csv          | (Note 5)                    | (Note 5)                     |
| oputput              | all_losses_human_severity_X_OELF_incremental_ratio.csv             | (Note 5)                    | (Note 5)                     |
| oputput              | all_portfolio_losses_economic_RLA_cumulative_absolute.csv          | (Note 5)                    | (Note 5)                     |
| oputput              | all_portfolio_losses_economic_RLA_cumulative_ratio.csv             | (Note 5)                    | (Note 5)                     |
| oputput              | all_portfolio_losses_economic_RLA_incremental_absolute.csv         | (Note 5)                    | (Note 5)                     |
| oputput              | all_portfolio_losses_economic_RLA_incremental_ratio.csv            | (Note 5)                    | (Note 5)                     |
| oputput              | all_portfolio_losses_human_severity_X_RLA_cumulative_absolute.csv  | (Note 5)                    | (Note 5)                     |
| oputput              | all_portfolio_losses_human_severity_X_RLA_cumulative_ratio.csv     | (Note 5)                    | (Note 5)                     |
| oputput              | all_portfolio_losses_human_severity_X_RLA_incremental_absolute.csv | (Note 5)                    | (Note 5)                     |
| oputput              | all_portfolio_losses_human_severity_X_RLA_incremental_ratio.csv    | (Note 5)                    | (Note 5)                     |
| ruptures/rla         | rupture_EQ.xml                                                     | yes                         | yes                          |
| ruptures/oelf        | forecast/RUP_SES-EQ.xml                                            | yes                         | yes                          |

Notes on nomenclature:

1. `SES` refers to "stochastic event set", that is, each individual realisation of seismicity
for a given period of time in the seismicity forecast. As part of a file name, it represents the
ID of the SES.
2. `EQ` refers to the earthquake ID.
3. `trig_name` refers to each of the CSV files contained in `catalogues` and listed in the
`triggering.csv` file.
4. `X` refers tp the severity level in the scale defined by the user through `injuries_scale`
(see above).
5. These are independent of `store_intermediate` and only depend on `post_process>collect_csv`
(see below).

### `store_openquake`

If set to `True`, OpenQuake HDF5 files will be stored and jobs will be kept in OpenQuake's
database. If set to `False`, these will be deleted after the damage results are retrieved.
Setting `store_openquake` to False avoids very long lists of run jobs when running the OpenQuake
commands `oq engine --lrc`.

### `post_process`

Parameters controlling the post-processing of results (after all RLA and OELF calculations
specified in the `triggering.csv` file have been run):

**collect_csv**

If set to `True`, individual damage and loss results (due to each RLA earthquake and each OELF
catalogue) are collected and exported to one RLA and one OELF CSV file (per result type).

## Triggering file

The `triggering.csv` file is used to simulate the triggering of a rapid loss assessment when an
earthquake of interest occurs (i.e., when it is detected by a network, its source parameters are
calculated, etc) and an operational earthquake loss forecast when a short-term seismicity
catalogue becomes available (an operation potentially triggered in itself by the occurrence of a
real earthquake, or at pre-defined moments in time).

The `triggering.csv` file consists of two mandatory columns and a third optional column:

- `catalogue_filename`: name of the catalogue CSV file (each of the files contained in
`main_path/catalogues`, see explanation on [input files](03_Input.md)); 
- `type_analysis`: type of analysis to run with the corresponding catalogue, either RLA (rapid
loss assessment) or OELF (operational earthquake loss forecast). When RLA is indicated, only the
first row of the catalogue (apart from the column names) is read.
- `rupture_xml` (optional): name of the rupture XML file for the RLA analyses. Leave empty for
OELF analyses or when providing a [source_parameters.csv](03_Input.md#rupture-parameters-for-rla)
file for the software to build the XML rupture. Rupture XML files listed here must exist under
`main_path/ruptures/rla`. Rupture XML files listed here take precedence over rupture parameters
in the `source_parameters.csv` for the same earthquake.

Example (without `rupture_xml` column):

```
catalogue_filename,type_analysis
forecast_YYYY_MM_DDT00_00_00.csv,OELF
real_EQ_01.csv,RLA
forecast_after_real_EQ_01.csv,OELF
...
```

Example (with `rupture_xml` column):

```
catalogue_filename,type_analysis,rupture_xml
forecast_YYYY_MM_DDT00_00_00.csv,OELF,
real_EQ_01.csv,RLA,
forecast_after_real_EQ_01.csv,OELF,
real_EQ_02.csv,RLA,earthquake_02_rupt.xml
...
```

In this last example, the Real-Time Loss Tools will build the rupture for `real_EQ_01.csv,RLA`
using the parameters in `source_parameters.csv`, and directly take the file
`earthquake_02_rupt.xml` for `real_EQ_02.csv`.

## job.ini for OpenQuake

A `job.ini` file placed under `main_path/current` is needed to run
[OpenQuake](https://github.com/gem/oq-engine). The file is updated by the Real-Time Loss tools
for running each earthquake, but a series of parameters do remain the same across all runs. An
example `job.ini` file can be found [here](../job_ini_example.ini).

The `job.ini` file contains the following parameters:

### `general`

**description**

Its original content is not relevant, as it will be replaced by the contents of
[description_general](#description_general) from the config.yml file.

**calculation_mode**

It must be `scenario_damage`.

**ses_seed**

If desired, indicate an integer for the purpose of reproducibility.

### `exposure`

**exposure_file**

Path and filename to the exposure XML file. It must be `exposure_model.xml`.

**taxonomy_mapping_csv**

Path and filename to the exposure-vulnerability mapping CSV. The path must be
`../static/exposure_vulnerability_mapping.csv`, though the name of the CSV file can be
different.

**time_event**

It can contain any string, as it will be replaced by the time of the day associated with each
earthquake to be run.

### `fragility`

**structural_fragility_file**

Path and filename to the fragility XML file. It must be `../static/fragility_model.xml`.

### `Rupture information`

**rupture_model_file**

Path and filename to the rupture XML file. It can contain any string, as it will be replaced by
the Real-Time Loss Tools with the name and location of each rupture XML file to be used to run
OpenQuake.

**rupture_mesh_spacing**

Spacing used to discretise the rupture plane (in km).

### `Site conditions`

**site_model_file**

Path and filename to the site model CSV file. It must be `../static/site_model.csv`.

### `Calculation parameters`

**gsim_logic_tree_file**

Path and filename to the ground motion logic tree XML file. It must be
`../static/gmpe_logic_tree.xml`.

**truncation_level**

Number of standard deviations to consider for sampling ground motions from the ground motion
prediction equation.

**maximum_distance**

Maximum epicentral distance (km) between earthquake source and site. All exposure sites located
at a longer distance from the earthquake are ignored by OpenQuake.

**number_of_ground_motion_fields**

Number of stochastic realisations of ground motion to be generated for each earthquake.

**minimum_intensity: DO NOT USE**

Minimum ground motion intensity at a site for damage to be calculated. This parameter should
**NOT** be set, as OpenQuake assumes zero damage if ground motion intensities are below this
threshold. While this is useful for state-independent calculations, in the case of
state-dependent calculations low ground motions cause no **additional** damage, which is not the
same as **no** damage (i.e. resetting a damaged building to an undamaged condition).


The aforementioned parameters are not necessarily the only ones that could be specified in the
`job.ini`, as [OpenQuake](https://github.com/gem/oq-engine) is a complete software of its own
with a large number of options and configurable parameters. Parameters that do not need updating
in between runs (of different RLA or OELF calculations) could be directly incorporated to the
`job.ini` and other OpenQuake-specific input files. It is noted, however, that the behaviour of
the Real-Time Loss Tools has not been tested with parameters other than the ones described
above.

Return to [documentation index](README.md).
