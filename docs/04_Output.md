# Output files

Output files can be classified into three broad groups: those containing damage states, those
containing economic losses, and those containing human casualties. Each of these are explained
under the headings below. An additional [log for quick checks](#log-for-quick-checks) named
`quick_input_check.txt` is created as well.

In all results associated with human casualties, `severity_X` refers to the severity of the
sustained injuries as per the injury severity levels listed in the `config.yml` file under
[injuries_scale](02_Configuration.md#injuries_scale).

All main output files are written to `main_path/output`. Intermediate OpenQuake output files are
written to `main_path/openquake_output`.

## Damage States

Output files associated with individual RLA earthquakes or OELF forecasts:
- `damage_states_after_RLA_EQ_XX.csv`: Expected probabilities or number of buildings
per `building_id` in each damage state after the RLA run for `EQ_XX.csv` (as per the name of the
corresponding catalogue file). Values are cumulative (i.e. considering all RLAs run up to each
point in time). One row per combination of `building_id` and damage state. Missing damage states
are associated with zero values.
- `damage_states_after_OELF_forecast_name_X.csv`: The same, but after the OELF run of the
catalogue named `forecast_name_X`. Values are cumulative (i.e., considering damage from all RLAs
run before this OELF catalogue and the OELF catalogue itself).

Output files gathering results from all RLA earthquakes or all OELF forecasts (one row per
combination of `building_id` and damage state, one column per RLA or OELF calculation):
- `all_damage_states_after_OELF_cumulative.csv`: Expected probabilities or number of buildings
per `building_id` in each damage state after each OELF calculation. Values are cumulative with
respect to already-run RLAs.
- `all_damage_states_after_RLA_cumulative.csv`: Expected probabilities or number of buildings
per `building_id` in each damage state after each RLA. Values are cumulative (i.e. considering
all RLAs run up to each point in time).
- `all_damage_states_after_RLA_incremental.csv`: Expected changes in probabilities or number of
buildings per `building_id` for each damage state due to each RLA. 

If [store_intermediate](02_Configuration.md#store_intermediate) is set to `True` in the
`config.yml` file, one sub-folder will be created for each OELF catalogue and, within each
subfolder:
- `damage_states_after_OELF_forecast_name_X_realisation_N.csv`: Expected probabilities or number
of buildings per `building_id` in each damage state after the calculation run for stochastic
event set `N` of the OELF catalogue with name `forecast_name_X`. Values are cumulative (i.e.,
considering damage from all RLAs run before this OELF catalogue and this stochastic event set
itself).

If [store_intermediate](02_Configuration.md#store_intermediate) is set to `True` in the
`config.yml` file, the following files/directories are created under
`main_path/openquake_output`:
- `EQ_XX_damages_OQ_raw.csv`: Damage outputs as retrieved from OpenQuake directly. Each row
corresponds to a combination of `asset_id` and damage state.
- `EQ_XX_damages_OQ.csv`: The same as `EQ_XX_damages_OQ_raw.csv`, but adjusted to ensure that no
small negative values are assigned to the no damage condition due to floating point precision
issues.
- One sub-folder for each OELF catalogue and, within each subfolder:
  - `NNN-EE_damages_OQ_raw.csv`: The same as `EQ_XX_damages_OQ_raw.csv`, for earthquake `EE` of
  stochastic event set `NNN` of this OELF catalogue.
  - `NNN-EE_damages_OQ.csv`: The same as `NNN-EE_damages_OQ_raw.csv`, but adjusted to ensure
  that no small negative values are assigned to the no damage condition due to floating point
  precision issues.

## Economic Losses

All economic losses are calculated using the replacement values specified in the exposure model.
As a result, the currency of the losses is the same as for those replacement costs.

Output files associated with individual RLA earthquakes or OELF forecasts:
- `losses_economic_after_RLA_EQ_XX.csv`: Expected economic losses per `building_id` after the
RLA run for `EQ_XX.csv`. Values are cumulative (i.e. considering all RLAs run up to each
point in time).
- `losses_economic_after_OELF_forecast_name_X.csv`: The same, but after the OELF run of the
catalogue named `forecast_name_X`. Values are cumulative (i.e., considering losses from all RLAs
run before this OELF catalogue and the OELF catalogue itself).

Output files gathering results from all RLA earthquakes or all OELF forecasts:
- `all_losses_economic_OELF_cumulative_absolute.csv`: Expected economic losses per `building_id`
after each OELF calculation (mean of all stochastic event sets of seismicity). Values are
cumulative with respect to already-run RLAs.
- `all_losses_economic_OELF_cumulative_ratio.csv`: Expected economic loss ratios per
`building_id` (as percentages) after each OELF calculation (mean of all stochastic event sets of
seismicity). Values are cumulative with respect to already-run RLAs.
- `all_losses_economic_RLA_cumulative_absolute.csv`: Expected economic losses per `building_id`
after each RLA calculation. Values are cumulative (i.e. considering all RLAs run up to each
point in time).
- `all_losses_economic_RLA_cumulative_ratio.csv`: Expected economic loss ratios per
`building_id` (as percentages) after each RLA calculation. Values are cumulative (i.e.
considering all RLAs run up to each point in time).
- `all_losses_economic_RLA_incremental_absolute.csv`: Expected incremental economic losses per
`building_id` due to each RLA calculation (i.e. current economic loss minus economic loss up to
the previous earthquake).
- `all_losses_economic_RLA_incremental_ratio.csv`: Expected incremental economic loss ratios per
`building_id` (as percentages) due to each RLA calculation (i.e. current economic loss ratio
minus economic loss ratio up to the previous earthquake).
- `all_portfolio_losses_economic_RLA_cumulative_absolute.csv`: Expected economic losses for the
whole building portfolio after each RLA calculation. Values are cumulative (i.e. considering all
RLAs run up to each point in time). Equivalent of
`all_losses_economic_RLA_cumulative_absolute.csv` but for the whole portfolio.
- `all_portfolio_losses_economic_RLA_cumulative_ratio.csv`: Expected economic loss ratios for
the whole building portfolio (as percentages) after each RLA calculation. Values are cumulative
(i.e. considering all RLAs run up to each point in time). Equivalent of
`all_losses_economic_RLA_cumulative_ratio.csv` but for the whole portfolio.
- `all_portfolio_losses_economic_RLA_incremental_absolute.csv`: Expected incremental economic
losses for the whole building portfolio due to each RLA calculation (i.e. current economic loss
minus economic loss up to the previous earthquake). Equivalent of
`all_losses_economic_RLA_incremental_absolute.csv` but for the whole portfolio.
- `all_portfolio_losses_economic_RLA_incremental_ratio.csv`: Expected incremental economic loss
ratios for the whole building portfolio (as percentages) due to each RLA calculation (i.e. current
economic loss ratio minus economic loss ratio up to the previous earthquake). Equivalent of
`all_losses_economic_RLA_incremental_ratio.csv` but for the whole portfolio.

If [store_intermediate](02_Configuration.md#store_intermediate) is set to `True` in the
`config.yml` file, one sub-folder will be created for each OELF catalogue and, within each
subfolder:
- `losses_economic_after_OELF_forecast_name_X_realisation_N.csv`: Expected economic losses per
`building_id` after the calculation run for stochastic event set `N` of the OELF catalogue with
name `forecast_name_X`. Values are cumulative (i.e., considering damage from all RLAs run before
this OELF catalogue and this stochastic event set itself).

## Human Casualties

In all output files enumerated below, the ratios of people injured with different degrees of
severity are calculated with respect to the total number of census occupants of the buildings,
not with respect to the number of people present in the building at the time of the earthquake.

Output files associated with individual RLA earthquakes or OELF forecasts:
- `losses_human_after_RLA_EQ_XX.csv`: Expected human casualties of all severity levels (one
column per injury severity level `X` listed in the `config.yml` file under
[injuries_scale](02_Configuration.md#injuries_scale)) per `building_id` after the RLA run for
`EQ_XX.csv` . Values are incremental (due only to this earthquake).
- `losses_human_after_OELF_forecast_name_X.csv`: The same, but after the OELF run of the
catalogue named `forecast_name_X`. Values are incremental (due only to this OELF).

Output files gathering results from all RLA earthquakes or all OELF forecasts:
- `all_losses_human_severity_X_OELF_incremental_absolute.csv`: Expected human casualties of
severity level `X` due to each OELF calculation (mean of all stochastic event sets of
seismicity) per `building_id`.
- `all_losses_human_severity_X_OELF_incremental_ratio.csv`: Expected ratio of human casualties
of severity level `X` due to each OELF calculation (mean of all stochastic event sets of
seismicity) per `building_id`. The ratio is calculated with respect to the total number of
census occupants associated with each `building_id` and expressed as a percentage.
- `all_losses_human_severity_X_RLA_cumulative_absolute.csv`: Expected human casualties of
severity level `X` after each RLA calculation per `building_id`. Values are cumulative (i.e.
considering all RLAs run up to each point in time).
- `all_losses_human_severity_X_RLA_cumulative_ratio.csv`: Expected human casualties of severity
level `X` after each RLA calculation per `building_id`. Ratios are cumulative (i.e. considering
all RLAs run up to each point in time) and calculated with respect to the total number of census
occupants associated with each `building_id` (expressed as percentages).
- `all_losses_human_severity_X_RLA_incremental_absolute.csv`: Expected human casualties of
severity level `X` due to each RLA calculation per `building_id`.
- `all_losses_human_severity_X_RLA_incremental_ratio.csv`: Expected ratio of human casualties of
severity level `X` due to each RLA calculation per `building_id`. The ratio is calculated with
respect to the total number of census occupants associated with each `building_id` and expressed
as a percentage.
- `all_portfolio_losses_human_severity_X_RLA_cumulative_absolute.csv`: Equivalent of
`all_losses_human_severity_X_RLA_cumulative_absolute.csv` but for the whole building portfolio.
- `all_portfolio_losses_human_severity_X_RLA_cumulative_ratio.csv`: Equivalent of
`all_losses_human_severity_X_RLA_cumulative_ratio.csv` but for the whole building portfolio.
- `all_portfolio_losses_human_severity_X_RLA_incremental_absolute.csv`: Equivalent of
`all_losses_human_severity_X_RLA_incremental_absolute.csv` but for the whole building portfolio.
- `all_portfolio_losses_human_severity_X_RLA_incremental_ratio.csv`: Equivalent of
`all_losses_human_severity_X_RLA_incremental_ratio.csv` but for the whole building portfolio.

If [store_intermediate](02_Configuration.md#store_intermediate) is set to `True` in the
`config.yml` file, one sub-folder will be created for each OELF catalogue and, within each
subfolder:
- `losses_human_after_OELF_forecast_name_X_realisation_N.csv`: Expected human casualties of all
severity levels (one column per injury severity level `X`) per `building_id` after the
calculation run for stochastic event set `N` of the OELF catalogue with name `forecast_name_X`.
Values are cumulative (i.e., considering damage from all RLAs run before this OELF catalogue and
this stochastic event set itself).

## Log for quick checks

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
- `With update of occupants in 'recovery_injuries'`: The code adds up the number of days input
by the user in `recovery_injuries.csv`. See explanation above.

Return to [documentation index](README.md).
