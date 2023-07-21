# Notes

`test_oelf_01.csv` contains a small fictitious seismicity forecast composed of four assumed
stochastic event sets:
- SES 1: The first two earthquakes have magnitudes, dates/times, and hypocentral locations
corresponding to event IDs IT-2009-0009 and IT-2009-0032 of [ITACA](https://itaca.mi.ingv.it)
(Russo et al., 2022). The third earthquake is fictitious. Damage is expected only from the first
two events, as the third one is too small and should lead to no ground motion fields being
generated. Injuries are expected only from the first one, as the other two should have zero
occupants.
- SES 2: It is not present in `test_oelf_01.csv`; simulating the case in which the SES only
contains earthquakes with magnitudes smaller than the threshold used to generate the output of
the seismicity forecast. No damage, no losses.
- SES 3: The two earthquakes are also fictitious and are expected to cause damage. The first
earthquake should cause injuries, but the second one should not due to the first one occurring
shortly before.
- SES 4: The first earthquake is expected to cause damage and injuries. The second earthquake
causes further damage but no casualties due to the first one occurring shortly before.

## References

Russo E, Felicetta C, D Amico M, Sgobba S, Lanzano G, Mascandola C, Pacor F, Luzi L (2022)
Italian Accelerometric Archive v3.2 - Istituto Nazionale di Geofisica e Vulcanologia,
Dipartimento della Protezione Civile Nazionale. doi:10.13127/itaca.3.2
