# Notes

`test_oelf_01.csv` contains a small fictitious seismicity forecast composed of four assumed
stochastic event sets, as described in tests/data/integration_oelf/catalogues/README.md.
In the present test, the actual earthquakes are mostly irrelevant, as the fragility model has
been modified so that none of the earthquakes causes damage. The only condition for the
earthquakes is that they should still cause OpenQuake to calculate the ground motion fields.

## References

Russo E, Felicetta C, D Amico M, Sgobba S, Lanzano G, Mascandola C, Pacor F, Luzi L (2022)
Italian Accelerometric Archive v3.2 - Istituto Nazionale di Geofisica e Vulcanologia,
Dipartimento della Protezione Civile Nazionale. doi:10.13127/itaca.3.2
