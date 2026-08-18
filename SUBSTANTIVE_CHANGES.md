# Substantive changes in the Python rewrite

This list records changes that can affect the scientific result, numerical output, or interpretation. Purely syntactic MATLAB-to-Python changes are omitted.

1. **Python 3.13 is specified rather than 3.14.** Current UTide package metadata explicitly advertises support through Python 3.13, so 3.13 is the conservative reproducible choice. GSW-Python supports Python 3.14.
2. **Python UTide replaces MATLAB UTide.** It is a Python reimplementation of the same UTide analysis family, but numerical details and option defaults are not guaranteed to be identical.
3. **Nodal/satellite corrections are enabled.** The MATLAB practical used `NodSatNone`; the rewrite uses `nodal=True`, which is more appropriate for long records because constituent amplitudes/phases are modulated over the lunar nodal cycle.
4. **Section 3 performs the harmonic fit on the complete valid tide-gauge record.** It no longer loads a full-record synthetic tide derived from only five years in a chosen decade. This removes the assumption that a short-record tidal solution is stationary over the full record and allows the updated record to be analysed through the present.
5. **The default full-record fit uses eight major constituents (M2, S2, N2, K2, K1, O1, P1, Q1).** This bounds memory/runtime and makes the practical reproducible. Students can expand the list; Section 2 still permits UTide `auto` selection.
6. **The running mean is time-aware and gap-aware.** Pandas time-based rolling means replace filtering of a compressed non-NaN sequence. The original approach could average observations on opposite sides of long data gaps.
7. **Long-term trends are fitted to annual means.** This avoids treating hundreds of thousands of serially correlated hourly/smoothed observations as independent regression points and gives each adequately sampled year comparable weight. Years with poor tide-gauge coverage are excluded by an explicit threshold.
8. **Trend uncertainty is a confidence interval on the slope.** The MATLAB code used `polyconf(..., 'Predopt','curve')` and labelled a vertical confidence-band width as `mm/year`; that is dimensionally not a slope uncertainty. The rewrite reports a Student-t OLS interval for the slope itself.
9. **The steric calculation uses TEOS-10 variables consistently.** Archived Practical Salinity and in-situ temperature are converted to Absolute Salinity and Conservative Temperature before calculating density. The reference state follows the same conversion pathway.
10. **Steric integration uses geometric layer thickness.** Pressure is converted to vertical coordinate using `gsw.z_from_p`; the original used pressure increments in dbar directly as an approximate depth increment.
11. **Steric smoothing uses a centred time-based running mean.** This replaces the original forward-looking overlapping bins. The default is 180 days (rather than 30 days) because the hydrographic casts are sparse compared with the hourly tide-gauge record; the window remains student-editable.
12. **Steric and tide-gauge trends are compared over exactly the same annual samples.** In the MATLAB Section 4, the steric fit used the selected hydrographic period but the tide-gauge trend was recalculated over the complete tide-gauge record. The rewrite aligns the two annual time series before fitting.
13. **The Section 3 running mean is applied to the detided series.** The MATLAB code calculated `DeTide` but then smoothed `TGLevel` (raw sea level). The rewrite smooths the detided record explicitly.
14. **Steric profiles are standardised to a 0–2000 dbar column.** Small endpoint gaps are filled by interpolation/limited endpoint extrapolation; profiles not reaching at least 1900 dbar or not approaching the surface are excluded. This avoids silently integrating different depth ranges through time.
15. **Daily means are used for plotting the raw detided residual in Section 3.** This is a display-only reduction; harmonic analysis still uses every valid observation and the trend uses annual means.

## Remaining statistical caveat

The OLS slope confidence intervals assume independent annual residuals. Annual averaging reduces short-period serial correlation but does not guarantee independence. A more advanced treatment could use generalized least squares, an autoregressive residual model, or a block bootstrap.
