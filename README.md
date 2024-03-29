# Auxiliary-function-based time delay estimation (AuxTDE)
This repository contains the codes for the AuxTDE algorithm.

## Author
[Kouei Yamaoka](https://k-yamaoka.net/en/)

## Installation
Clone this repository and run following commands:

``` shell
% conda env create -f=environment.yml
```
or
``` shell
% conda install numpy matplotlib
```

## codes
- AuxTDE: AuxTDE (main)
- single_AuxTDE: AuxTDE for 2ch sensor array
- TDE: conventional TDE methods

## sample codes
- sample\_AuxTDE: sample code (8ch sensor array)
- sample\_AuxTDE\_2ch: sample code (2ch sensor array)
  - PW-AuxTDE, AuxTDE_unitAmp, and AuxTDE_shrdAmp are theoretically identical in this case
- sample\_AuxTDE\_simple: the simplest sample code

## Visualization
- plot\_single\_aux: plot auxiliary function for 2ch sensor array
- plot\_multi\_aux: plot auxiliary function

## References
1. Kouei Yamaoka, Yukoh Wakabayashi, and Nobutaka Ono, &quot;Estimation of Consistent Time Delays in Subsample via Auxiliary-Function-Based Iterative Updates,&quot;
2. Kouei Yamaoka, Robin Scheibler, Nobutaka Ono, and Yukoh Wakabayashi, &quot;Sub-Sample Time Delay Estimation via Auxiliary-Function-Based Iterative Updates,&quot; in Proc. WASPAA, pp. 125&ndash;129, Oct. 2019.
