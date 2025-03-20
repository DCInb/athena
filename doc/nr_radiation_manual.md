# NR_Radiation Module User Manual

## Overview
The NR_Radiation module implements moment-based radiation transport methods for Athena++, supporting both explicit and implicit time integration schemes. It handles multi-group radiative transfer with various microphysical processes including absorption, scattering, and Compton effects.

## Key Features
- Multi-group transport with arbitrary frequency binning
- Angular moment integration for radiation fields
- Implicit solvers for stiff radiation problems
- Frame transformations for relativistic applications
- Source terms implementation including:
  - Thermal absorption/emission
  - Compton scattering
  - Multi-group dust scattering

## Core Classes

### NRRadiation
Main class handling radiation transport. Key responsibilities:
- Initialization and configuration
- Frequency and angular grid setup
- Boundary value handling
- Data storage for radiation intensities and moments

### RadIntegrator
Handles numerical integration of radiation transport equations. Implements:
- Flux divergence calculations
- Implicit angular fluxes
- Source term calculations
- Compton scattering
- Multi-group frequency mapping
- Frame transformations

## Configuration

### Enable in configure.py
```python
radiation = 'n_radiation'
```

### Input File Parameters
```makefile
<radiation>
  nfre_ang = 8           # Number of angular bins
  radiation_type = multi_group
  implicit = true        # Use implicit time integration
</radiation>
```

### Key Runtime Parameters
- `n_frequency` - Number of frequency groups
- `emissivity` - Thermal emission model selection
- `opacity_table` - Microphysics data file path
- `nmu` - Number of polar angles
- `nzeta` - Number of polar angles (alternative)
- `npsi` - Number of azimuthal angles
- `vmax` - Maximum velocity for relativistic effects
- `tunit` - Temperature unit
- `rhounit` - Density unit
- `lunit` - Length unit

## Numerical Methods

### Flux Calculation
- First-order and higher-order flux divergence
- Implicit angular fluxes
- Multi-group frequency mapping

### Source Terms
- Thermal absorption/emission
- Compton scattering
- Multi-group dust scattering

### Frame Transformations
- Lab to comoving frame
- Comoving to lab frame
- Frequency shifting for relativistic effects

## Example Configuration

### Basic Setup
```makefile
<radiation>
  nfre_ang = 8
  radiation_type = multi_group
  implicit = true
  n_frequency = 10
  emissivity = thermal
  opacity_table = data/opacity_table.dat
</radiation>
```

### Relativistic Setup
```makefile
<radiation>
  nfre_ang = 16
  radiation_type = multi_group
  implicit = true
  vmax = 0.9
  n_frequency = 20
  emissivity = compton
  opacity_table = data/rel_opacity.dat
</radiation>
```

## References
1. Jiang et al. (2022) - Implicit Radiation Transport Formulation
2. Zhang & Davis (2017) - Multi-group Moment Method
3. Athena++ Radiation Manual (docs/radiation.pdf)
