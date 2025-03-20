# NR_Radiation Module

## Overview
The nr_radiation module implements moment-based radiation transport methods for Athena++, supporting both explicit and implicit time integration schemes. It handles multi-group radiative transfer with various microphysical processes including absorption, scattering, and Compton effects.

## Key Features
- **Multi-group transport** with arbitrary frequency binning (`frequencygrid.cpp`)
- **Angular moment integration** for radiation fields (`get_moments.cpp`)
- **Implicit solvers** for stiff radiation problems (`implicit/`)
- **Frame transformations** for relativistic applications (`frame_transform.cpp`)
- **Source terms** implementation including:
  - Thermal absorption/emission
  - Compton scattering
  - Multi-group dust scattering

## Directory Structure
```
nr_radiation/
├── radiation.cpp          # Main radiation transport driver
├── radiation.hpp          # Class definitions and interfaces
├── implicit/              # Implicit time integration schemes
│   ├── rad_iteration.cpp  # Nonlinear iteration controller
│   └── radiation_implicit.hpp  # Implicit solver base class
├── integrators/           # Core radiation transport implementations
│   ├── rad_transport.cpp  # Explicit transport implementation
│   ├── multi_group.cpp    # Multi-group moment integration
│   └── srcterms/          # Microphysical source terms
└── inputs/radiation/      # Example input files (see Athena++ inputs directory)
```

## Usage
1. Enable in configure.py:
```python
radiation = 'n_radiation'
```

2. Set radiation parameters in input file:
```makefile
<radiation>
  nfre_ang = 8           # Number of angular bins
  radiation_type = multi_group
  implicit = true        # Use implicit time integration
</radiation>
```

3. Key runtime parameters:
- `n_frequency` - Number of frequency groups
- `emissivity` - Thermal emission model selection
- `opacity_table` - Microphysics data file path

## References
[1] Jiang et al. (2022) - Implicit Radiation Transport Formulation  
[2] Zhang & Davis (2017) - Multi-group Moment Method  
[3] Athena++ Radiation Manual (docs/radiation.pdf)
