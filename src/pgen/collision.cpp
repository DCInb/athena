//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file collision.cpp
//! \brief Problem generator for collision of plasma slabs.

// C++ headers
#include <sstream>
#include <cmath>
#include <stdexcept>
#include <fstream>
#include <iostream>
#include <random>

// Athena++ headers
#include "../athena.hpp"
#include "../globals.hpp"
#include "../athena_arrays.hpp"
#include "../parameter_input.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../bvals/bvals.hpp"
#include "../utils/utils.hpp"
#include "../outputs/outputs.hpp"
#include "../scalars/scalars.hpp"
#include "../inputs/hdf5_reader.hpp"  // HDF5ReadRealArray()

namespace {
// declare the functions and variables that are used in the file 
Real den0, v0, b0;
Real a0, ky_over_2pi, kz_over_2pi;
Real ky, kz;
Real isDenTurb, denTurb;
std::mt19937 gen(12345);
std::uniform_real_distribution<> dist(0.0, 1.0);
}

//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//! \brief Function to initialize problem-specific data in mesh class.  Can also be used
//! to initialize variables which are global to (and therefore can be passed to) other
//! functions in this file.  Called in Mesh constructor.
//========================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin) {
	// input from problem
  den0 = pin->GetOrAddReal("problem","den0",1);
	v0   = pin->GetOrAddReal("problem","v0",1);
	b0   = pin->GetOrAddReal("problem","b0",1);

	a0            = pin->GetOrAddReal("problem","a0",1);
	ky_over_2pi   = pin->GetOrAddReal("problem","ky_over_2pi",1);
	kz_over_2pi   = pin->GetOrAddReal("problem","kz_over_2pi",1);
  isDenTurb     = pin->GetOrAddBoolean("problem","isDenTurb",false);
  denTurb       = pin->GetOrAddReal("problem","denTurb",0);

	ky = ky_over_2pi * 2 * PI;
	kz = kz_over_2pi * 2 * PI;

  // Read the initial density distribution from hdf5 file
  
}

//======================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Spherical Coords planetary engulfment problem generator
//======================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  // Reset the random number generator using the meshblock ID
  gen.seed(gid);
  
  // SETUP THE INITIAL CONDITIONS ON MESH
  for (int k=ks; k<=ke; k++) {
    Real z = pcoord->x3v(k);
    for (int j=js; j<=je; j++) {
      Real y = pcoord->x2v(j);
      for (int i=is; i<=ie; i++) {
        Real x  = pcoord->x1v(i);

        // set the density
	    	phydro->u(IDN,k,j,i) = den0;
        if (isDenTurb) {
          a0=0;
          if (dist(gen)>0.9 and abs(x)<1.0){
            phydro->u(IDN,k,j,i) += denTurb * dist(gen);
          }
        }
        
        // set the momenta components
				if (x > a0 * sin(ky*y) * sin(kz*z)) {
					phydro->u(IM1,k,j,i) = - phydro->u(IDN,k,j,i) * v0;	
				} else {
					phydro->u(IM1,k,j,i) = phydro->u(IDN,k,j,i) * v0;	
				}
	    	phydro->u(IM2,k,j,i) = 0.0;
        phydro->u(IM3,k,j,i) = 0.0;

        // set the scalar (conserved variables)
	    	if(x > a0 * sin(ky*y) * sin(kz*z)){
	        pscalars->s(0,k,j,i) = 1.0*phydro->u(IDN,k,j,i);
	      }else{
	        pscalars->s(0,k,j,i) = 1.e-30*phydro->u(IDN,k,j,i); // set small non-zero value
	      }
      }
    }
  } // end loop over cells

	// initialize interface B
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=is; i<=ie+1; ++i) {
          pfield->b.x1f(k,j,i) = 0.0;
        }
      }
    }
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je+1; ++j) {
        for (int i=is; i<=ie; ++i) {
          pfield->b.x2f(k,j,i) = b0;
        }
      }
    }
    for (int k=ks; k<=ke+1; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=is; i<=ie; ++i) {
          pfield->b.x3f(k,j,i) = 0.0;
        }
      }
    }
	}
  return;
} // end ProblemGenerator