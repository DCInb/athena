//======================================================================================
// Athena++ astrophysical MHD code
// Copyright (C) 2014 James M. Stone  <jmstone@princeton.edu>
//
// This program is free software: you can redistribute and/or modify it under the terms
// of the GNU General Public License (GPL) as published by the Free Software Foundation,
// either version 3 of the License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A 
// PARTICULAR PURPOSE.  See the GNU General Public License for more details.
//
// You should have received a copy of GNU GPL in the file LICENSE included in the code
// distribution.  If not see <http://www.gnu.org/licenses/>.
//======================================================================================
//! \file dmr.cpp
//  \brief Problem generator for double Mach reflection test.
//  Only works for genuinely 2D hydro problems in X1-X2 plane with adiabatic EOS.
//
// REFERENCE: P. Woodward & P. Colella, "The numerical simulation of two-dimensional
// fluid flow with strong shocks", JCP, 54, 115, sect. IVc.
//======================================================================================

// C++ headers
#include <iostream>   // endl
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../bvals/bvals.hpp"
#include "../parameter_input.hpp"
#include "../mesh.hpp"
#include "../hydro/hydro.hpp"
#include "../field/field.hpp"
#include "../hydro/eos/eos.hpp"
#include "../coordinates/coordinates.hpp"

// dmrbv_iib() - sets BCs on inner-x1 (left edge) of grid.  
// dmrbv_ijb() - sets BCs on inner-x2 (bottom edge) of grid.  
// dmrbv_ojb() - sets BCs on outer-x2 (top edge) of grid.  

void dmrbv_iib(MeshBlock *pmb, AthenaArray<Real> &a,
               int is, int ie, int js, int je, int ks, int ke);
void dmrbv_ijb(MeshBlock *pmb, AthenaArray<Real> &a,
               int is, int ie, int js, int je, int ks, int ke);
void dmrbv_ojb(MeshBlock *pmb, AthenaArray<Real> &a,
               int is, int ie, int js, int je, int ks, int ke);

// problem generator

void Mesh::ProblemGenerator(Hydro *phyd, Field *pfld, ParameterInput *pin)
{
  MeshBlock *pmb = phyd->pmy_block;
  std::stringstream msg;

  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;

  if (pmb->block_size.nx3 > 1) {
    msg << "### FATAL ERROR in Problem Generator" << std::endl << "nx3=" 
        << pmb->block_size.nx3 << " but this test only works for 2D" << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }

// Initialize shock using parameters defined in Woodward & Colella

  Real d0 = 8.0;
  Real e0 = 291.25;
  Real u0 =  8.25*sqrt(3.0)/2.0;
  Real v0 = -8.25*0.5;
  for (int j=js; j<=je; ++j) {
    for (int i=is; i<=ie; ++i) {
      Real shock_pos = 0.1666666666 + pmb->pcoord->x2v(j)/sqrt((double)3.0);
// upstream conditions
      phyd->u(IDN,ks,j,i) = 1.4;
      phyd->u(IEN,ks,j,i) = 2.5;
      phyd->u(IM1,ks,j,i) = 0.0;
      phyd->u(IM2,ks,j,i) = 0.0;
// downstream conditions
      if (pmb->pcoord->x1v(i) < shock_pos) {
        phyd->u(IDN,ks,j,i) = d0;
        phyd->u(IEN,ks,j,i) = e0 + 0.5*d0*(u0*u0+v0*v0);
        phyd->u(IM1,ks,j,i) = d0*u0;
        phyd->u(IM2,ks,j,i) = d0*v0;
      }
    }
  }

// Set boundary value function pointers

  pmb->pbval->EnrollHydroBoundaryFunction(inner_x1, dmrbv_iib);
  pmb->pbval->EnrollHydroBoundaryFunction(inner_x2, dmrbv_ijb);
  pmb->pbval->EnrollHydroBoundaryFunction(outer_x2, dmrbv_ojb);
}

//--------------------------------------------------------------------------------------
//! \fn void dmrbv_iib()
//  \brief Sets boundary condition on left X boundary (iib) for dmr test
//  Quantities at this boundary are held fixed at the downstream state

void dmrbv_iib(MeshBlock *pmb, AthenaArray<Real> &a,
               int is, int ie, int js, int je, int ks, int ke)
{
  Real d0 = 8.0;
  Real e0 = 291.25;
  Real u0 =  8.25*sqrt(3.0)/2.0;
  Real v0 = -8.25*0.5;

  for (int j=js; j<=je; ++j) {
    for (int i=1;  i<=(NGHOST); ++i) {
      a(IDN,ks,j,is-i) = d0;
      a(IM1,ks,j,is-i) = d0*u0;
      a(IM2,ks,j,is-i) = d0*v0;
      a(IEN,ks,j,is-i) = e0 + 0.5*d0*(u0*u0+v0*v0);
    }
  }
}

//--------------------------------------------------------------------------------------
//! \fn void dmrbv_ijb()
//  \brief  Sets boundary condition on lower Y boundary (ijb) for dmr test.
//  Quantaties at this boundary are held fixed at the downstream state for
//  x1 < 0.16666666, and are reflected for x1 > 0.16666666

void dmrbv_ijb(MeshBlock *pmb, AthenaArray<Real> &a,
               int is, int ie, int js, int je, int ks, int ke)
{
  Real d0 = 8.0;
  Real e0 = 291.25;
  Real u0 =  8.25*sqrt(3.0)/2.0;
  Real v0 = -8.25*0.5;

  for (int j=1;  j<=(NGHOST); ++j) {
    for (int i=is; i<=ie; ++i) {
      if (pmb->pcoord->x1v(i) < 0.1666666666) {
// fixed at downstream state
        a(IDN,ks,js-j,i) = d0;
        a(IM1,ks,js-j,i) = d0*u0;
        a(IM2,ks,js-j,i) = d0*v0;
        a(IEN,ks,js-j,i) = e0 + 0.5*d0*(u0*u0+v0*v0);
      } else {
// reflected
        a(IDN,ks,js-j,i) = a(IDN,ks,js+(j-1),i);
        a(IM1,ks,js-j,i) = a(IM1,ks,js+(j-1),i);
        a(IM2,ks,js-j,i) = -a(IM2,ks,js+(j-1),i);
        a(IEN,ks,js-j,i) = a(IEN,ks,js+(j-1),i);
      }
    }
  }
}

//--------------------------------------------------------------------------------------
//! \fn void dmrbv_ojb()
//  \brief Sets TIME-DEPENDENT boundary condition on upper Y boundary (ojb) for dmr test
//  Quantaties at this boundary are held fixed at the downstream state for
//  x1 < 0.16666666+v1_shock*time, and at the upstream state for
//  x1 > 0.16666666+v1_shock*time

void dmrbv_ojb(MeshBlock *pmb, AthenaArray<Real> &a,
               int is, int ie, int js, int je, int ks, int ke)
{
  Real d0 = 8.0;
  Real e0 = 291.25;
  Real u0 =  8.25*sqrt(3.0)/2.0;
  Real v0 = -8.25*0.5;
  Real shock_pos = 0.1666666666 + (1. + 20.*pmb->pmy_mesh->time)/sqrt(3.0);

  for (int j=1;  j<=(NGHOST); ++j) {
    for (int i=is; i<=ie; ++i) {
      if (pmb->pcoord->x1v(i) < shock_pos) {
// fixed at downstream state
        a(IDN,ks,je+j,i) = d0;
        a(IM1,ks,je+j,i) = d0*u0;
        a(IM2,ks,je+j,i) = d0*v0;
        a(IEN,ks,je+j,i) = e0 + 0.5*d0*(u0*u0+v0*v0);
      } else {
// fixed at upstream state
        a(IDN,ks,je+j,i) = 1.4;
        a(IM1,ks,je+j,i) = 0.0;
        a(IM2,ks,je+j,i) = 0.0;
        a(IEN,ks,je+j,i) = 2.5;
      }
    }
  }
}
