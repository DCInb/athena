//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file planetary_engulfment_circular.cpp
//! \brief Problem generator for circular orbit planetary engulfment

// C++ headers
#include <sstream>
#include <cmath>
#include <stdexcept>
#include <fstream>
#include <iostream>
#define NARRAY 2527   // length of profile
#define NGRAVL 200


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

namespace {
// declare the functions and variables that are used in the file 
Real Interpolate1DArray(Real *x, Real *y, Real x0, int length);
void SumGasOnParticleAccels(Mesh *pm, Real (&xi)[3],Real (&ag1i)[3], Real (&ag2i)[3]);
void ParticleAccels(Real (&xi)[3],Real (&vi)[3],Real (&ai)[3]);
void particle_step(Real dt,Real (&xi)[3],Real (&vi)[3],Real (&ai)[3]);
void kick(Real dt,Real (&xi)[3],Real (&vi)[3],Real (&ai)[3]);
void drift(Real dt,Real (&xi)[3],Real (&vi)[3],Real (&ai)[3]);
Real fspline(Real r, Real eps);
void WritePMTrackfile(Mesh *pm);
void SumMencProfile(Mesh *pm, Real (&menc)[NGRAVL]);

Real rho[NARRAY], p[NARRAY], rad[NARRAY], menc_init[NARRAY];  // initial profile
Real logr[NGRAVL],menc[NGRAVL]; // enclosed mass profile
Real Ggrav, rstar_initial, mstar_initial, gamma_gas; // FOR RESCALING
Real GM1, GM2; // two point masses
Real rsoft2;
Real xi[3], vi[3], ai[3], agas1i[3], agas2i[3];
Real separation_stop_min;
int n_particle_substeps;

Real t_relax, tau_relax;
int trackfile_number, update_grav_every;
Real trackfile_dt, trackfile_next_time;
Real d_insp, v_insp;
} // namespace

void DiodeOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,FaceField &b,
		 Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);
void TwoPointMass(MeshBlock *pmb, const Real time, const Real dt,
		  const AthenaArray<Real> &prim, const AthenaArray<Real> *flux,
		  const AthenaArray<Real> &prim_scalar, const AthenaArray<Real> &bcc,
		  AthenaArray<Real> &cons, AthenaArray<Real> &cons_scalar); 

//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//! \brief Function to initialize problem-specific data in mesh class.  Can also be used
//! to initialize variables which are global to (and therefore can be passed to) other
//! functions in this file.  Called in Mesh constructor.
//========================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin) {
  // input from problem
  Ggrav         = pin->GetOrAddReal("problem","Ggrav",6.67408e-8);
  GM2           = pin->GetOrAddReal("problem","GM2",0.0);
  rstar_initial = pin->GetReal("problem","rstar_initial");
  mstar_initial = pin->GetReal("problem","mstar_initial");

  rsoft2        = pin->GetReal("problem","rsoft2");
  Real ecc      = pin->GetReal("problem","ecc");
  Real sma      = pin->GetReal("problem","sma");

  separation_stop_min = pin->GetReal("problem","separation_stop_min");
  n_particle_substeps = pin->GetInteger("problem","n_particle_substeps");
  update_grav_every   = pin->GetInteger("problem","update_grav_every");

  t_relax       = pin->GetOrAddReal("problem","t_relax",1.0);
  tau_relax     = pin->GetOrAddReal("problem","tau_relax",0.1);
  trackfile_dt  = pin->GetReal("problem","trackfile_dt");

  d_insp        = pin->GetOrAddReal("problem","d_insp",1.0);
  v_insp        = pin->GetOrAddReal("problem","v_insp",0.0);

  // input from hydro
  gamma_gas     = pin->GetReal("hydro","gamma");

  // input from mesh
  Real rmin = pin->GetOrAddReal("mesh","x1min",0);
  Real rmax = pin->GetOrAddReal("mesh","x1max",1);

  // read in profile arrays from file
  // make sure about positivity of the data
  std::ifstream infile("profile.dat"); 
  for (int i=0;i<NARRAY;i++) {
    infile >> rad[i] >> rho[i] >> p[i] >> menc_init[i];
  }
  infile.close();

  // RESCALE
  for(int i=0;i<NARRAY;i++){
    rad[i] = rad[i]*rstar_initial;
    rho[i] = rho[i]*mstar_initial/pow(rstar_initial,3);
    p[i]   = p[i]*Ggrav*pow(mstar_initial,2)/pow(rstar_initial,4);
    menc_init[i] = menc_init[i]*mstar_initial;
  }

  // set the inner point mass based on excised mass
  Real menc_rin = Interpolate1DArray(rad, menc_init, rmin, NARRAY);
  GM1 = Ggrav*menc_rin;
  Real GMenv = Ggrav*Interpolate1DArray(rad,menc_init,1.01*rstar_initial, NARRAY) - GM1;

  // allocate the enclosed mass profile
  Real logr_min = log10(rmin);
  Real logr_max = log10(rmax);
  
  for(int i=0;i<NGRAVL;i++){
    logr[i] = logr_min + (logr_max-logr_min)/(NGRAVL-1)*i;
    menc[i] = Interpolate1DArray(rad,menc_init, pow(10,logr[i]), NGRAVL);
  }

  // initialize the secondary
  xi[0]=(1+ecc)*sma; //apocenter
  xi[1]=0;
  xi[2]=0;

  Real vcirc = sqrt((GM1+GM2+GMenv)/sma);
  vi[0] = 0.0;
  vi[1] = sqrt(vcirc*vcirc*(1.0 - ecc)/(1.0 + ecc) ); //v_apocenter
  vi[2] = 0.0;

  // always write at startup
  trackfile_next_time = time;
  trackfile_number = 0;

  // enroll the BCs
  if(mesh_bcs[BoundaryFace::outer_x1] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::outer_x1, DiodeOuterX1);
  }

  // Enroll a Source Function
  EnrollUserExplicitSourceFunction(TwoPointMass);
  return;
}

//======================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Spherical Coords planetary engulfment problem generator
//======================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  // local vars
  Real den, pres;

  // Prepare index bounds including ghost cells
  int il = is - NGHOST;
  int iu = ie + NGHOST;
  int jl = js;
  int ju = je;
  if (block_size.nx2 > 1) {
    jl -= (NGHOST);
    ju += (NGHOST);
  }
  int kl = ks;
  int ku = ke;
  if (block_size.nx3 > 1) {
    kl -= (NGHOST);
    ku += (NGHOST);
  }
  
  // SETUP THE INITIAL CONDITIONS ON MESH
  for (int k=kl; k<=ku; k++) {
    Real ph = pcoord->x3v(k);
    for (int j=jl; j<=ju; j++) {
      Real th = pcoord->x2v(j);
      for (int i=il; i<=iu; i++) {
        Real r  = pcoord->x1v(i);

        // interpolate the initial profile
        // get the density and pressure 
	      den  = Interpolate1DArray(rad, rho, r , NARRAY);
	      pres = Interpolate1DArray(rad, p,   r , NARRAY);

        // set the density
	      phydro->u(IDN,k,j,i) = den;
        // set the momenta components
	      phydro->u(IM1,k,j,i) = 0.0;
	      phydro->u(IM2,k,j,i) = 0.0;
        phydro->u(IM3,k,j,i) = 0.0; // non-rotating

        //set the energy (without magnetic energy)
	      phydro->u(IEN,k,j,i) = pres/(gamma_gas-1); // assuming gamma law EoS
	      phydro->u(IEN,k,j,i) += 0.5*(SQR(phydro->u(IM1,k,j,i))+SQR(phydro->u(IM2,k,j,i))
				                            + SQR(phydro->u(IM3,k,j,i)))/phydro->u(IDN,k,j,i);

        // set the scalar (conserved variables)
	      if(r<rstar_initial){
	        pscalars->s(0,k,j,i) = 1.0*phydro->u(IDN,k,j,i);
	      }else{
	        pscalars->s(0,k,j,i) = 1.e-30*phydro->u(IDN,k,j,i); // set small non-zero value
	      }
      }
    }
  } // end loop over cells
  return;
} // end ProblemGenerator

//======================================================================================
//! \fn void MeshBlock::UserWorkInLoop(void)
//  \brief Function called once every time step for user-defined work.
//======================================================================================
void MeshBlock::UserWorkInLoop() {
  Real time = pmy_mesh->time;
  Real dt = pmy_mesh->dt;
  Real tau;

  // get damping timescale
  if (time<t_relax) {
    if (time<0.2*t_relax) {
      tau = tau_relax;
    } else {
      tau = tau_relax*pow(10,2*(t_relax-time)/(0.8*t_relax));
    }
    if (Globals::my_rank==0){
      std::cout << "Relaxing: tau_damp ="<<tau<<std::endl;
    }
  }

  // damp the velocity
  if (time<t_relax) {
    for (int k=ks; k<=ke; ++k) {
      Real ph = pcoord->x3v(k);
      for (int j=js; j<=je; ++j) {
        Real th = pcoord->x2v(j);
        for (int i=is; i<=ie; ++i){
          Real r = pcoord->x1v(i);

          Real den = phydro->u(IDN,k,j,i);

          // velocities
          Real vr  =  phydro->u(IM1,k,j,i) / den;
	        Real vth =  phydro->u(IM2,k,j,i) / den;
	        Real vph =  phydro->u(IM3,k,j,i) / den;

          // damping terms
          Real src_r  = -dt*den*vr/tau;
          Real src_th = -dt*den*vth/tau;
          Real src_ph = -dt*den*vph/tau;

          // add into momentums
          phydro->u(IM1,k,j,i) += src_r;
          phydro->u(IM2,k,j,i) += src_th;
          phydro->u(IM3,k,j,i) += src_ph;

          // add into energy
          phydro->u(IEN,k,j,i) += src_r*vr+src_th*vth+src_ph*vph;
        }
      }
    }// end loop over cells
  }



}

//========================================================================================
//! \fn void Mesh::UserWorkInLoop() 
//  \brief Function called once every time step for user-defined work.
//========================================================================================

void Mesh::UserWorkInLoop() {
  
  Mesh *pm = my_blocks(0)->pmy_mesh;
  if ((ncycle==0)) {
    SumGasOnParticleAccels(pm, xi, agas1i, agas2i);

    // half step back
    ParticleAccels(xi,vi,ai);
    kick(-0.5*dt,xi,vi,ai);
  }
  
  if (Globals::my_rank==0 and time>t_relax){
    for (int i=1; i<=n_particle_substeps; ++i) {
        Real d = sqrt(xi[0]*xi[0] + xi[1]*xi[1] + xi[2]*xi[2]);
        // when the secondary reaches the inner boundary, stop evolve the secondary
        if (d<1.01*separation_stop_min) break;

	      // add the particle acceleration to ai
	      ParticleAccels(xi,vi,ai);
	      // advance the particle
	      particle_step(dt/n_particle_substeps,xi,vi,ai);
    }
  }

#ifdef MPI_PARALLEL
  // broadcast the position update from proc zero
  MPI_Bcast(xi,3,MPI_ATHENA_REAL,0,MPI_COMM_WORLD);
  MPI_Bcast(vi,3,MPI_ATHENA_REAL,0,MPI_COMM_WORLD);
#endif

  // sum the enclosed mass profile for monopole gravity
  if (ncycle%update_grav_every == 0) {
    SumMencProfile(pm,menc);
    if (Globals::my_rank == 0 ){
      std::cout << "enclosed mass updated... Menc(r=rstar_initial) = " << Interpolate1DArray(logr,menc,log10(1.01*rstar_initial), NGRAVL) <<"\n";
    }
  }

  if (time>t_relax) {
    SumGasOnParticleAccels(pm, xi, agas1i, agas2i);
  }

   // write the output to the trackfile
  if (time >= trackfile_next_time) {
    WritePMTrackfile(pm);
  }
}

namespace {
Real Interpolate1DArray(Real *x, Real *y, Real x0, int length) { 
  // check the lower bound
  if(x[0] >= x0){
    return y[0];
  }
  // check the upper bound
  if(x[length-1] <= x0){
    return y[length-1];
  }

  int i=0;
  while (x0>x[i+1]) i++;
  
  // if in the interior, do a linear interpolation
  if (x[i+1] >= x0){ 
    Real dx =  (x[i+1]-x[i]);
    Real d = (x0 - x[i]);
    Real s = (y[i+1]-y[i]) /dx;
    return s*d + y[i];
  }
  // should never get here, -9999.9 represents an error
  return -9999.9;
}

void SumGasOnParticleAccels(Mesh *pm, Real (&xi)[3], Real (&ag1i)[3], Real (&ag2i)[3]) {

  Real m1 =  GM1/Ggrav;
  // start by setting accelerations / positions to zero
  for (int ii = 0; ii < 3; ii++) {
    ag1i[ii] = 0.0;
    ag2i[ii] = 0.0;
  }
  
  MeshBlock *pmb=pm->my_blocks(0);
  AthenaArray<Real> vol;
  
  // ghost cells are included to get self-consistent index
  int ncells1 = pmb->block_size.nx1 + 2*(NGHOST);
  vol.NewAthenaArray(ncells1);

  // Loop over MeshBlocks
  for (int b=0; b<pm->nblocal; ++b) {
    pmb = pm->my_blocks(b);
    Hydro *phyd = pmb->phydro;

    // Sum history variables over cells.  Note ghost cells are never included in sums
    for (int k=pmb->ks; k<=pmb->ke; ++k) {
      Real ph= pmb->pcoord->x3v(k);
      Real sin_ph = sin(ph);
      Real cos_ph = cos(ph);
      for (int j=pmb->js; j<=pmb->je; ++j) {
	      Real th= pmb->pcoord->x2v(j);
	      Real sin_th = sin(th);
	      Real cos_th = cos(th);
	      pmb->pcoord->CellVolume(k,j,pmb->is,pmb->ie,vol);
	      for (int i=pmb->is; i<=pmb->ie; ++i) {
	        // cell mass dm
	        Real dm = vol(i) * phyd->u(IDN,k,j,i);
	  	  
	        Real r = pmb->pcoord->x1v(i);

	        // spherical polar coordinates, get local cartesian           
	        Real x = r*sin_th*cos_ph;
	        Real y = r*sin_th*sin_ph;
	        Real z = r*cos_th;

	        Real d2 = sqrt(pow(x-xi[0], 2) +
			                  pow(y-xi[1], 2) +
			                  pow(z-xi[2], 2) );

	        // if we're on the innermost zone of the innermost block, assuming reflecting bc
	        if((pmb->pbval->block_bcs[BoundaryFace::inner_x1] == BoundaryFlag::reflect) && (i==pmb->is)) {
	          // inner-r face area of cell i
	          Real dA = pmb->pcoord->GetFace1Area(k,j,i);
	  	 
	          // spherical velocities
	          Real vr =  phyd->u(IM1,k,j,i) / phyd->u(IDN,k,j,i);
	          Real vth =  phyd->u(IM2,k,j,i) / phyd->u(IDN,k,j,i);
	          Real vph =  phyd->u(IM3,k,j,i) / phyd->u(IDN,k,j,i);

	          // get the cartesian velocities from the spherical (vector)
	          Real vgas[3];
	          vgas[0] = sin_th*cos_ph*vr + cos_th*cos_ph*vth - sin_ph*vph;
	          vgas[1] = sin_th*sin_ph*vr + cos_th*sin_ph*vth + cos_ph*vph;
	          vgas[2] = cos_th*vr - sin_th*vth;

	          Real dAvec[3];
	          dAvec[0] = dA*(x/r);
	          dAvec[1] = dA*(y/r);
	          dAvec[2] = dA*(z/r);

	          // pressure terms (surface force/m1)
	          // note: not including the surface forces because the BC doesn't provide an appropriate counterterm
	          //ag1i[0] +=  -phyd->w(IPR,k,j,i)*dAvec[0]/m1;
	          //ag1i[1] +=  -phyd->w(IPR,k,j,i)*dAvec[1]/m1;
	          //ag1i[2] +=  -phyd->w(IPR,k,j,i)*dAvec[2]/m1;

	          // momentum flux terms
	          Real dAv = vgas[0]*dAvec[0] + vgas[1]*dAvec[1] + vgas[2]*dAvec[2];
	          ag1i[0] += phyd->u(IDN,k,j,i)*dAv*vgas[0]/m1;
	          ag1i[1] += phyd->u(IDN,k,j,i)*dAv*vgas[1]/m1;
	          ag1i[2] += phyd->u(IDN,k,j,i)*dAv*vgas[2]/m1;
	        }

	        // gravitational accels in cartesian coordinates
          Real d1c=pow(r,3);
	        ag1i[0] += Ggrav*dm/d1c * x;
	        ag1i[1] += Ggrav*dm/d1c * y;
	        ag1i[2] += Ggrav*dm/d1c * z;
	  
	        ag2i[0] += Ggrav*dm * fspline(d2,rsoft2) * (x-xi[0]);
	        ag2i[1] += Ggrav*dm * fspline(d2,rsoft2) * (y-xi[1]);
	        ag2i[2] += Ggrav*dm * fspline(d2,rsoft2) * (z-xi[2]);
	  
	      }
      }
    }//end loop over cells
  }//end loop over meshblocks

#ifdef MPI_PARALLEL
  // sum over all ranks
  if (Globals::my_rank == 0) {
    MPI_Reduce(MPI_IN_PLACE, ag1i, 3, MPI_ATHENA_REAL, MPI_SUM, 0,MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, ag2i, 3, MPI_ATHENA_REAL, MPI_SUM, 0,MPI_COMM_WORLD);
  } else {
    MPI_Reduce(ag1i,ag1i,3, MPI_ATHENA_REAL, MPI_SUM, 0,MPI_COMM_WORLD);
    MPI_Reduce(ag2i,ag2i,3, MPI_ATHENA_REAL, MPI_SUM, 0,MPI_COMM_WORLD);
  }

  // and broadcast the result
  MPI_Bcast(ag1i,3,MPI_ATHENA_REAL,0,MPI_COMM_WORLD);
  MPI_Bcast(ag2i,3,MPI_ATHENA_REAL,0,MPI_COMM_WORLD);
#endif    
} // end SumGasOnParticleAccels

void ParticleAccels(Real (&xi)[3],Real (&vi)[3],Real (&ai)[3]) {

  Real d = sqrt(xi[0]*xi[0] + xi[1]*xi[1] + xi[2]*xi[2]);
  Real v = sqrt(vi[0]*vi[0] + vi[1]*vi[1] + vi[2]*vi[2]);

  for (int i = 0; i < 3; i++) {
    // fill in the accelerations for the orbiting frame
    ai[i] = - GM1/pow(d,3) * xi[i] - GM2/pow(d,3) * xi[i];

    // add the gas acceleration to ai
    // the 2-gas interaction is updated every step, which is changed every substep, be careful
    ai[i] += -agas1i[i] + agas2i[i]; 

    if (d>=d_insp) {
      // add the artificial inspiral friction
      ai[i] += -Ggrav * mstar_initial * vi[i] /2/d/d/v/v * v_insp;
    }
  }

  
}

void particle_step(Real dt,Real (&xi)[3],Real (&vi)[3],Real (&ai)[3]) {
  // Leapfrog algorithm (KDK)

  // kick a full step
  kick(dt,xi,vi,ai);

  // drift a full step
  drift(dt,xi,vi,ai);
}

// kick the velocities dt using the accelerations given in ai
void kick(Real dt,Real (&xi)[3],Real (&vi)[3],Real (&ai)[3]) {
  for (int i = 0; i < 3; i++){
    vi[i] += dt*ai[i];
  }
}

// drift the velocities dt using the velocities given in vi
void drift(Real dt,Real (&xi)[3],Real (&vi)[3],Real (&ai)[3]) {
  for (int i = 0; i < 3; i++){
    xi[i] += dt*vi[i];
  }
}

Real fspline(Real r, Real eps) {
  // Hernquist & Katz 1989 spline kernel F=-GM r f(r,e) EQ A2
  Real u = r/eps;
  Real u2 = u*u;

  if (u<1.0){
    return pow(eps,-3) * (4./3. - 1.2*pow(u,2) + 0.5*pow(u,3) );
  } else if(u<2.0){
    return pow(r,-3) * (-1./15. + 8./3.*pow(u,3) - 3.*pow(u,4) + 1.2*pow(u,5) - 1./6.*pow(u,6));
  } else{
    return pow(r,-3);
  }

} // end spline function

void WritePMTrackfile(Mesh *pm){
  
  if (Globals::my_rank == 0) {
    std::string fname;
    fname.assign("pm_trackfile.dat");
    
    // open file for output
    FILE *pfile;
    std::stringstream msg;
    if((pfile = fopen(fname.c_str(),"a")) == NULL){
      msg << "### FATAL ERROR in function [WritePMTrackfile]" << std::endl
          << "Output file '" << fname << "' could not be opened";
      throw std::runtime_error(msg.str().c_str());
    }
  
    if (trackfile_number==0) {
      fprintf(pfile,"#    ncycle                  ");
      fprintf(pfile,"time                ");
      fprintf(pfile,"dt                  ");
      fprintf(pfile,"m1                  ");
      fprintf(pfile,"m2                  ");
      fprintf(pfile,"menv                ");
      fprintf(pfile,"x                   ");
      fprintf(pfile,"y                   ");
      fprintf(pfile,"z                   ");
      fprintf(pfile,"vx                  ");
      fprintf(pfile,"vy                  ");
      fprintf(pfile,"vz                  ");
      fprintf(pfile,"agas1x              ");
      fprintf(pfile,"agas1y              ");
      fprintf(pfile,"agas1z              ");
      fprintf(pfile,"agas2x              ");
      fprintf(pfile,"agas2y              ");
      fprintf(pfile,"agas2z              ");
      fprintf(pfile,"\n");
    }

    // write the data line
    fprintf(pfile,"%20i",pm->ncycle);
    fprintf(pfile,"%20.6e",pm->time);
    fprintf(pfile,"%20.6e",pm->dt);
    fprintf(pfile,"%20.6e",GM1/Ggrav);
    fprintf(pfile,"%20.6e",GM2/Ggrav);
    fprintf(pfile,"%20.6e",Interpolate1DArray(logr,menc,log10(1.01*rstar_initial), NGRAVL)-GM1/Ggrav);
    fprintf(pfile,"%20.6e",xi[0]);
    fprintf(pfile,"%20.6e",xi[1]);
    fprintf(pfile,"%20.6e",xi[2]);
    fprintf(pfile,"%20.6e",vi[0]);
    fprintf(pfile,"%20.6e",vi[1]);
    fprintf(pfile,"%20.6e",vi[2]);
    fprintf(pfile,"%20.6e",agas1i[0]);
    fprintf(pfile,"%20.6e",agas1i[1]);
    fprintf(pfile,"%20.6e",agas1i[2]);
    fprintf(pfile,"%20.6e",agas2i[0]);
    fprintf(pfile,"%20.6e",agas2i[1]);
    fprintf(pfile,"%20.6e",agas2i[2]);
    fprintf(pfile,"\n");

    // close the file
    fclose(pfile);  
  } // end rank==0

  // increment counters
  trackfile_number++;
  trackfile_next_time += trackfile_dt;
 
  return;
}

void SumMencProfile(Mesh *pm, Real (&menc)[NGRAVL]) {

  Real m1 =  GM1/Ggrav;
  // start by setting enclosed mass at each radius to zero
  for (int ii = 0; ii <NGRAVL; ii++){
    menc[ii] = 0.0;
  }
  
  MeshBlock *pmb=pm->my_blocks(0);
  AthenaArray<Real> vol;
  
  int ncells1 = pmb->block_size.nx1 + 2*(NGHOST);
  vol.NewAthenaArray(ncells1);

  // Loop over MeshBlocks
  for (int b=0; b<pm->nblocal; ++b) {
    pmb = pm->my_blocks(b);
    Hydro *phyd = pmb->phydro;

    // Sum history variables over cells.  Note ghost cells are never included in sums
    for (int k=pmb->ks; k<=pmb->ke; ++k) {
      for (int j=pmb->js; j<=pmb->je; ++j) {
	      pmb->pcoord->CellVolume(k,j,pmb->is,pmb->ie,vol);
	      for (int i=pmb->is; i<=pmb->ie; ++i) {
	        // cell mass dm
	        Real dm = vol(i) * phyd->u(IDN,k,j,i);
	        Real logr_cell = log10( pmb->pcoord->x1v(i) );

	        // loop over radii in profile
	        for (int ii = 0; ii <NGRAVL; ii++) {
	          if (logr_cell < logr[ii]) {
	            menc[ii] += dm;
	          }
	        } // end loop over radii
	      }
      }
    }//end loop over cells
  }//end loop over meshblocks

#ifdef MPI_PARALLEL
  // sum over all ranks, add m1
  if (Globals::my_rank == 0) {
    for (int ii = 0; ii <NGRAVL; ii++){
      menc[ii] += m1;
    }
    MPI_Reduce(MPI_IN_PLACE, menc, NGRAVL, MPI_ATHENA_REAL, MPI_SUM, 0, MPI_COMM_WORLD);
  } else {
    MPI_Reduce(menc, menc, NGRAVL, MPI_ATHENA_REAL, MPI_SUM, 0, MPI_COMM_WORLD);
  }
  
  // and broadcast the result
  MPI_Bcast(menc, NGRAVL, MPI_ATHENA_REAL, 0, MPI_COMM_WORLD);
#endif  
}
} // end namespace

//--------------------------------------------------------------------------------------
//! \fn void OutflowOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
//                         FaceField &b, Real time, Real dt,
//                         int is, int ie, int js, int je, int ks, int ke)
//  \brief OUTFLOW boundary conditions, outer x1 boundary

void DiodeOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
		    FaceField &b, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh) {
  // copy hydro variables into ghost zones, don't allow inflow
  for (int n=0; n<(NHYDRO); ++n) {
    if (n==(IVX)) {
      for (int k=ks; k<=ke; ++k) {
	      for (int j=js; j<=je; ++j) {
          #pragma simd
	        for (int i=1; i<=(NGHOST); ++i) {
	          prim(IVX,k,j,ie+i) =  std::max( 0.0, prim(IVX,k,j,(ie-i+1)) );  // positive velocities only
	        }
	      }
      }
    } else {
      for (int k=ks; k<=ke; ++k) {
	      for (int j=js; j<=je; ++j) {
          #pragma simd
	        for (int i=1; i<=(NGHOST); ++i) {
	          prim(n,k,j,ie+i) = prim(n,k,j,(ie-i+1));
	        }
	      }
      }
    }
  }


  // copy face-centered magnetic fields into ghost zones
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        #pragma simd
	      for (int i=1; i<=(NGHOST); ++i) {
	        b.x1f(k,j,(ie+i+1)) = b.x1f(k,j,(ie+1));
	      }
      }
    }

    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je+1; ++j) {
        #pragma simd
	      for (int i=1; i<=(NGHOST); ++i) {
	        b.x2f(k,j,(ie+i)) = b.x2f(k,j,ie);
	      }
      }
    }

    for (int k=ks; k<=ke+1; ++k) {
      for (int j=js; j<=je; ++j) {
        #pragma simd
	      for (int i=1; i<=(NGHOST); ++i) {
	        b.x3f(k,j,(ie+i)) = b.x3f(k,j,ie);
	      }
      }
    }
  }

  return;
}

// Source Function for two point masses
void TwoPointMass(MeshBlock *pmb, const Real time, const Real dt,
		  const AthenaArray<Real> &prim, const AthenaArray<Real> *flux,
		  const AthenaArray<Real> &prim_scalar, const AthenaArray<Real> &bcc,
		  AthenaArray<Real> &cons, AthenaArray<Real> &cons_scalar) {
  // Gravitational acceleration from orbital motion
  for (int k=pmb->ks; k<=pmb->ke; k++) {
    Real ph= pmb->pcoord->x3v(k);
    Real sin_ph = sin(ph);
    Real cos_ph = cos(ph);
    for (int j=pmb->js; j<=pmb->je; j++) {
      Real th= pmb->pcoord->x2v(j);
      Real sin_th = sin(th);
      Real cos_th = cos(th);
      for (int i=pmb->is; i<=pmb->ie; i++) {
	      Real r = pmb->pcoord->x1v(i);
	
	      // current position of the secondary
	      Real x_2 = xi[0];
	      Real y_2 = xi[1];
	      Real z_2 = xi[2];
	      Real d12 = pow(xi[0]*xi[0] + xi[1]*xi[1] + xi[2]*xi[2], 0.5);

        // spherical polar coordinates, get local cartesian           
	      Real x = r*sin_th*cos_ph;
	      Real y = r*sin_th*sin_ph;
	      Real z = r*cos_th;
  
	      Real d2  = sqrt(pow(x-x_2, 2) +
			                  pow(y-y_2, 2) +
			                  pow(z-z_2, 2) );

        //
	      //  COMPUTE ACCELERATIONS
        //
        // a averaged version is in pointmass.cpp
        Real a_r1 = -GM1*pmb->pcoord->coord_src1_i_(i)/r;
	      Real a_sg = -Ggrav*Interpolate1DArray(logr, menc, log10(r), NGRAVL) *
                    pmb->pcoord->coord_src1_i_(i)/r-a_r1;
        // Real a_r1 = 0;
        // Real a_sg = -Ggrav*Interpolate1DArray(logr, menc, log10(r), NGRAVL) / pow(r,2);
	
	      // PM2 gravitational accels in cartesian coordinates
	      Real a_x = - GM2 * fspline(d2,rsoft2) * (x-x_2);   
	      Real a_y = - GM2 * fspline(d2,rsoft2) * (y-y_2);  
	      Real a_z = - GM2 * fspline(d2,rsoft2) * (z-z_2);
	
        // add the correction for the orbiting frame (relative to the COM
        Real d12c = pow(d12,3);
	      a_x += -  GM2 / d12c * x_2;
	      a_y += -  GM2 / d12c * y_2;
	      a_z += -  GM2 / d12c * z_2;
	
	      // add the gas acceleration of the frame of ref
	      a_x += -agas1i[0];
	      a_y += -agas1i[1];
	      a_z += -agas1i[2];    
		
	      // convert back to spherical
	      Real a_r  = sin_th*cos_ph*a_x + sin_th*sin_ph*a_y + cos_th*a_z;
	      Real a_th = cos_th*cos_ph*a_x + cos_th*sin_ph*a_y - sin_th*a_z;
	      Real a_ph = -sin_ph*a_x + cos_ph*a_y;
	
	      // add the PM1 accel and self-gravity
	      a_r += a_r1+a_sg;
	
	      //
	      // ADD SOURCE TERMS TO THE GAS MOMENTA/ENERGY
	      //
	      Real den = prim(IDN,k,j,i);
	
	      Real src_1 = dt*den*a_r; 
	      Real src_2 = dt*den*a_th;
	      Real src_3 = dt*den*a_ph;

	      // add the source term to the momenta  (source = - den * a * dt)
	      cons(IM1,k,j,i) += src_1;
	      cons(IM2,k,j,i) += src_2;
	      cons(IM3,k,j,i) += src_3;
	
	      // update the energy (source = - den * dt * v dot a)
	      // cons(IEN,k,j,i) += src_1*prim(IVX,k,j,i) + src_2*prim(IVY,k,j,i) + src_3*prim(IVZ,k,j,i);
        cons(IEN,k,j,i) += dt*a_r  * 0.5*(flux[X1DIR](IDN,k,j,i) + flux[X1DIR](IDN,k,j,i+1));
	      cons(IEN,k,j,i) += dt*a_th * 0.5*(flux[X2DIR](IDN,k,j,i) + flux[X2DIR](IDN,k,j+1,i)); 
	      cons(IEN,k,j,i) += dt*a_ph * 0.5*(flux[X3DIR](IDN,k,j,i) + flux[X3DIR](IDN,k+1,j,i));
        // cons(IEN,k,j,i) += src_2*prim(IVY,k,j,i) + src_3*prim(IVZ,k,j,i);
      }
    }
  } // end loop over cells
} // end source function