//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file collision.cpp
//! \brief Problem generator for collision of plasma slabs.

// C++ headers
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <random>
#include <sstream>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../field/field.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "../scalars/scalars.hpp"
#include "../utils/utils.hpp"

#ifdef FFT
#include <fftw3.h>
#endif

namespace {
Real den0, v0, b0;
Real a0, ky_over_2pi, kz_over_2pi;
Real ky, kz;

bool is_den_turb;
Real den_turb;
int den_seed;
Real den_ps_index;
int den_nlow, den_nhigh;
Real den_center1, den_center2, den_center3;
Real den_sigma1, den_sigma2, den_sigma3;
Real den_floor;

std::int64_t GetKComp(int idx, int disp, int nx) {
  return (idx + disp)
         - static_cast<std::int64_t>(2 * (idx + disp) / nx) * nx;
}

Real GaussianAxisWeight(Real x, Real center, Real sigma) {
  if (sigma <= 0.0) {
    return 1.0;
  }
  Real arg = (x - center) / sigma;
  return std::exp(-0.5 * arg * arg);
}

Real GaussianWindow(Real x1, Real x2, Real x3) {
  return GaussianAxisWeight(x1, den_center1, den_sigma1)
         * GaussianAxisWeight(x2, den_center2, den_sigma2)
         * GaussianAxisWeight(x3, den_center3, den_sigma3);
}
}  // namespace

//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//! \brief Initialize problem-specific data in mesh class.
//========================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin) {
  den0 = pin->GetOrAddReal("problem", "den0", 1.0);
  v0 = pin->GetOrAddReal("problem", "v0", 1.0);
  b0 = pin->GetOrAddReal("problem", "b0", 1.0);

  a0 = pin->GetOrAddReal("problem", "a0", 1.0);
  ky_over_2pi = pin->GetOrAddReal("problem", "ky_over_2pi", 1.0);
  kz_over_2pi = pin->GetOrAddReal("problem", "kz_over_2pi", 1.0);
  is_den_turb = pin->GetOrAddBoolean("problem", "isDenTurb", false);
  den_turb = pin->GetOrAddReal("problem", "denTurb", 0.0);

  ky = ky_over_2pi * TWO_PI;
  kz = kz_over_2pi * TWO_PI;
  is_den_turb = is_den_turb && (den_turb != 0.0);

  if (!is_den_turb) {
    return;
  }

  if (ndim != 3) {
    std::stringstream msg;
    msg << "### FATAL ERROR in collision.cpp Mesh::InitUserMeshData" << std::endl
        << "The spectral density initializer requires a 3D mesh." << std::endl;
    ATHENA_ERROR(msg);
  }

  if (multilevel) {
    std::stringstream msg;
    msg << "### FATAL ERROR in collision.cpp Mesh::InitUserMeshData" << std::endl
        << "The spectral density initializer does not support SMR/AMR." << std::endl;
    ATHENA_ERROR(msg);
  }

  den_seed = pin->GetOrAddInteger("problem", "den_seed", 12345);
  den_ps_index = pin->GetOrAddReal("problem", "den_ps_index", 11.0/3.0);
  den_nlow = pin->GetOrAddInteger("problem", "den_nlow", 1);
  den_nhigh = pin->GetOrAddInteger(
      "problem", "den_nhigh",
      std::max(den_nlow + 1, std::min(mesh_size.nx1,
               std::min(mesh_size.nx2, mesh_size.nx3)) / 2));

  Real x1mid = 0.5 * (mesh_size.x1min + mesh_size.x1max);
  Real x2mid = 0.5 * (mesh_size.x2min + mesh_size.x2max);
  Real x3mid = 0.5 * (mesh_size.x3min + mesh_size.x3max);
  Real lx1 = mesh_size.x1max - mesh_size.x1min;
  Real lx2 = mesh_size.x2max - mesh_size.x2min;
  Real lx3 = mesh_size.x3max - mesh_size.x3min;

  den_center1 = pin->GetOrAddReal("problem", "den_center1", x1mid);
  den_center2 = pin->GetOrAddReal("problem", "den_center2", x2mid);
  den_center3 = pin->GetOrAddReal("problem", "den_center3", x3mid);
  den_sigma1 = pin->GetOrAddReal("problem", "den_sigma1", 0.25 * lx1);
  den_sigma2 = pin->GetOrAddReal("problem", "den_sigma2", 0.25 * lx2);
  den_sigma3 = pin->GetOrAddReal("problem", "den_sigma3", 0.25 * lx3);
  den_floor = pin->GetOrAddReal("problem", "den_floor", 1.0e-6 * den0);

  if (den_nlow < 0 || den_nhigh <= den_nlow) {
    std::stringstream msg;
    msg << "### FATAL ERROR in collision.cpp Mesh::InitUserMeshData" << std::endl
        << "Require den_nhigh > den_nlow >= 0." << std::endl;
    ATHENA_ERROR(msg);
  }

  if (den_floor <= 0.0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in collision.cpp Mesh::InitUserMeshData" << std::endl
        << "den_floor must be positive." << std::endl;
    ATHENA_ERROR(msg);
  }

  AllocateRealUserMeshDataField(1);
  ruser_mesh_data[0].NewAthenaArray(mesh_size.nx3, mesh_size.nx2, mesh_size.nx1);
  AthenaArray<Real> &density = ruser_mesh_data[0];
  density.ZeroClear();

#ifndef FFT
  std::stringstream msg;
  msg << "### FATAL ERROR in collision.cpp Mesh::InitUserMeshData" << std::endl
      << "The spectral density initializer requires Athena++ to be built with FFT."
      << std::endl;
  ATHENA_ERROR(msg);
#else
  std::int64_t ntot = static_cast<std::int64_t>(mesh_size.nx1)
      * static_cast<std::int64_t>(mesh_size.nx2)
      * static_cast<std::int64_t>(mesh_size.nx3);
  if (Globals::my_rank == 0) {
    fftw_complex *fourier_modes =
        fftw_alloc_complex(static_cast<std::size_t>(ntot));
    if (fourier_modes == nullptr) {
      std::stringstream msg;
      msg << "### FATAL ERROR in collision.cpp Mesh::InitUserMeshData" << std::endl
          << "Unable to allocate the Fourier density workspace." << std::endl;
      ATHENA_ERROR(msg);
    }

    for (std::int64_t idx = 0; idx < ntot; ++idx) {
      fourier_modes[idx][0] = 0.0;
      fourier_modes[idx][1] = 0.0;
    }

    std::mt19937_64 rng(static_cast<std::uint64_t>(den_seed));
    std::normal_distribution<Real> ndist(0.0, 1.0);
    std::uniform_real_distribution<Real> udist(0.0, 1.0);

    Real dkx = TWO_PI / lx1;
    Real dky = TWO_PI / lx2;
    Real dkz = TWO_PI / lx3;
    Real kref = std::min(dkx, std::min(dky, dkz));

    for (int k = 0; k < mesh_size.nx3; ++k) {
      std::int64_t nz = GetKComp(k, 0, mesh_size.nx3);
      Real kz_spec = nz * dkz;
      for (int j = 0; j < mesh_size.nx2; ++j) {
        std::int64_t ny = GetKComp(j, 0, mesh_size.nx2);
        Real ky_spec = ny * dky;
        for (int i = 0; i < mesh_size.nx1; ++i) {
          std::int64_t nx = GetKComp(i, 0, mesh_size.nx1);
          Real kx_spec = nx * dkx;
          Real kmag = std::sqrt(kx_spec * kx_spec + ky_spec * ky_spec
                                + kz_spec * kz_spec);
          Real nmag = kmag / kref;

          Real amp = 0.0;
          if (kmag > 0.0 && nmag >= den_nlow && nmag <= den_nhigh) {
            amp = std::pow(kmag, -0.5 * den_ps_index);
          }

          Real gauss = ndist(rng);
          Real phase = udist(rng) * TWO_PI;
          std::int64_t idx =
              (static_cast<std::int64_t>(k) * mesh_size.nx2 + j) * mesh_size.nx1 + i;
          fourier_modes[idx][0] = amp * gauss * std::cos(phase);
          fourier_modes[idx][1] = amp * gauss * std::sin(phase);
        }
      }
    }

    fftw_plan backward_plan = fftw_plan_dft_3d(
        mesh_size.nx3, mesh_size.nx2, mesh_size.nx1, fourier_modes,
        fourier_modes, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(backward_plan);

    Real inv_ntot = 1.0 / static_cast<Real>(ntot);
    for (int k = 0; k < mesh_size.nx3; ++k) {
      for (int j = 0; j < mesh_size.nx2; ++j) {
        for (int i = 0; i < mesh_size.nx1; ++i) {
          std::int64_t idx =
              (static_cast<std::int64_t>(k) * mesh_size.nx2 + j) * mesh_size.nx1 + i;
          density(k, j, i) = fourier_modes[idx][0] * inv_ntot;
        }
      }
    }

    fftw_destroy_plan(backward_plan);
    fftw_free(fourier_modes);
  }

#ifdef MPI_PARALLEL
  int ntot_mpi = mesh_size.nx1 * mesh_size.nx2 * mesh_size.nx3;
  MPI_Bcast(density.data(), ntot_mpi, MPI_ATHENA_REAL, 0, MPI_COMM_WORLD);
#endif

  Real raw_sum = 0.0;
  Real raw_sum2 = 0.0;
  for (int k = 0; k < mesh_size.nx3; ++k) {
    for (int j = 0; j < mesh_size.nx2; ++j) {
      for (int i = 0; i < mesh_size.nx1; ++i) {
        Real value = density(k, j, i);
        raw_sum += value;
        raw_sum2 += value * value;
      }
    }
  }

  Real raw_mean = raw_sum / static_cast<Real>(ntot);
  Real raw_var = raw_sum2 / static_cast<Real>(ntot) - raw_mean * raw_mean;
  raw_var = std::max(raw_var, static_cast<Real>(TINY_NUMBER));
  Real inv_raw_std = 1.0 / std::sqrt(raw_var);

  Real dx1 = (mesh_size.x1max - mesh_size.x1min) / static_cast<Real>(mesh_size.nx1);
  Real dx2 = (mesh_size.x2max - mesh_size.x2min) / static_cast<Real>(mesh_size.nx2);
  Real dx3 = (mesh_size.x3max - mesh_size.x3min) / static_cast<Real>(mesh_size.nx3);

  Real windowed_mean = 0.0;
  Real windowed_norm = 0.0;
  for (int k = 0; k < mesh_size.nx3; ++k) {
    Real x3 = mesh_size.x3min + (static_cast<Real>(k) + 0.5) * dx3;
    for (int j = 0; j < mesh_size.nx2; ++j) {
      Real x2 = mesh_size.x2min + (static_cast<Real>(j) + 0.5) * dx2;
      for (int i = 0; i < mesh_size.nx1; ++i) {
        Real x1 = mesh_size.x1min + (static_cast<Real>(i) + 0.5) * dx1;
        Real fluct = (density(k, j, i) - raw_mean) * inv_raw_std;
        Real window = GaussianWindow(x1, x2, x3);
        density(k, j, i) = fluct * window;
        windowed_mean += density(k, j, i);
        windowed_norm += window * window;
      }
    }
  }

  windowed_mean /= static_cast<Real>(ntot);
  Real windowed_sum2 = 0.0;
  for (int k = 0; k < mesh_size.nx3; ++k) {
    for (int j = 0; j < mesh_size.nx2; ++j) {
      for (int i = 0; i < mesh_size.nx1; ++i) {
        density(k, j, i) -= windowed_mean;
        windowed_sum2 += density(k, j, i) * density(k, j, i);
      }
    }
  }

  Real windowed_rms = std::sqrt(windowed_sum2
      / std::max(windowed_norm, static_cast<Real>(TINY_NUMBER)));
  windowed_rms = std::max(windowed_rms, static_cast<Real>(TINY_NUMBER));

  for (int k = 0; k < mesh_size.nx3; ++k) {
    for (int j = 0; j < mesh_size.nx2; ++j) {
      for (int i = 0; i < mesh_size.nx1; ++i) {
        Real rho = den0 + den_turb * density(k, j, i) / windowed_rms;
        density(k, j, i) = std::max(den_floor, rho);
      }
    }
  }
#endif  // FFT
}

//======================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//! \brief Problem generator for collision of plasma slabs.
//======================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  AthenaArray<Real> *density_data =
      (is_den_turb && pmy_mesh->ruser_mesh_data != nullptr)
          ? pmy_mesh->ruser_mesh_data : nullptr;

  for (int k = ks; k <= ke; ++k) {
    Real z = pcoord->x3v(k);
    int gk = static_cast<int>(loc.lx3) * block_size.nx3 + (k - ks);
    for (int j = js; j <= je; ++j) {
      Real y = pcoord->x2v(j);
      int gj = static_cast<int>(loc.lx2) * block_size.nx2 + (j - js);
      for (int i = is; i <= ie; ++i) {
        Real x = pcoord->x1v(i);
        int gi = static_cast<int>(loc.lx1) * block_size.nx1 + (i - is);

        phydro->u(IDN, k, j, i) = den0;
        if (density_data != nullptr) {
          phydro->u(IDN, k, j, i) = (*density_data)(gk, gj, gi);
        }

        if (x > a0 * std::sin(ky * y) * std::sin(kz * z)) {
          phydro->u(IM1, k, j, i) = -phydro->u(IDN, k, j, i) * v0;
        } else {
          phydro->u(IM1, k, j, i) = phydro->u(IDN, k, j, i) * v0;
        }
        phydro->u(IM2, k, j, i) = 0.0;
        phydro->u(IM3, k, j, i) = 0.0;

        if (x > a0 * std::sin(ky * y) * std::sin(kz * z)) {
          pscalars->s(0, k, j, i) = phydro->u(IDN, k, j, i);
        } else {
          pscalars->s(0, k, j, i) = 1.0e-30 * phydro->u(IDN, k, j, i);
        }
      }
    }
  }

  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k = ks; k <= ke; ++k) {
      for (int j = js; j <= je; ++j) {
        for (int i = is; i <= ie + 1; ++i) {
          pfield->b.x1f(k, j, i) = b0;
        }
      }
    }
    for (int k = ks; k <= ke; ++k) {
      for (int j = js; j <= je + 1; ++j) {
        for (int i = is; i <= ie; ++i) {
          pfield->b.x2f(k, j, i) = 0.0;
        }
      }
    }
    for (int k = ks; k <= ke + 1; ++k) {
      for (int j = js; j <= je; ++j) {
        for (int i = is; i <= ie; ++i) {
          pfield->b.x3f(k, j, i) = 0.0;
        }
      }
    }
  }
}
