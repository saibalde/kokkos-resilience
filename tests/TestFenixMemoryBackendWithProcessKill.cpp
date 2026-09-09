/*
 *
 *                        Kokkos v. 3.0
 *       Copyright (2020) National Technology & Engineering
 *               Solutions of Sandia, LLC (NTESS).
 *
 * Under the terms of Contract DE-NA0003525 with NTESS,
 * the U.S. Government retains certain rights in this software.
 *
 * Kokkos is licensed under 3-clause BSD terms of use:
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 * 1. Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the Corporation nor the names of the
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Questions? Contact Christian R. Trott (crtrott@sandia.gov)
 */
#include <string>
#include <random>

#include <signal.h>
#include <unistd.h>
#include <fenix.h>

#include <Kokkos_Core.hpp>

#include "resilience/AutomaticCheckpoint.hpp"
#include "resilience/backend/FenixBackend.hpp"
#include "resilience/context/MPIContext.hpp"

template <typename ExecSpace, typename Layout, typename Context>
static int test(Context& ctx, std::size_t dimx, std::size_t dimy, bool kill_process = false) {
  using exec_space   = ExecSpace;
  using memory_space = typename exec_space::memory_space;
  using view_type    = KokkosResilience::View<double**, Layout, memory_space>;

  ctx.backend().reset();

  auto e  = std::default_random_engine(0);
  auto ud = std::uniform_real_distribution<double>(-10.0, 10.0);

  view_type main_view("main_view", dimx, dimy);
  auto host_mirror = Kokkos::create_mirror_view(main_view);

  for (std::size_t x = 0; x < dimx; ++x) {
    for (std::size_t y = 0; y < dimy; ++y) {
      host_mirror(x, y) = ud(e);
    }
  }

  Kokkos::deep_copy(main_view, host_mirror);

  Kokkos::fence();

  KokkosResilience::checkpoint(ctx, "test_checkpoint", 0, [=]() {
    Kokkos::parallel_for(
        Kokkos::RangePolicy<exec_space>(0, dimx), KOKKOS_LAMBDA(int i) {
          for (std::size_t j = 0; j < dimy; ++j) main_view(i, j) -= 1.0;
        });
  });

  // Clobber main_view, should be reloaded at checkpoint
  Kokkos::parallel_for(
      Kokkos::RangePolicy<exec_space>(0, dimx), KOKKOS_LAMBDA(int i) {
        for (std::size_t j = 0; j < dimy; ++j) main_view(i, j) = 0.0;
      });

  // Clobber host view just in case
  for (std::size_t x = 0; x < dimx; ++x) {
    for (std::size_t y = 0; y < dimy; ++y) {
      host_mirror(x, y) = 0.0;
    }
  }

  // kill process
  if (kill_process) {
    pid_t pid = getpid();
    kill(pid, SIGTERM);
  }

  // The lambda shouldn't be executed, instead recovery should start
  KokkosResilience::checkpoint(ctx, "test_checkpoint", 0, [main_view]() {
    std::cout << "========== !!! THIS PRINT STATEMENT SHOULD HAVE BEEN SKIPPED !!! ==========" << std::endl;
  });

  Kokkos::fence();

  Kokkos::deep_copy(host_mirror, main_view);

  e.seed(0);

  for (std::size_t x = 0; x < dimx; ++x) {
    for (std::size_t y = 0; y < dimy; ++y) {
      if (host_mirror(x, y) != ud(e) - 1.0) {
        return -1;
      }
    }
  }

  return 0;
}

template <typename ExecSpace>
void test_fenix_memory_backend(MPI_Comm res_comm) {
  int res_rank;
  MPI_Comm_rank(res_comm, &res_rank);

  {
    using namespace std::string_literals;
    KokkosResilience::Config cfg;
    cfg["backend"].set("fenix"s);
    KokkosResilience::MPIContext<KokkosResilience::FenixMemoryBackend> ctx(res_comm, cfg);

    for (std::size_t dimx = 1; dimx < 5; ++dimx) {
      for (std::size_t dimy = 1; dimy < 5; ++dimy) {
        if (dimx == 2 && dimy == 3) {
          // flag process 0 of the resilient communicator to be killed
          bool kill_process = false; // (res_rank == 0);
          test<ExecSpace, Kokkos::LayoutRight>(ctx, dimx, dimy, kill_process);
        } else {
          test<ExecSpace, Kokkos::LayoutRight>(ctx, dimx, dimy);
        }
        test<ExecSpace, Kokkos::LayoutLeft>(ctx, dimx, dimy);
      }
    }
  }
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);

  int mpi_size;
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  int mpi_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

  if (mpi_size < 3) {
    if (mpi_rank == 0) {
      std::cerr << "fenix in-memory data recovery with process kill needs at least 3 ranks\n" << std::flush;
    }
    MPI_Abort(MPI_COMM_WORLD, -1);
  }

  int role;
  MPI_Comm res_comm;
  int spare_ranks = 1;
  int status;
  Fenix_Init(&role, MPI_COMM_WORLD, &res_comm, &argc, &argv, spare_ranks, &status);

  if (status != FENIX_SUCCESS) {
    if (mpi_rank == 0) {
      std::cerr << "failed to initialize fenix\n" << std::flush;
    }
    MPI_Abort(MPI_COMM_WORLD, -1);
  }

  int res_rank;
  MPI_Comm_rank(res_comm, &res_rank);

  if (argc != 2) {
    if (res_rank == 0) {
      std::cerr << "usage: " << argv[0] << " <serial|openmp|cuda|hpx>\n" << std::flush;
    }
    MPI_Abort(res_comm, -1);
  }

  const std::string backend = std::string(argv[1]);

  if (backend == "serial") {
#ifdef KOKKOS_ENABLE_SERIAL
    test_fenix_memory_backend<Kokkos::Serial>(res_comm);
#else
    if (res_rank == 0) {
      std::cerr << "serial backend is not supported in current Kokkos build\n" << std::flush;
    }
    MPI_Abort(res_comm, -1);
#endif
  } else if (backend == "openmp") {
#ifdef KOKKOS_ENABLE_OPENMP
    test_fenix_memory_backend<Kokkos::OpenMP>(res_comm);
#else
    if (res_rank == 0) {
      std::cerr << "openmp backend is not supported in current Kokkos build\n" << std::flush;
    }
    MPI_Abort(res_comm, -1);
#endif
  } else if (backend == "cuda") {
#ifdef KOKKOS_ENABLE_CUDA
    test_fenix_memory_backend<Kokkos::Cuda>(res_comm);
#else
    if (res_rank == 0) {
      std::cerr << "cuda backend is not supported in current Kokkos build\n" << std::flush;
    }
    MPI_Abort(res_comm, -1);
#endif
  } else if (backend == "hpx") {
#ifdef KOKKOS_ENABLE_HPX
    test_fenix_memory_backend<Kokkos::Experimental::HPX>(res_comm);
#else
    if (res_rank == 0) {
      std::cerr << "hpx backend is not supported in current Kokkos build\n" << std::flush;
    }
    MPI_Abort(res_comm, -1);
#endif
  } else {
    if (res_rank == 0) {
      std::cerr << "unknown backend " << backend << "\n" << std::flush;
    }
    MPI_Abort(res_comm, -1);
  }

  Fenix_Finalize();

  Kokkos::finalize();
  MPI_Finalize();

  return 0;
}
