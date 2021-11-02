#pragma once

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

#include "containers.cuh"
#include "hash.cuh"

// structure for count_min_table
struct count_min_pars {
  int width_bits;
  int hash_per_hash;
  uint64_t mask;
  uint32_t table_width;
  int table_rows;
  int bloom_width_mult;
};

// taken from ppsketchlib/src/gpu/sketch.cu
// countmin and binsign
// using unsigned long long int = uint64_t due to atomicCAS prototype
__host__ __device__ unsigned int probe(unsigned int *table, uint64_t hash_val,
                                       count_min_pars *pars, const int k,
                                       const bool update, const bool bloom) {
  unsigned int min_count = UINT32_MAX;
  bool found = true;
  for (int hash_nr = 0; hash_nr < pars->table_rows;
       hash_nr += pars->hash_per_hash) {
    uint64_t current_hash = hash_val;
    for (uint i = 0; i < pars->hash_per_hash; i++) {
      uint32_t hash_val_masked = current_hash & pars->mask;
      unsigned int *cell_ptr = table + hash_val_masked;
      if (!bloom) {
        cell_ptr += (hash_nr + i) * pars->table_width;
      }
      unsigned int cell_count;
      if (update) {
#ifdef __CUDA_ARCH__
        cell_count = atomicInc(cell_ptr, UINT32_MAX) + 1;
#else
        cell_count = ++(*cell_ptr);
#endif
      } else {
        cell_count = *cell_ptr;
      }

      if (bloom) {
        found &= cell_count > 0;
      }
      if (cell_count < min_count) {
        min_count = cell_count;
      }
      current_hash = current_hash >> pars->width_bits;
    }
    hash_val = shifthash(hash_val, k, hash_nr / 2);
  }
  if (bloom) {
    min_count = found;
  }
  return (min_count);
}

__device__ size_t copy_reads_to_shared(char *&read_seq,
                                      const size_t read_length,
                                      const size_t n_reads) {
  const size_t bank_bytes = 8;
  const size_t read_length_bank_pad +=
      read_length % bank_bytes ? bank_bytes - read_length % bank_bytes : 0;
  extern __shared__ char read_shared[];
  auto block = cooperative_groups::this_thread_block();
  size_t n_reads_in_block = blockDim.x;
  if (blockDim.x * (blockIdx.x + 1) > n_reads) {
    n_reads_in_block = n_reads - blockDim.x * blockIdx.x;
  }
  for (int read_idx = 0; read_idx < n_reads_in_block; ++read_idx) {
    // Copies one read into shared
    cooperative_groups::memcpy_async(
        block, read_shared + read_idx * read_length_bank_pad,
        read_seq + read_length * (blockIdx.x * blockDim.x + read_idx),
        sizeof(char) * read_length);
  }
  cooperative_groups::wait(block);
  return read_length_bank_pad;
}

__global__ void fill_kmers(char *read_seq, const size_t n_reads,
                           const size_t read_length, const int k,
                           unsigned int *countmin_table, count_min_pars *pars,
                           const bool use_rc) {
  uint64_t fhVal, rhVal, hVal;
  size_t read_stride = copy_reads_to_shared(read_seq, read_length, n_reads);
  int read_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (read_index < n_reads) {
    // Get first valid k-mer
    if (use_rc) {
      NTC64(read_seq + threadIdx.x * read_stride, k, fhVal, rhVal, hVal, 1);
      probe(countmin_table, hVal, pars, k, true, false);
    } else {
      NT64(read_seq + threadIdx.x * read_stride, k, fhVal, 1);
      probe(countmin_table, hVal, pars, k, true, false);
    }

    // Roll through remaining k-mers in the read
    for (int pos = 0; pos < read_length - k; pos++) {
      fhVal = NTF64(fhVal, k, read_seq[threadIdx.x * read_stride + pos],
                    read_seq[threadIdx.x * read_stride + pos + k]);
      if (use_rc) {
        rhVal = NTR64(fhVal, k, read_seq[threadIdx.x * read_stride + pos],
                      read_seq[threadIdx.x * read_stride + pos + k]);
        hVal = (rhVal < fhVal) ? rhVal : fhVal;
        probe(countmin_table, hVal, pars, k, true, false);
      } else {
        probe(countmin_table, hVal, pars, k, true, false);
      }
    }
  }
  __syncwarp();
}

__global__ void count_kmers(char *read_seq, const size_t n_reads,
                            const size_t read_length, const int k,
                            unsigned int *countmin_table, count_min_pars *pars,
                            unsigned int *bloom_table, unsigned int *hist_table,
                            const bool use_rc) {
  uint64_t fhVal, rhVal, hVal;
  size_t read_stride = copy_reads_to_shared(read_seq, read_length, n_reads);
  int read_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (read_index < n_reads)
    // Get first valid k-mer
    bool counted;
    if (use_rc) {
      NTC64(read_seq + threadIdx.x * read_stride, k, fhVal, rhVal, hVal, 1);
      counted = probe(bloom_table, hVal, pars, k, false, true);
    } else {
      NT64(read_seq + threadIdx.x * read_stride, k, fhVal, 1);
      counted = probe(bloom_table, hVal, pars, k, false, true);
    }
    if (!counted) {
      hist_table[read_index] =
          probe(countmin_table, hVal, pars, k, false, false);
    }
    __syncwarp();

    // Roll through remaining k-mers in the read
    for (int pos = 0; pos < read_length - k; pos++) {
      fhVal = // stall short scoreboard
          NTF64(fhVal, k, read_seq[threadIdx.x * read_stride + pos],
                read_seq[threadIdx.x * read_stride + pos + k]);
      if (use_rc) {
        rhVal = // stall short scoreboard
            NTR64(rhVal, k, read_seq[threadIdx.x * read_stride + pos],
                  read_seq[threadIdx.x * read_stride + pos + k]);
        hVal = (rhVal < fhVal) ? rhVal : fhVal;
        counted = probe(bloom_table, hVal, pars, k, false, true);
      } else {
        counted = probe(bloom_table, hVal, pars, k, false, true);
      }
      if (!counted) {
        hist_table[read_index + pos * n_reads] =
            probe(countmin_table, hVal, pars, k, false, false);
      }
      __syncwarp();
    }
  }
}

// TODO: use cub select if on this output
