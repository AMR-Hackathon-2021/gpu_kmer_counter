#include "count_kmers.cuh"

// taken from ppsketchlib/src/gpu/sketch.cu
// countmin and binsign
// using unsigned long long int = uint64_t due to atomicCAS prototype
__device__ unsigned int probe(unsigned int * table,
                              uint64_t hash_val, count_min_pars pars,
                              const int k, const bool update) {
    unsigned int min_count = UINT32_MAX;
    for (int hash_nr = 0; hash_nr < pars.table_rows; hash_nr += pars.hash_per_hash) {
        uint64_t current_hash = hash_val;
        for (uint i = 0; i < pars.hash_per_hash; i++) {
            uint32_t hash_val_masked = current_hash & pars.mask;
            cell_ptr = table + (hash_nr + i) * pars.table_width + hash_val_masked;
            unsigned int cell_count;
            if (update) {
                cell_count = atomicInc(cell_ptr, UINT32_MAX) + 1;
            } else {
                cell_count = *cell_ptr;
            }

            if (cell_count < min_count) {
                min_count = cell_count;
            }
            current_hash = current_hash >> table_width_bits;
        }
        hash_val = shifthash(hash_val, k, hash_nr / 2);
    }
    return (min_count);
}

__global__ void count_kmers(char *read_seq, const size_t n_reads,
                              const size_t read_length, const int k,
                              unsigned int *countmin_table, count_min_pars pars) {
    // Load reads in block into shared memory
    char *read_ptr;
    //int read_length_bank_pad = read_length;
    // TODO: another possible optimisation would be to put signs into shared
    // may affect occupancy though
//    if (use_shared) {
//        const int bank_bytes = 8;
//        read_length_bank_pad +=
//                read_length % bank_bytes ? bank_bytes - read_length % bank_bytes : 0;
//        extern __shared__ char read_shared[];
//        auto block = cooperative_groups::this_thread_block();
//        size_t n_reads_in_block = blockDim.x;
//        if (blockDim.x * (blockIdx.x + 1) > n_reads) {
//            n_reads_in_block = n_reads - blockDim.x * blockIdx.x;
//        }
//        // TODO: better performance if the reads are padded to 4 bytes
//        // best performance if aligned to 128
//        // NOTE: I think the optimal thing to do here is to align blockSize lots of
//        // reads in global memory to 128 when reading in, then read in padded to 4
//        // bytes Then can read in all at once with single memcpy_async with size
//        // padded to align at 128, and individual reads padded to align at 4
//        // NOTE 2: It may just be easiest to pack this into a class/type with
//        // 4 chars when reading, or even a DNA alphabet bit vector
//        for (int read_idx = 0; read_idx < n_reads_in_block; ++read_idx) {
//            // Copies one read into shared
//            cooperative_groups::memcpy_async(
//                    block, read_shared + read_idx * read_length_bank_pad,
//                    read_seq + read_length * (blockIdx.x * blockDim.x + read_idx),
//                    sizeof(char) * read_length);
//        }
//        cooperative_groups::wait(block);
//        read_ptr = read_shared;
//    } else {
//        read_ptr = read_seq + read_length * (blockIdx.x * blockDim.x);
//    }

    read_ptr = read_seq + read_length * (blockIdx.x * blockDim.x);

    int read_index = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t fhVal, rhVal, hVal;
    if (read_index < n_reads) {
        // Get first valid k-mer
        if (use_rc) {
            NTC64(read_ptr + threadIdx.x * read_length_bank_pad, k, fhVal, rhVal,
                  hVal);
            probe(countmin_table, hVal, cm_pars, k, true);
        } else {
            NT64(read_ptr + threadIdx.x * read_length_bank_pad, k, fhVal);
            probe(countmin_table, hVal, cm_pars, k, true);
        }

        // Roll through remaining k-mers in the read
        for (int pos = 0; pos < read_length - k; pos++) {
            fhVal = // stall short scoreboard
                    NTF64(fhVal, k, read_ptr[threadIdx.x * read_length_bank_pad + pos],
                          read_ptr[threadIdx.x * read_length_bank_pad + pos + k]);
            if (use_rc) {
                rhVal = // stall short scoreboard
                        NTR64(rhVal, k, read_ptr[threadIdx.x * read_length_bank_pad + pos],
                              read_ptr[threadIdx.x * read_length_bank_pad + pos + k]);
                hVal = (rhVal < fhVal) ? rhVal : fhVal;
                probe(countmin_table, hVal, cm_pars, k, true);
            } else {
                probe(countmin_table, hVal, cm_pars, k, true);
            }
        }
    }
    __syncwarp();
}