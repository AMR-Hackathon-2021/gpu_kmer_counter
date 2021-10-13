#pragma once

#include <string>
#include <vector>
#include <cub/cub.cuh>

#include "containers.cuh"
#include "count_kmers.cuh"

#include "seqio.h"

//TODO create new struct for bloom filter params
const int bloom_mult = 1;

class CountMin {
public:
  CountMin(const std::vector<std::string> &filenames, const size_t width,
           const size_t height, const size_t n_threads, const size_t width_bits,
           const size_t hash_per_hash, const int k, const int table_rows, 
           const bool use_rc, const int hist_upper_level, const int device_id = 0)
      : width_(width), height_(height), k_(k), count_min_(width * height) {
    CUDA_CALL(cudaSetDevice(device_id));
    copyNtHashTablesToDevice();

    // set parameters for hash table
    pars_.width_bits = width_bits;
    pars_.hash_per_hash = hash_per_hash;
    pars_.table_rows = table_rows;
    pars_.bloom_width_mult = bloom_mult;
    uint64_t mask = 1;
    for (size_t i = 0; i < width_bits - 1; ++i) {
      mask = mask << 1;
      mask++;
    }
    pars_.mask = mask;
    pars_.table_width = static_cast<uint32_t>(mask);

    // function for pulling read sequences from fastq files
    auto sequence = SeqBuf(filenames, k_);
    auto seq = sequence.as_square_array(n_threads);

    // get the number of reads and read length
    n_reads_ = seq.size();
    read_len_ = sequence.max_length();

    // copy to device memory
    d_pars_ = device_value<count_min_pars>(pars_);
    device_array<char> d_reads(seq);
    construct_table(d_reads, use_rc, hist_upper_level);
  }

  // To count the histogram, use a bloom filter
  // Bloom filter keeps track of whether k-mer has already been counted
  // For those uncounted, return the value from countmin table
  std::vector<int> histogram() const {
    return histogram_;
  }

  uint32_t get_count(const std::string &kmer) {
    uint64_t fhVal, rhVal, hVal;
    NTC64(kmer.data(), k_, fhVal, rhVal, hVal, 1);
    return probe(count_min_.data(), hVal, &pars_, k_, false, false);
  }

private:
  void construct_table(device_array<char> &reads, const bool use_rc,
                       const int hist_upper_level) {
    const size_t blockSize = 64;
    const size_t blockCount = (n_reads_ + blockSize - 1) / blockSize;

    // Fill in the countmin table, and copy back to host
    device_array<uint32_t> d_count_min(width_ * height_);
    fill_kmers<<<blockCount, blockSize>>>(reads.data(), n_reads_, read_len_, k_,
                                          d_count_min.data(), d_pars_.data(), use_rc);
    d_count_min.get_array(count_min_);

    // Use a bloom filter to count k-mers (once) from the countmin table
    device_array<uint32_t> d_bloom_filter(width_ * pars_.bloom_width_mult);
    device_array<uint32_t> d_hist_in(reads.size());
    count_kmers<<<blockCount, blockSize>>>(
        reads.data(), n_reads_, read_len_, k_, d_count_min.data(),
        d_pars_.data(), d_bloom_filter.data(), d_hist_in.data(), use_rc);
    CUDA_CALL(cudaDeviceSynchronize());

    // Set up cub to get the non-zero k-mer counts from hist_in
    const int num_levels = hist_upper_level;
    device_array<int> d_hist_out(num_levels);
    device_array<void> d_temp_storage;
    size_t temp_storage_bytes = 0;

    // Compute histograms
      cub::DeviceHistogram::HistogramEven(d_temp_storage.data(), temp_storage_bytes,
                                          d_hist_in.data(), d_hist_out.data(),
                                          num_levels, 1.0f, (float)hist_upper_level, (int)d_hist_in.size());
    // Allocate temporary storage
    d_temp_storage.set_size(temp_storage_bytes);
    // Run selection
    cub::DeviceHistogram::HistogramEven(d_temp_storage.data(), temp_storage_bytes,
                                        d_hist_in.data(), d_hist_out.data(),
                                          num_levels, 1.0f, (float)hist_upper_level, (int)d_hist_in.size());

    // Save results on host
    histogram_.resize(num_levels);
    d_hist_out.get_array(histogram_);
  }

  size_t width_;
  size_t height_;
  size_t n_reads_;
  size_t read_len_;
  int k_;

  std::vector<int> histogram_;
  std::vector<uint32_t> count_min_;

  // parameters for count_min table
  count_min_pars pars_;
  device_value<count_min_pars> d_pars_;
};
