#pragma once

#include <string>
#include <vector>

#include <Eigen/Dense>

#include "containers.cuh"

typedef Eigen::Matrix<uint32_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    NumpyIntMatrix;

// TODO
// 1. Get function to read from fastq (inside setup.py an alternative - Louise)
// 2. Fill in count_reads_gpu()

// is count_min_pars required or can this be calculated?
struct count_min_pars {
    const int width_bits;
    const int hash_per_hash;
    uint64_t mask;
    uint32_t table_width;
    int table_rows;
};

// d_count_min already constructed upon constructor of countmin
class CountMin {
public:
  CountMin(const std::vector<std::string> &filename, const size_t width,
           const size_t height, const size_t n_threads)
      : width_(width), height_(height), d_count_min_(width * height), d_bloom_(width) {
    seq_ = get_reads(filename, n_threads);
    // need to flatten std::vector<std::string> seq_ into std::vector<char>
    d_reads_ = device_array<char>(seq_);
    count_reads_gpu();
  }

  // To count the histogram, use a bloom filter
  // Bloom filter keeps track of whether k-mer has already been counted
  // For those uncounted, return the value from countmin table
  NumpyIntMatrix histogram() {
    std::vector<uint32_t> counts;
    NumpyIntMatrix hist_mat = Eigen::Map<
              Eigen::Matrix<uint32_t, Eigen::Dynamic, 1>>(
              counts.data(), counts.size());
  }

  uint32_t get_count(const std::string& kmer) {
    probe(kmer);
  }

private:
  void construct_table() {}

  uint32_t probe_table() {}

  size_t width_;
  size_t height_;

  std::vector<char> seq_;
  device_array<char> d_reads_;
  device_array<uint32_t> d_count_min_;
  device_array<uint32_t> d_bloom_;

  // delete move and copy to avoid accidentally using them
  GPUCountMin(const GPUCountMin &) = delete;
  GPUCountMin(GPUCountMin &&) = delete;

  // parameters for count_min table
  count_min_pars _pars;
  count_min_pars * _d_pars;
};
