#pragma once

#include <string>
#include <vector>

#include <Eigen/Dense>

#include "containers.cuh"

typedef Eigen::Matrix<uint32_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    NumpyIntMatrix;

// TODO
// 1. Get function to read from fastq
// 2. Fill in count_reads_gpu()

class CountMin {
public:
  CountMin(const std::vector<std::string> &filename, const size_t width,
           const size_t height, const size_t n_threads)
      : width_(width), height_(height), d_count_min_(width * height), d_bloom_(width) {
    seq_ = get_reads(filename, n_threads);
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

};
