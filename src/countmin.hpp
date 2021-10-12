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
  CountMin(const std::vector<std::string> &filenames,
           const size_t width,
           const size_t height,
           const size_t n_threads,
           const size_t width_bits,
           const size_t hash_per_hash,
           const int table_rows) :
           width_(width), height_(height), d_count_min_(width * height), d_bloom_(width) {

    // set parameters for hash table
    pars_.width_bits = width_bits;
    pars_.hash_per_hash = hash_per_hash;
    pars_.table_rows = table_rows;
    mask = 1;
    for (size_t i = 0; i < width_bits - 1; ++i) {
        _mask = _mask << 1;
        mask++;
    }
    pars_.mask = mask;
    pars_.table_width = static_cast<uint32_t>(mask);

    // function for pulling read sequences from fastq files
    seq_ = get_reads(filenames, n_threads);

    // get the number of reads and read length
    n_reads_ = seq_.size();
    read_len_ = (seq_.at(0)).size();
    // need to flatten std::vector<std::string> seq_ into std::vector<char>

    // copy to device memory
    d_pars_ = device_value<count_min_pars>(pars_)
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
  void construct_table(const int k,
                       const uint16_t min_count) {
      const size_t blockSize = 64;
      const size_t blockCount = (n_reads_ + blockSize - 1) / blockSize;

      count_kmers<<<blockCount, blockSize,
              reads.length() * blockSize * sizeof(char)>>>(
                      d_reads_.data(), n_reads_, read_len_, k,
                              d_count_min_.data(), d_pars_.get_value());

      CUDA_CALL(cudaGetLastError());
      CUDA_CALL(cudaDeviceSynchronize());
      fprintf(stderr, "%c100%%", 13);
  }

  uint32_t probe_table() {}

  size_t width_;
  size_t height_;
  size_t n_reads_;
  size_t read_len_;

  std::vector<char> seq_;
  device_array<char> d_reads_;
  device_array<uint32_t> d_count_min_;
  device_array<uint32_t> d_bloom_;

  // parameters for count_min table
  count_min_pars pars_;
  device_value<count_min_pars> d_pars_;
};
