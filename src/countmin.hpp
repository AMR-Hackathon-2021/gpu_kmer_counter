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
  CountMin(const std::vector<std::string> &filenames, const size_t width,
           const size_t height, const size_t n_threads, const size_t width_bits,
           const size_t hash_per_hash, const int k, const int table_rows,
           const int device_id = 0)
      : width_(width), height_(height), k_(k), count_min_(width * height) {
    CUDA_CALL(cudaSetDevice(device_id));
    copyNtHashTablesToDevice();

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
    std::vector<char> seq = get_reads(filenames, n_threads);

    // get the number of reads and read length
    n_reads_ = seq.size();
    read_len_ = (seq.at(0)).size();
    // need to flatten std::vector<std::string> seq_ into std::vector<char>

    // copy to device memory
    d_pars_ = device_value<count_min_pars>(pars_);
    device_array<char> d_reads(seq);
    construct_table(d_reads);
  }

  // To count the histogram, use a bloom filter
  // Bloom filter keeps track of whether k-mer has already been counted
  // For those uncounted, return the value from countmin table
  NumpyIntMatrix histogram() {
    std::vector<uint32_t> counts;
    NumpyIntMatrix hist_mat =
        Eigen::Map<Eigen::Matrix<uint32_t, Eigen::Dynamic, 1>>(
            histogram_.data(), histogram_.size());
  }

  uint32_t get_count(const std::string &kmer) {
    uint64_t fhVal, rhVal, hVal;
    // TODO: make sure nthash tables are available on host too
    NTC64(kmer.data(), k_, fhVal, rhVal, hVal);
    return probe(count_min_.data(), hVal, pars_, k_, false, false);
  }

private:
  struct GtThan {
    int compare;
    CUB_RUNTIME_FUNCTION __forceinline__ LessThan(int compare)
        : compare(compare) {}
    CUB_RUNTIME_FUNCTION __forceinline__ bool operator()(const int &a) const {
      return (a > compare);
    }
  };

  void construct_table(device_array<char> &reads) {
    const size_t blockSize = 64;
    const size_t blockCount = (n_reads_ + blockSize - 1) / blockSize;

    device_array<uint32_t> d_count_min(width_ * height_);
    fill_kmers<<<blockCount, blockSize>>>(reads.data(), n_reads_, read_len_, k_,
                                          d_count_min.data(), d_pars_.data());
    d_count_min.get_array(count_min_);

    device_array<uint32_t> d_bloom_filter(width_);
    device_array<uint32_t> d_hist_in(reads.size());
    count_kmers<<<blockCount, blockSize>>>(
        reads.data(), n_reads_, read_len_, k_, d_count_min.data(),
        count_min_pars * pars, d_bloom_filter.data(), d_hist_in.data());
    CUDA_CALL(cudaDeviceSynchronize());

    // Set up cub
    device_array<uint32_t> d_hist_out(reads.size());
    device_value<int> d_num_selected_out;
    device_array<void> d_temp_storage;
    size_t temp_storage_bytes = 0;
    GtThan select_op(0);

    // Determine temporary device storage requirements
    cub::DeviceSelect::If(d_temp_storage.data(), temp_storage_bytes,
                          d_hist_in.data(), d_hist_out.data(),
                          d_num_selected_out.data(), d_hist_in.size(),
                          select_op);
    // Allocate temporary storage
    d_temp_storage.set_size(temp_storage_bytes);
    // Run selection
    cub::DeviceSelect::If(d_temp_storage.data(), temp_storage_bytes,
                          d_hist_in.data(), d_hist_out.data(),
                          d_num_selected_out.data(), d_hist_in.size(),
                          select_op);
    histogram_.resize(d_num_selected_out.get_value());
    d_hist_out.get_array(histogram_, histogram_.size());
  }

  size_t width_;
  size_t height_;
  size_t n_reads_;
  size_t read_len_;
  int k_;

  std::vector<uint32_t> histogram_;
  std::vector<uint32_t> count_min_;

  // parameters for count_min table
  count_min_pars pars_;
  device_value<count_min_pars> d_pars_;
};
