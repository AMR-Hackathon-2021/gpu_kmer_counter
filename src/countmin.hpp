#pragma once

#include <chrono>
#include <string>
#include <vector>
#include <cub/cub.cuh>

#include "containers.cuh"
#include "count_kmers.cuh"

#include "seqio.h"

using namespace std::literals;

//TODO create new struct for bloom filter params
const int bloom_mult = 1;

class CountMin {
public:
  CountMin(const std::vector<std::string> &filenames,
           const size_t n_threads, const size_t width_bits, const size_t n_hashes,
           const int k, const bool use_rc, const int hist_upper_level, const int device_id = 0)
      : k_(k) {
    CUDA_CALL(cudaSetDevice(device_id));
    copyNtHashTablesToDevice();

    // set parameters for hash table
    pars_.width_bits = width_bits;
    pars_.hash_per_hash = 2;
    pars_.table_rows = n_hashes;
    pars_.bloom_width_mult = bloom_mult;
    uint64_t mask = 1;
    for (size_t i = 0; i < width_bits - 1; ++i) {
      mask = mask << 1;
      mask++;
    }
    pars_.mask = mask;
    pars_.table_width = static_cast<uint32_t>(mask);

    // function for pulling read sequences from fastq files
    const auto start = std::chrono::steady_clock::now();
    auto sequence = SeqBuf(filenames, k_);
    const auto intermediate = std::chrono::steady_clock::now();
    auto seq = sequence.as_square_array(n_threads);
    const auto end = std::chrono::steady_clock::now();
    std::cerr << "Reading reads took "
              << (intermediate - start) / 1ms << "ms" << std::endl
              << "transposing: " << (end - intermediate) / 1ms << "ms" << std::endl;

    // get the number of reads and read length
    n_reads_ = sequence.n_full_seqs();
    read_len_ = sequence.max_length();

    // copy to device memory
    count_min_.resize(pars_.table_width * pars_.table_rows);
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
    if (kmer.size() != k_) {
      throw std::runtime_error("k-mer query must be same length as used for construction");
    }
    uint64_t fhVal, rhVal, hVal;
    NTC64(kmer.data(), k_, fhVal, rhVal, hVal, 1);
    return probe_cm(count_min_.data(), hVal, &pars_, k_, false);
  }

private:
  void construct_table(device_array<char> &reads, const bool use_rc,
                       const int hist_upper_level) {
    const size_t blockSize = 64;
    const size_t blockCount = (n_reads_ + blockSize - 1) / blockSize;
    const size_t shared_padding = bank_padding(read_len_);

    // Fill in the countmin table, and copy back to host
    const auto s1 = std::chrono::steady_clock::now();
    device_array<uint32_t> d_count_min(count_min_.size());
    const auto s2 = std::chrono::steady_clock::now();
    fill_kmers<<<blockCount, blockSize, blockSize * shared_padding>>>(
      reads.data(), n_reads_, read_len_, k_,
      d_count_min.data(), d_pars_.data(), use_rc);
    CUDA_CALL(cudaDeviceSynchronize());
    const auto s3 = std::chrono::steady_clock::now();
    d_count_min.get_array(count_min_);
    const auto s4 = std::chrono::steady_clock::now();

    std::cerr << "Filling: " << std::endl
              << "Alloc: "
              << (s2 - s1) / 1ms << "ms" << std::endl
              << "Kernel: "
              << (s3 - s2) / 1ms << "ms" << std::endl
              << "Copy: "
              << (s4 - s3) / 1ms << "ms" << std::endl;

    // Use a bloom filter to count k-mers (once) from the countmin table
    device_array<uint32_t> d_bloom_filter(pars_.table_width * pars_.bloom_width_mult);
    device_array<uint32_t> d_hist_in(reads.size());
    const auto s5 = std::chrono::steady_clock::now();
    count_kmers<<<blockCount, blockSize, blockSize * shared_padding>>>(
        reads.data(), n_reads_, read_len_, k_, d_count_min.data(),
        d_pars_.data(), d_bloom_filter.data(), d_hist_in.data(), use_rc);
    CUDA_CALL(cudaDeviceSynchronize());
    const auto s6 = std::chrono::steady_clock::now();
    std::cerr << "Counting: "
              << (s6 - s5) / 1ms << "ms" << std::endl;

    // Set up cub to get the non-zero k-mer counts from hist_in
    const auto s7 = std::chrono::steady_clock::now();
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
    const auto s8 = std::chrono::steady_clock::now();
    cub::DeviceHistogram::HistogramEven(d_temp_storage.data(), temp_storage_bytes,
                                        d_hist_in.data(), d_hist_out.data(),
                                          num_levels, 1.0f, (float)hist_upper_level, (int)d_hist_in.size());
    const auto s9 = std::chrono::steady_clock::now();

    // Save results on host
    histogram_.resize(num_levels);
    d_hist_out.get_array(histogram_);
    const auto s10 = std::chrono::steady_clock::now();

    std::cerr << "Histogram: " << std::endl
          << "Alloc: "
          << (s8 - s7) / 1ms << "ms" << std::endl
          << "Kernel: "
          << (s9 - s8) / 1ms << "ms" << std::endl
          << "Copy: "
          << (s10 - s9) / 1ms << "ms" << std::endl;
  }

  size_t n_reads_;
  size_t read_len_;
  int k_;

  std::vector<int> histogram_;
  std::vector<uint32_t> count_min_;

  // parameters for count_min table
  count_min_pars pars_;
  device_value<count_min_pars> d_pars_;
};

class SortingTable {
public:
  SortingTable(const std::vector<std::string> &filenames,
           const size_t n_threads, const int k, const bool use_rc,
           const int hist_upper_level, const int device_id = 0) :k_(k) {
    CUDA_CALL(cudaSetDevice(device_id));
    copyNtHashTablesToDevice();

    // function for pulling read sequences from fastq files
    const auto start = std::chrono::steady_clock::now();
    auto sequence = SeqBuf(filenames, k_);
    const auto intermediate = std::chrono::steady_clock::now();
    auto seq = sequence.as_square_array(n_threads);
    const auto end = std::chrono::steady_clock::now();
    std::cerr << "Reading reads took "
              << (intermediate - start) / 1ms << "ms" << std::endl
              << "transposing: " << (end - intermediate) / 1ms << "ms" << std::endl;

    // get the number of reads and read length
    n_reads_ = sequence.n_full_seqs();
    read_len_ = sequence.max_length();

    // copy to device memory
    device_array<char> d_reads(seq);
    construct_hist(d_reads, use_rc, hist_upper_level);
  }

  std::vector<int> histogram() const {
    return histogram_;
  }

private:
  size_t n_reads_;
  size_t read_len_;
  int k_;

  std::vector<int> histogram_;

  void construct_hist(device_array<char> &reads, const bool use_rc,
                      const int hist_upper_level) {
    const size_t blockSize = 64;
    const size_t blockCount = (n_reads_ + blockSize - 1) / blockSize;
    const size_t shared_padding = bank_padding(read_len_);

    // Fill in the countmin table, and copy back to host
    device_array<uint32_t> d_k_table(n_reads_ * (read_len_ - k_ + 1));
    const auto s1 = std::chrono::steady_clock::now();
    enum_kmers<<<blockCount, blockSize, blockSize * shared_padding>>>(
      reads.data(), n_reads_, read_len_, k_,
      d_k_table.data(), use_rc);
    CUDA_CALL(cudaDeviceSynchronize());
    const auto s2 = std::chrono::steady_clock::now();

    std::cerr << "Filling: "
              << (s2 - s1) / 1ms << "ms" << std::endl;

    // sort
    device_array<uint32_t> sorted_k_table(d_k_table.size());
    // Allocate temporary storage
    device_array<void> d_temp_storage;
    size_t temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortKeys(d_temp_storage.data(), temp_storage_bytes,
                                   d_k_table.data(), sorted_k_table.data(),
                                   d_k_table.size());
    d_temp_storage.set_size(temp_storage_bytes);
    // Run sort
    const auto s3 = std::chrono::steady_clock::now();
    cub::DeviceRadixSort::SortKeys(d_temp_storage.data(), temp_storage_bytes,
                                   d_k_table.data(), sorted_k_table.data(),
                                   d_k_table.size());
    CUDA_CALL(cudaDeviceSynchronize());
    const auto s4 = std::chrono::steady_clock::now();

    std::cerr << "Sorting: "
              << (s4 - s3) / 1ms << "ms" << std::endl;

    // runlen encode
    device_array<uint32_t> unique_out(sorted_k_table.size());
    device_array<int> counts_out(sorted_k_table.size());
    device_value<int> num_runs_out;
    // Allocate temporary storage
    temp_storage_bytes = 0;
    cub::DeviceRunLengthEncode::Encode(d_temp_storage.data(), temp_storage_bytes,
                                   sorted_k_table.data(), unique_out.data(),
                                   counts_out.data(), num_runs_out.data(), sorted_k_table.size());
    d_temp_storage.set_size(temp_storage_bytes);
    // Run the encoding
    const auto s5 = std::chrono::steady_clock::now();
    cub::DeviceRunLengthEncode::Encode(d_temp_storage.data(), temp_storage_bytes,
                                      sorted_k_table.data(), unique_out.data(),
                                      counts_out.data(), num_runs_out.data(), sorted_k_table.size());
    CUDA_CALL(cudaDeviceSynchronize());
    const auto s6 = std::chrono::steady_clock::now();

    std::cerr << "Counting: "
              << (s6 - s5) / 1ms << "ms" << std::endl;

    // Set up cub to get the non-zero k-mer counts from hist_in
    const int num_levels = hist_upper_level;
    device_array<int> d_hist_out(num_levels);
    int count_len = num_runs_out.get_value();

    // Compute histograms
    temp_storage_bytes = 0;
    cub::DeviceHistogram::HistogramEven(d_temp_storage.data(), temp_storage_bytes,
                                          counts_out.data(), d_hist_out.data(),
                                          num_levels, 1.0f, (float)hist_upper_level, count_len);
    // Allocate temporary storage
    d_temp_storage.set_size(temp_storage_bytes);
    // Run selection
    const auto s7 = std::chrono::steady_clock::now();
        cub::DeviceHistogram::HistogramEven(d_temp_storage.data(), temp_storage_bytes,
                                          counts_out.data(), d_hist_out.data(),
                                          num_levels, 1.0f, (float)hist_upper_level, count_len);
    const auto s8 = std::chrono::steady_clock::now();

    std::cerr << "Histogram: "
              << (s8 - s7) / 1ms << "ms" << std::endl;

    // Save results on host
    histogram_.resize(num_levels);
    d_hist_out.get_array(histogram_);
  }
};
