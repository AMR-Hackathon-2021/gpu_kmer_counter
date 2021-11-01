import sys, os
sys.path.insert(0, '/home/jlees/installs/gpu_kmer_counter/build/lib.linux-x86_64-3.8')

import cuda_kmers

reads = ['/home/jlees/data/listeria/reads/12673_8#24_1.fastq.gz', '/home/jlees/data/listeria/reads/12673_8#24_2.fastq.gz']
width_bits = 30
width = int(0x3FFFFFFF)
height = 4
hash_per_hash = 2
table_rows = 4
use_rc = True
n_threads = 16
hist_upper_level = 1000
device_id = 0
k = 15

table = cuda_kmers.count_min_table(reads, width, height, n_threads, width_bits, hash_per_hash, k, table_rows, use_rc, hist_upper_level, device_id)

print(table.histogram())

print(table.get_count('AACCCAACCCAACCC'))

