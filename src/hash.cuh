/*
 *
 * nthash
 * Author: Hamid Mohamadi
 * Genome Sciences Centre,
 * British Columbia Cancer Agency
 *
 * Modified for CUDA by John Lees (2021)
 */
#pragma once

#include <stdint.h>

#include "cuda_call.cuh"

// offset for the complement base in the random seeds table
const uint8_t cpOff = 0x07;

// shift for gerenerating multiple hash values
const int multiShift = 27;

// seed for gerenerating multiple hash values
static const uint64_t multiSeed = 0x90b45d39fb6da1fa;

// 64-bit random seeds corresponding to bases and their complements
static const uint64_t seedA = 0x3c8bfbb395c60474;
static const uint64_t seedC = 0x3193c18562a02b4c;
static const uint64_t seedG = 0x20323ed082572324;
static const uint64_t seedT = 0x295549f54be24456;
static const uint64_t seedN = 0x0000000000000000;

static const uint64_t seedTab[256] = {
    seedN, seedT, seedN, seedG, seedA, seedA, seedN, seedC, // 0..7
    seedN, seedN, seedN, seedN, seedN, seedN, seedN, seedN, // 8..15
    seedN, seedN, seedN, seedN, seedN, seedN, seedN, seedN, // 16..23
    seedN, seedN, seedN, seedN, seedN, seedN, seedN, seedN, // 24..31
    seedN, seedN, seedN, seedN, seedN, seedN, seedN, seedN, // 32..39
    seedN, seedN, seedN, seedN, seedN, seedN, seedN, seedN, // 40..47
    seedN, seedN, seedN, seedN, seedN, seedN, seedN, seedN, // 48..55
    seedN, seedN, seedN, seedN, seedN, seedN, seedN, seedN, // 56..63
    seedN, seedA, seedN, seedC, seedN, seedN, seedN, seedG, // 64..71
    seedN, seedN, seedN, seedN, seedN, seedN, seedN, seedN, // 72..79
    seedN, seedN, seedN, seedN, seedT, seedT, seedN, seedN, // 80..87
    seedN, seedN, seedN, seedN, seedN, seedN, seedN, seedN, // 88..95
    seedN, seedA, seedN, seedC, seedN, seedN, seedN, seedG, // 96..103
    seedN, seedN, seedN, seedN, seedN, seedN, seedN, seedN, // 104..111
    seedN, seedN, seedN, seedN, seedT, seedT, seedN, seedN, // 112..119
    seedN, seedN, seedN, seedN, seedN, seedN, seedN, seedN, // 120..127
    seedN, seedN, seedN, seedN, seedN, seedN, seedN, seedN, // 128..135
    seedN, seedN, seedN, seedN, seedN, seedN, seedN, seedN, // 136..143
    seedN, seedN, seedN, seedN, seedN, seedN, seedN, seedN, // 144..151
    seedN, seedN, seedN, seedN, seedN, seedN, seedN, seedN, // 152..159
    seedN, seedN, seedN, seedN, seedN, seedN, seedN, seedN, // 160..167
    seedN, seedN, seedN, seedN, seedN, seedN, seedN, seedN, // 168..175
    seedN, seedN, seedN, seedN, seedN, seedN, seedN, seedN, // 176..183
    seedN, seedN, seedN, seedN, seedN, seedN, seedN, seedN, // 184..191
    seedN, seedN, seedN, seedN, seedN, seedN, seedN, seedN, // 192..199
    seedN, seedN, seedN, seedN, seedN, seedN, seedN, seedN, // 200..207
    seedN, seedN, seedN, seedN, seedN, seedN, seedN, seedN, // 208..215
    seedN, seedN, seedN, seedN, seedN, seedN, seedN, seedN, // 216..223
    seedN, seedN, seedN, seedN, seedN, seedN, seedN, seedN, // 224..231
    seedN, seedN, seedN, seedN, seedN, seedN, seedN, seedN, // 232..239
    seedN, seedN, seedN, seedN, seedN, seedN, seedN, seedN, // 240..247
    seedN, seedN, seedN, seedN, seedN, seedN, seedN, seedN  // 248..255
};

static const uint64_t A33r[33] = {
    0x195c60474, 0x12b8c08e9, 0x571811d3, 0xae3023a6, 0x15c60474c, 0xb8c08e99, 0x171811d32, 0xe3023a65, 0x1c60474ca, 0x18c08e995, 0x11811d32b, 0x3023a657, 0x60474cae, 0xc08e995c, 0x1811d32b8, 0x1023a6571, 0x474cae3, 0x8e995c6, 0x11d32b8c, 0x23a65718, 0x474cae30, 0x8e995c60, 0x11d32b8c0, 0x3a657181, 0x74cae302, 0xe995c604, 0x1d32b8c08, 0x1a6571811, 0x14cae3023, 0x995c6047, 0x132b8c08e, 0x6571811d, 0xcae3023a};

static const uint64_t A31l[31] = {
    0x3c8bfbb200000000, 0x7917f76400000000, 0xf22feec800000000, 0xe45fdd9200000000, 0xc8bfbb2600000000, 0x917f764e00000000, 0x22feec9e00000000, 0x45fdd93c00000000, 0x8bfbb27800000000, 0x17f764f200000000, 0x2feec9e400000000, 0x5fdd93c800000000, 0xbfbb279000000000, 0x7f764f2200000000, 0xfeec9e4400000000, 0xfdd93c8a00000000, 0xfbb2791600000000, 0xf764f22e00000000, 0xeec9e45e00000000, 0xdd93c8be00000000, 0xbb27917e00000000, 0x764f22fe00000000, 0xec9e45fc00000000, 0xd93c8bfa00000000, 0xb27917f600000000, 0x64f22fee00000000, 0xc9e45fdc00000000, 0x93c8bfba00000000, 0x27917f7600000000, 0x4f22feec00000000, 0x9e45fdd800000000};

static const uint64_t C33r[33] = {
    0x162a02b4c, 0xc5405699, 0x18a80ad32, 0x115015a65, 0x2a02b4cb, 0x54056996, 0xa80ad32c, 0x15015a658, 0xa02b4cb1, 0x140569962, 0x80ad32c5, 0x1015a658a, 0x2b4cb15, 0x569962a, 0xad32c54, 0x15a658a8, 0x2b4cb150, 0x569962a0, 0xad32c540, 0x15a658a80, 0xb4cb1501, 0x169962a02, 0xd32c5405, 0x1a658a80a, 0x14cb15015, 0x9962a02b, 0x132c54056, 0x658a80ad, 0xcb15015a, 0x1962a02b4, 0x12c540569, 0x58a80ad3, 0xb15015a6};

static const uint64_t C31l[31] = {
    0x3193c18400000000, 0x6327830800000000, 0xc64f061000000000, 0x8c9e0c2200000000, 0x193c184600000000, 0x3278308c00000000, 0x64f0611800000000, 0xc9e0c23000000000, 0x93c1846200000000, 0x278308c600000000, 0x4f06118c00000000, 0x9e0c231800000000, 0x3c18463200000000, 0x78308c6400000000, 0xf06118c800000000, 0xe0c2319200000000, 0xc184632600000000, 0x8308c64e00000000, 0x6118c9e00000000, 0xc23193c00000000, 0x1846327800000000, 0x308c64f000000000, 0x6118c9e000000000, 0xc23193c000000000, 0x8463278200000000, 0x8c64f0600000000, 0x118c9e0c00000000, 0x23193c1800000000, 0x4632783000000000, 0x8c64f06000000000, 0x18c9e0c200000000};

static const uint64_t G33r[33] = {
    0x82572324, 0x104ae4648, 0x95c8c91, 0x12b91922, 0x25723244, 0x4ae46488, 0x95c8c910, 0x12b919220, 0x57232441, 0xae464882, 0x15c8c9104, 0xb9192209, 0x172324412, 0xe4648825, 0x1c8c9104a, 0x191922095, 0x12324412b, 0x46488257, 0x8c9104ae, 0x11922095c, 0x324412b9, 0x64882572, 0xc9104ae4, 0x1922095c8, 0x124412b91, 0x48825723, 0x9104ae46, 0x122095c8c, 0x4412b919, 0x88257232, 0x1104ae464, 0x2095c8c9, 0x412b9192};

static const uint64_t G31l[31] = {
    0x20323ed000000000, 0x40647da000000000, 0x80c8fb4000000000, 0x191f68200000000, 0x323ed0400000000, 0x647da0800000000, 0xc8fb41000000000, 0x191f682000000000, 0x323ed04000000000, 0x647da08000000000, 0xc8fb410000000000, 0x91f6820200000000, 0x23ed040600000000, 0x47da080c00000000, 0x8fb4101800000000, 0x1f68203200000000, 0x3ed0406400000000, 0x7da080c800000000, 0xfb41019000000000, 0xf682032200000000, 0xed04064600000000, 0xda080c8e00000000, 0xb410191e00000000, 0x6820323e00000000, 0xd040647c00000000, 0xa080c8fa00000000, 0x410191f600000000, 0x820323ec00000000, 0x40647da00000000, 0x80c8fb400000000, 0x10191f6800000000};

static const uint64_t T33r[33] = {
    0x14be24456, 0x97c488ad, 0x12f89115a, 0x5f1222b5, 0xbe24456a, 0x17c488ad4, 0xf89115a9, 0x1f1222b52, 0x1e24456a5, 0x1c488ad4b, 0x189115a97, 0x11222b52f, 0x24456a5f, 0x488ad4be, 0x9115a97c, 0x1222b52f8, 0x4456a5f1, 0x88ad4be2, 0x1115a97c4, 0x22b52f89, 0x456a5f12, 0x8ad4be24, 0x115a97c48, 0x2b52f891, 0x56a5f122, 0xad4be244, 0x15a97c488, 0xb52f8911, 0x16a5f1222, 0xd4be2445, 0x1a97c488a, 0x152f89115, 0xa5f1222b};

static const uint64_t T31l[31] = {
    0x295549f400000000, 0x52aa93e800000000, 0xa55527d000000000, 0x4aaa4fa200000000, 0x95549f4400000000, 0x2aa93e8a00000000, 0x55527d1400000000, 0xaaa4fa2800000000, 0x5549f45200000000, 0xaa93e8a400000000, 0x5527d14a00000000, 0xaa4fa29400000000, 0x549f452a00000000, 0xa93e8a5400000000, 0x527d14aa00000000, 0xa4fa295400000000, 0x49f452aa00000000, 0x93e8a55400000000, 0x27d14aaa00000000, 0x4fa2955400000000, 0x9f452aa800000000, 0x3e8a555200000000, 0x7d14aaa400000000, 0xfa29554800000000, 0xf452aa9200000000, 0xe8a5552600000000, 0xd14aaa4e00000000, 0xa295549e00000000, 0x452aa93e00000000, 0x8a55527c00000000, 0x14aaa4fa00000000};

static const uint64_t N33r[33] = {
    seedN, seedN, seedN, seedN, seedN, seedN, seedN, seedN,
    seedN, seedN, seedN, seedN, seedN, seedN, seedN, seedN,
    seedN, seedN, seedN, seedN, seedN, seedN, seedN, seedN,
    seedN, seedN, seedN, seedN, seedN, seedN, seedN, seedN,
    seedN};

static const uint64_t N31l[31] = {
    seedN, seedN, seedN, seedN, seedN, seedN, seedN, seedN,
    seedN, seedN, seedN, seedN, seedN, seedN, seedN, seedN,
    seedN, seedN, seedN, seedN, seedN, seedN, seedN, seedN,
    seedN, seedN, seedN, seedN, seedN, seedN, seedN};

static const uint64_t *msTab33r[256] = {
    N33r, T33r, N33r, G33r, A33r, A33r, N33r, C33r, // 0..7
    N33r, N33r, N33r, N33r, N33r, N33r, N33r, N33r, // 8..15
    N33r, N33r, N33r, N33r, N33r, N33r, N33r, N33r, // 16..23
    N33r, N33r, N33r, N33r, N33r, N33r, N33r, N33r, // 24..31
    N33r, N33r, N33r, N33r, N33r, N33r, N33r, N33r, // 32..39
    N33r, N33r, N33r, N33r, N33r, N33r, N33r, N33r, // 40..47
    N33r, N33r, N33r, N33r, N33r, N33r, N33r, N33r, // 48..55
    N33r, N33r, N33r, N33r, N33r, N33r, N33r, N33r, // 56..63
    N33r, A33r, N33r, C33r, N33r, N33r, N33r, G33r, // 64..71
    N33r, N33r, N33r, N33r, N33r, N33r, N33r, N33r, // 72..79
    N33r, N33r, N33r, N33r, T33r, T33r, N33r, N33r, // 80..87
    N33r, N33r, N33r, N33r, N33r, N33r, N33r, N33r, // 88..95
    N33r, A33r, N33r, C33r, N33r, N33r, N33r, G33r, // 96..103
    N33r, N33r, N33r, N33r, N33r, N33r, N33r, N33r, // 104..111
    N33r, N33r, N33r, N33r, T33r, T33r, N33r, N33r, // 112..119
    N33r, N33r, N33r, N33r, N33r, N33r, N33r, N33r, // 120..127
    N33r, N33r, N33r, N33r, N33r, N33r, N33r, N33r, // 128..135
    N33r, N33r, N33r, N33r, N33r, N33r, N33r, N33r, // 136..143
    N33r, N33r, N33r, N33r, N33r, N33r, N33r, N33r, // 144..151
    N33r, N33r, N33r, N33r, N33r, N33r, N33r, N33r, // 152..159
    N33r, N33r, N33r, N33r, N33r, N33r, N33r, N33r, // 160..167
    N33r, N33r, N33r, N33r, N33r, N33r, N33r, N33r, // 168..175
    N33r, N33r, N33r, N33r, N33r, N33r, N33r, N33r, // 176..183
    N33r, N33r, N33r, N33r, N33r, N33r, N33r, N33r, // 184..191
    N33r, N33r, N33r, N33r, N33r, N33r, N33r, N33r, // 192..199
    N33r, N33r, N33r, N33r, N33r, N33r, N33r, N33r, // 200..207
    N33r, N33r, N33r, N33r, N33r, N33r, N33r, N33r, // 208..215
    N33r, N33r, N33r, N33r, N33r, N33r, N33r, N33r, // 216..223
    N33r, N33r, N33r, N33r, N33r, N33r, N33r, N33r, // 224..231
    N33r, N33r, N33r, N33r, N33r, N33r, N33r, N33r, // 232..239
    N33r, N33r, N33r, N33r, N33r, N33r, N33r, N33r, // 240..247
    N33r, N33r, N33r, N33r, N33r, N33r, N33r, N33r  // 248..255
};

static const uint64_t *msTab31l[256] = {
    N31l, T31l, N31l, G31l, A31l, A31l, N31l, C31l, // 0..7
    N31l, N31l, N31l, N31l, N31l, N31l, N31l, N31l, // 8..15
    N31l, N31l, N31l, N31l, N31l, N31l, N31l, N31l, // 16..23
    N31l, N31l, N31l, N31l, N31l, N31l, N31l, N31l, // 24..31
    N31l, N31l, N31l, N31l, N31l, N31l, N31l, N31l, // 32..39
    N31l, N31l, N31l, N31l, N31l, N31l, N31l, N31l, // 40..47
    N31l, N31l, N31l, N31l, N31l, N31l, N31l, N31l, // 48..55
    N31l, N31l, N31l, N31l, N31l, N31l, N31l, N31l, // 56..63
    N31l, A31l, N31l, C31l, N31l, N31l, N31l, G31l, // 64..71
    N31l, N31l, N31l, N31l, N31l, N31l, N31l, N31l, // 72..79
    N31l, N31l, N31l, N31l, T31l, T31l, N31l, N31l, // 80..87
    N31l, N31l, N31l, N31l, N31l, N31l, N31l, N31l, // 88..95
    N31l, A31l, N31l, C31l, N31l, N31l, N31l, G31l, // 96..103
    N31l, N31l, N31l, N31l, N31l, N31l, N31l, N31l, // 104..111
    N31l, N31l, N31l, N31l, T31l, T31l, N31l, N31l, // 112..119
    N31l, N31l, N31l, N31l, N31l, N31l, N31l, N31l, // 120..127
    N31l, N31l, N31l, N31l, N31l, N31l, N31l, N31l, // 128..135
    N31l, N31l, N31l, N31l, N31l, N31l, N31l, N31l, // 136..143
    N31l, N31l, N31l, N31l, N31l, N31l, N31l, N31l, // 144..151
    N31l, N31l, N31l, N31l, N31l, N31l, N31l, N31l, // 152..159
    N31l, N31l, N31l, N31l, N31l, N31l, N31l, N31l, // 160..167
    N31l, N31l, N31l, N31l, N31l, N31l, N31l, N31l, // 168..175
    N31l, N31l, N31l, N31l, N31l, N31l, N31l, N31l, // 176..183
    N31l, N31l, N31l, N31l, N31l, N31l, N31l, N31l, // 184..191
    N31l, N31l, N31l, N31l, N31l, N31l, N31l, N31l, // 192..199
    N31l, N31l, N31l, N31l, N31l, N31l, N31l, N31l, // 200..207
    N31l, N31l, N31l, N31l, N31l, N31l, N31l, N31l, // 208..215
    N31l, N31l, N31l, N31l, N31l, N31l, N31l, N31l, // 216..223
    N31l, N31l, N31l, N31l, N31l, N31l, N31l, N31l, // 224..231
    N31l, N31l, N31l, N31l, N31l, N31l, N31l, N31l, // 232..239
    N31l, N31l, N31l, N31l, N31l, N31l, N31l, N31l, // 240..247
    N31l, N31l, N31l, N31l, N31l, N31l, N31l, N31l  // 248..255
};

// Tables on device
__constant__ uint64_t d_seedTab[256];
__constant__ uint64_t d_A33r[33];
__constant__ uint64_t d_A31l[31];
__constant__ uint64_t d_C33r[33];
__constant__ uint64_t d_C31l[31];
__constant__ uint64_t d_G33r[33];
__constant__ uint64_t d_G31l[31];
__constant__ uint64_t d_T33r[33];
__constant__ uint64_t d_T31l[31];
__constant__ uint64_t d_N33r[33];
__constant__ uint64_t d_N31l[31];
__constant__ uint64_t *d_msTab33r[256];
__constant__ uint64_t *d_msTab31l[256];

// main nthash functions - see nthash.hpp
// All others are built from calling these

__host__ __device__ inline uint64_t rol1(const uint64_t v) {
  return (v << 1) | (v >> 63);
}

__host__ __device__ inline uint64_t ror1(const uint64_t v) {
  return (v >> 1) | (v << 63);
}

__host__ __device__ inline uint64_t rol31(const uint64_t v, unsigned s) {
  s %= 31;
  return ((v << s) | (v >> (31 - s))) & 0x7FFFFFFF;
}

__host__ __device__ inline uint64_t rol33(const uint64_t v, unsigned s) {
  s %= 33;
  return ((v << s) | (v >> (33 - s))) & 0x1FFFFFFFF;
}

__host__ __device__ inline uint64_t swapbits033(const uint64_t v) {
  uint64_t x = (v ^ (v >> 33)) & 1;
  return v ^ (x | (x << 33));
}

__host__ __device__ inline uint64_t swapbits3263(const uint64_t v) {
  uint64_t x = ((v >> 32) ^ (v >> 63)) & 1;
  return v ^ ((x << 32) | (x << 63));
}

// Forward strand hash for first k-mer
__host__ __device__ inline void NT64(const char *kmerSeq, const unsigned k,
                            uint64_t &fhVal, const size_t baseStride) {
  fhVal = 0;
  for (int i = k - 1; i >= 0; i--) {
    // Ns are removed, but this is how to check for them
    /*
    if(seedTab[(unsigned char)kmerSeq[i * baseStride]]==seedN) {
        locN=i;
        return false;
    }
    */
    fhVal = rol1(fhVal);
    fhVal = swapbits033(fhVal);
#ifdef __CUDA_ARCH__
    fhVal ^= d_seedTab[(unsigned char)kmerSeq[(k - 1 - i) * baseStride]];
#else
    fhVal ^= seedTab[(unsigned char)kmerSeq[(k - 1 - i) * baseStride]];
#endif
  }
  // return true;
}

// Both strand hashes for first k-mer
__host__ __device__ inline void NTC64(const char *kmerSeq, const unsigned k,
                             uint64_t &fhVal, uint64_t &rhVal, uint64_t &hVal,
                             const size_t baseStride) {
  hVal = fhVal = rhVal = 0;
  for (int i = (k - 1); i >= 0; i--) {
    // Ns are removed, but this is how to check for them
    /*
    if(seedTab[(unsigned char)kmerSeq[i * baseStride]]==seedN) {
        locN = i;
        return false;
    }
    */
    fhVal = rol1(fhVal);
    fhVal = swapbits033(fhVal);
#ifdef __CUDA_ARCH__
    fhVal ^= d_seedTab[(unsigned char)kmerSeq[(k - 1 - i) * baseStride]];
#else
    fhVal ^= seedTab[(unsigned char)kmerSeq[(k - 1 - i) * baseStride]];
#endif

    rhVal = rol1(rhVal);
    rhVal = swapbits033(rhVal);
#ifdef __CUDA_ARCH__
    rhVal ^= d_seedTab[(unsigned char)kmerSeq[i * baseStride] & cpOff];
#else
    rhVal ^= seedTab[(unsigned char)kmerSeq[i * baseStride] & cpOff];
#endif
  }
  hVal = (rhVal < fhVal) ? rhVal : fhVal;
  // return true;
}

// forward-strand ntHash for subsequent sliding k-mers
__host__ __device__ inline uint64_t NTF64(const uint64_t fhVal, const unsigned k,
                                 const unsigned char charOut,
                                 const unsigned char charIn) {
  uint64_t hVal = rol1(fhVal);
  hVal = swapbits033(hVal);
#ifdef __CUDA_ARCH__
  hVal ^= d_seedTab[charIn];
  hVal ^= (d_msTab31l[charOut][k % 31] | d_msTab33r[charOut][k % 33]);
#else
  hVal ^= seedTab[charIn];
  hVal ^= (msTab31l[charOut][k % 31] | msTab33r[charOut][k % 33]);
#endif
  return hVal;
}

// reverse-complement ntHash for subsequent sliding k-mers
__host__ __device__ inline uint64_t NTR64(const uint64_t rhVal, const unsigned k,
                                 const unsigned char charOut,
                                 const unsigned char charIn) {
#ifdef __CUDA_ARCH__
  uint64_t hVal = rhVal ^ (d_msTab31l[charIn & cpOff][k % 31] |
                           d_msTab33r[charIn & cpOff][k % 33]);
  hVal ^= d_seedTab[charOut & cpOff];
#else
  uint64_t hVal = rhVal ^ (msTab31l[charIn & cpOff][k % 31] |
                           msTab33r[charIn & cpOff][k % 33]);
  hVal ^= seedTab[charOut & cpOff];
#endif
  hVal = ror1(hVal);
  hVal = swapbits3263(hVal);
  return hVal;
}

// Create a new hash from an nthash
__host__ __device__ inline uint64_t shifthash(const uint64_t hVal, const unsigned k,
                                     const unsigned i) {
  uint64_t tVal = hVal * (i ^ k * multiSeed);
  tVal ^= tVal >> multiShift;
  return (tVal);
}

void copyNtHashTablesToDevice() {
  CUDA_CALL(cudaMemcpyToSymbolAsync(d_seedTab, seedTab, 256 * sizeof(uint64_t)));
  CUDA_CALL(cudaMemcpyToSymbolAsync(d_A33r, A33r, 33 * sizeof(uint64_t)));
  CUDA_CALL(cudaMemcpyToSymbolAsync(d_A31l, A31l, 31 * sizeof(uint64_t)));
  CUDA_CALL(cudaMemcpyToSymbolAsync(d_C33r, C33r, 33 * sizeof(uint64_t)));
  CUDA_CALL(cudaMemcpyToSymbolAsync(d_C31l, C31l, 31 * sizeof(uint64_t)));
  CUDA_CALL(cudaMemcpyToSymbolAsync(d_G33r, G33r, 33 * sizeof(uint64_t)));
  CUDA_CALL(cudaMemcpyToSymbolAsync(d_G31l, G31l, 31 * sizeof(uint64_t)));
  CUDA_CALL(cudaMemcpyToSymbolAsync(d_T33r, T33r, 33 * sizeof(uint64_t)));
  CUDA_CALL(cudaMemcpyToSymbolAsync(d_T31l, T31l, 31 * sizeof(uint64_t)));
  CUDA_CALL(cudaMemcpyToSymbolAsync(d_N33r, N33r, 33 * sizeof(uint64_t)));
  CUDA_CALL(cudaMemcpyToSymbolAsync(d_N31l, N31l, 31 * sizeof(uint64_t)));
  CUDA_CALL(cudaDeviceSynchronize());

  uint64_t *A33r_ptr, *A31l_ptr, *C33r_ptr, *C31l_ptr, *G33r_ptr, *G31l_ptr,
           *T33r_ptr, *T31l_ptr, *N33r_ptr, *N31l_ptr;
  CUDA_CALL(cudaGetSymbolAddress((void **)&A33r_ptr, d_A33r));
  CUDA_CALL(cudaGetSymbolAddress((void **)&A31l_ptr, d_A31l));
  CUDA_CALL(cudaGetSymbolAddress((void **)&C33r_ptr, d_C33r));
  CUDA_CALL(cudaGetSymbolAddress((void **)&C31l_ptr, d_C31l));
  CUDA_CALL(cudaGetSymbolAddress((void **)&G33r_ptr, d_G33r));
  CUDA_CALL(cudaGetSymbolAddress((void **)&G31l_ptr, d_G31l));
  CUDA_CALL(cudaGetSymbolAddress((void **)&T33r_ptr, d_T33r));
  CUDA_CALL(cudaGetSymbolAddress((void **)&T31l_ptr, d_T31l));
  CUDA_CALL(cudaGetSymbolAddress((void **)&N33r_ptr, d_N33r));
  CUDA_CALL(cudaGetSymbolAddress((void **)&N31l_ptr, d_N31l));

  static const uint64_t *d_addr_msTab33r[256] = {
      N33r_ptr, T33r_ptr, N33r_ptr, G33r_ptr,
      A33r_ptr, A33r_ptr, N33r_ptr, C33r_ptr, // 0..7
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr,
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, // 8..15
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr,
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, // 16..23
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr,
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, // 24..31
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr,
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, // 32..39
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr,
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, // 40..47
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr,
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, // 48..55
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr,
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, // 56..63
      N33r_ptr, A33r_ptr, N33r_ptr, C33r_ptr,
      N33r_ptr, N33r_ptr, N33r_ptr, G33r_ptr, // 64..71
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr,
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, // 72..79
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr,
      T33r_ptr, T33r_ptr, N33r_ptr, N33r_ptr, // 80..87
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr,
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, // 88..95
      N33r_ptr, A33r_ptr, N33r_ptr, C33r_ptr,
      N33r_ptr, N33r_ptr, N33r_ptr, G33r_ptr, // 96..103
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr,
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, // 104..111
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr,
      T33r_ptr, T33r_ptr, N33r_ptr, N33r_ptr, // 112..119
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr,
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, // 120..127
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr,
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, // 128..135
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr,
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, // 136..143
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr,
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, // 144..151
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr,
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, // 152..159
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr,
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, // 160..167
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr,
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, // 168..175
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr,
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, // 176..183
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr,
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, // 184..191
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr,
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, // 192..199
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr,
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, // 200..207
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr,
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, // 208..215
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr,
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, // 216..223
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr,
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, // 224..231
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr,
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, // 232..239
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr,
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, // 240..247
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr,
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr // 248..255
  };

  static const uint64_t *d_addr_msTab31l[256] = {
      N31l_ptr, T31l_ptr, N31l_ptr, G31l_ptr,
      A31l_ptr, A31l_ptr, N31l_ptr, C31l_ptr, // 0..7
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr,
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, // 8..15
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr,
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, // 16..23
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr,
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, // 24..31
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr,
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, // 32..39
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr,
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, // 40..47
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr,
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, // 48..55
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr,
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, // 56..63
      N31l_ptr, A31l_ptr, N31l_ptr, C31l_ptr,
      N31l_ptr, N31l_ptr, N31l_ptr, G31l_ptr, // 64..71
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr,
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, // 72..79
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr,
      T31l_ptr, T31l_ptr, N31l_ptr, N31l_ptr, // 80..87
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr,
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, // 88..95
      N31l_ptr, A31l_ptr, N31l_ptr, C31l_ptr,
      N31l_ptr, N31l_ptr, N31l_ptr, G31l_ptr, // 96..103
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr,
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, // 104..111
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr,
      T31l_ptr, T31l_ptr, N31l_ptr, N31l_ptr, // 112..119
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr,
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, // 120..127
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr,
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, // 128..135
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr,
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, // 136..143
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr,
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, // 144..151
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr,
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, // 152..159
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr,
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, // 160..167
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr,
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, // 168..175
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr,
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, // 176..183
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr,
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, // 184..191
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr,
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, // 192..199
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr,
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, // 200..207
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr,
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, // 208..215
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr,
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, // 216..223
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr,
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, // 224..231
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr,
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, // 232..239
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr,
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, // 240..247
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr,
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr // 248..255
  };

  CUDA_CALL(cudaMemcpyToSymbolAsync(d_msTab31l, d_addr_msTab31l,
                                    256 * sizeof(uint64_t *)));
  CUDA_CALL(cudaMemcpyToSymbolAsync(d_msTab33r, d_addr_msTab33r,
                                    256 * sizeof(uint64_t *)));
  CUDA_CALL(cudaDeviceSynchronize());
}
