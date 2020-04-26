#include <cuda.h>
#include <malloc.h>
#include <stdint.h>
#include <stdio.h>

#ifdef USE_OPENMP
#include <omp.h>
#endif

// Original idea from Martin Roberts ...
// http://extremelearning.com.au/isotropic-blue-noise-point-sets/

// Reference Java implementation from Tommy Ettinger ...
// https://github.com/tommyettinger/sarong/blob/master/src/test/java/sarong/PermutationEtc.java

// Reference C/C++/ISPC implementation from Max Tarpini ...
// https://github.com/RomboDev/Miscellaneous/tree/master/MRbluenoisepointsets


using uint = unsigned int;

inline __device__ uint WangHash(uint seed)
{
    seed = (seed ^ 61) ^ (seed >> 16);
    seed *= 9;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2d;
    seed = seed ^ (seed >> 15);
    return seed;
}

// https://github.com/tommyettinger/sarong/blob/master/src/test/java/sarong/PermutationEtc.java
inline __device__ int NextIntBounded(int bound, uint64_t & stateA, uint64_t & stateB)
{
    const uint64_t s = (stateA += 0xC6BC279692B5C323ULL);
    const uint64_t z = ((s < 0x800000006F17146DULL) ? stateB : (stateB += 0x9479D2858AF899E6)) * (s ^ s >> 31);
    return (int)(bound * ((z ^ z >> 25) & 0xFFFFFFFFULL) >> 32);
}

template <typename T>
inline __device__ void Swap(T & lhs, T & rhs)
{
    const T tmp = lhs;
    lhs = rhs;
    rhs = tmp;
}

inline __device__ uint LaneId()
{
    uint ret;
    asm volatile("mov.u32 %0, %laneid;" : "=r"(ret));
    return ret;
}

// Set the k-th bit of x to 1
inline __device__ void SetBit(uint & x, uint k)
{
    #if 0
    x |= (1 << k);
    #else
	asm volatile("bfi.b32 %0, 0x1, %1, %2, 0x1;" : "=r"(x) : "r"(x), "r"(k));
    #endif
}

// Extracts k-th bit from x
inline __device__ uint ReadBit(uint x, uint k)
{
    #if 0
    return (x >> k) & 0x1;
    #else
    uint ret;
	asm volatile("bfe.u32 %0, %1, %2, 0x1;" : "=r"(ret) : "r"(x), "r"(k));
    return ret;
    #endif
}

// The array of bit submasks is manually indexed to avoid spilling to local memory (i.e. per-thread global memory)
// because we cannot dynamically index registers in CUDA
// (cf slide 5 of https://developer.download.nvidia.com/CUDA/training/register_spilling.pdf)
template <uint N>
inline __device__ void Bitmask_SetBit(uint (&bitSubmasks)[N], uint k)
{
    static_assert(N >= 1 && N <= 4, "N must be in [1, 4]");

    if(N == 1) // [0, 32)
    {
        SetBit(bitSubmasks[0], k);
    }
    else if(N == 2) // [0, 64)
    {
        const uint i = k < 32 ? k : k - 32;
        k < 32 ? SetBit(bitSubmasks[0], i) : SetBit(bitSubmasks[1], i);
    }
    else if(N == 3) // [0, 96)
    {
        const uint i = k < 32 ? k : (k < 64 ? k - 32 : k - 64);
        k < 32 ? SetBit(bitSubmasks[0], i) : (k < 64 ? SetBit(bitSubmasks[1], i) : SetBit(bitSubmasks[2], i));
    }
    else if(N == 4) // [0, 128)
    {
        const uint i = k < 64 ? (k < 32 ? k : k - 32) : (k < 96 ? k - 64 : k - 96);
        k < 64 ? (k < 32 ? SetBit(bitSubmasks[0], i) : SetBit(bitSubmasks[1], i)) : (k < 96 ? SetBit(bitSubmasks[2], i) : SetBit(bitSubmasks[3], i));
    }
}

template <uint N>
inline __device__ uint Bitmask_ReadBit(uint const (&bitSubmasks)[N], uint k)
{
    static_assert(N >= 1 && N <= 4, "N must be in [1, 4]");

    if(N == 1) // [0, 32)
    {
        return ReadBit(bitSubmasks[0], k);
    }
    else if(N == 2) // [0, 64)
    {
        const uint i = k < 32 ? k : k - 32;
        return k < 32 ? ReadBit(bitSubmasks[0], i) : ReadBit(bitSubmasks[1], i);
    }
    else if(N == 3) // [0, 96)
    {
        const uint i = k < 32 ? k : (k < 64 ? k - 32 : k - 64);
        return k < 32 ? ReadBit(bitSubmasks[0], i) : (k < 64 ? ReadBit(bitSubmasks[1], i) : ReadBit(bitSubmasks[2], i));
    }
    else if(N == 4) // [0, 128)
    {
        const uint i = k < 64 ? (k < 32 ? k : k - 32) : (k < 96 ? k - 64 : k - 96);
        return k < 64 ? (k < 32 ? ReadBit(bitSubmasks[0], i) : ReadBit(bitSubmasks[1], i)) : (k < 96 ? ReadBit(bitSubmasks[2], i) : ReadBit(bitSubmasks[3], i));
    }

    return 0;
}

template <uint NumSubBitmasks>
__global__ void BalancedPermutationsImpl(int * balancedPermutations, int * atomicCounter, const uint numThreadsPerGroup, const uint permLen, const uint numPerms)
{
    // The items and the deltas share the same shared memory array
    // When an new item is data, the delta that occupied the slot is moved to the slot of the delta used during the iteration
    // 1st iteration: | delta[0] | ... | delta[permLen-1]
    // 2nd iteration: | items[1] | ... | delta[permLen-1] or delta[0]
    // ...
    // permLen-th iteration: | items[1] | items[2] | ... | items[permLen-1] | unused
    // Note: item 0 is stored outside of shared memory array (in register)
    extern __shared__ char SharedMem[];
    char * items = &SharedMem[0];
    char * delta = &SharedMem[0]; // Permutation length up to 126 (because deltas are stored in char type)

    const uint laneId = LaneId();
    const uint gtid = blockIdx.x * blockDim.x + threadIdx.x;

    uint64_t stateA = WangHash(gtid);
    uint64_t stateB = WangHash(gtid + blockDim.x * gridDim.x);

    const uint halfPermLen = permLen >> 1;

    while(true)
    {
        uint numBalancedPermsFound = 0;
        if(laneId == 0) numBalancedPermsFound = *atomicCounter; // Only lane 0 reads the atomic counter...
        numBalancedPermsFound = __shfl_sync(0xFFFFFFFF, numBalancedPermsFound, 0); // ...that is then broadcasted to the other lanes

        if(numBalancedPermsFound >= numPerms) return;

        // Generate a random permutations of deltas
        for(int i = 0, smi = threadIdx.x; i < halfPermLen; ++i, smi += numThreadsPerGroup)
        {
			delta[smi] = i + 1;
			delta[smi + numThreadsPerGroup * halfPermLen] = ~i;
		}

        // Shuffle delta array in-place
        for (int i = permLen, smi = threadIdx.x + numThreadsPerGroup * (permLen-1); i > 1; --i, smi -= numThreadsPerGroup)
        {
            Swap(delta[smi], delta[threadIdx.x + numThreadsPerGroup * NextIntBounded(i, stateA, stateB)]);
        }

        // Try to generate a balanced permutation from the random deltas
        const uint item0 = NextIntBounded(permLen, stateA, stateB) + 1;

        uint usedItemMasks[NumSubBitmasks] = { 0 };
        Bitmask_SetBit(usedItemMasks, item0 - 1);

        uint currItem = item0;
        bool bFoundBalancedPerm = true;
        for(int i = 0, smi = threadIdx.x; i < permLen-1; ++i, smi += numThreadsPerGroup)
        {
            bool bFoundIthItem = false;
            for(int j = i, smj = smi; j < permLen; ++j, smj += numThreadsPerGroup) // Starts from i because the items and deltas share the same array
            {
                const int t = currItem + delta[smj];
                if(t > 0 && t <= permLen)
                {
                    const uint usedItemIdx = t - 1;
                    const bool bUsedItem = Bitmask_ReadBit(usedItemMasks, usedItemIdx);
                    if(!bUsedItem)
                    {
                        // The items and deltas share the same array so before updating the i-th item
                        // we need to move the delta value that stands there
                        delta[smj] = delta[smi];
                        items[smi] = t;
                        currItem = t;
                        Bitmask_SetBit(usedItemMasks, usedItemIdx);
                        bFoundIthItem = true;
                        break;
                    }
                }
            }

            if(!bFoundIthItem)
            {
                bFoundBalancedPerm = false;
                break;
            }
        }

        // Write found balanced permutation to global memory (if limit has been not reached)
        if(bFoundBalancedPerm)
        {
            // TODO: check that we didn't already write the permutation using a bloom filter

            const uint m = __activemask();
            const uint laneIdOfFirstActiveLane = __ffs(m) - 1;
            const uint numActiveLanes = __popc(m);

            // Only first active lane increments the atomic counter...
            if(laneId == laneIdOfFirstActiveLane)
                numBalancedPermsFound = atomicAdd(atomicCounter, numActiveLanes);

            // ...and then broadcast the counter value to the other lanes
            numBalancedPermsFound = __shfl_sync(m, numBalancedPermsFound, laneIdOfFirstActiveLane);

            const uint laneLTMask = (1 << laneId) - 1;
            const uint idxAmongActiveLanes = __popc(m & laneLTMask);
            const uint balancedPermIdx = numBalancedPermsFound + idxAmongActiveLanes;

            if(balancedPermIdx > numPerms) return;

            // Write permutation to output
            int * output = &balancedPermutations[balancedPermIdx * permLen];
            output[0] = item0;
            for(int k = 1, smk = threadIdx.x; k < permLen; ++k, smk += numThreadsPerGroup)
            {
                output[k] = items[smk];
            }
        }
    }
}

void BalancedPermutations(uint numGroups, uint numThreadsPerGroup, size_t sharedMemByteSize, int * balancedPermutations, int * atomicCounter, uint permLen, uint numPerms)
{
    if(permLen <= 32)
        BalancedPermutationsImpl<1><<<numGroups, numThreadsPerGroup, sharedMemByteSize>>>(balancedPermutations, atomicCounter, numThreadsPerGroup, permLen, numPerms);
    else if(permLen <= 64)
        BalancedPermutationsImpl<2><<<numGroups, numThreadsPerGroup, sharedMemByteSize>>>(balancedPermutations, atomicCounter, numThreadsPerGroup, permLen, numPerms);
    else if(permLen <= 96)
        BalancedPermutationsImpl<3><<<numGroups, numThreadsPerGroup, sharedMemByteSize>>>(balancedPermutations, atomicCounter, numThreadsPerGroup, permLen, numPerms);
    else if(permLen <= 126)
        BalancedPermutationsImpl<4><<<numGroups, numThreadsPerGroup, sharedMemByteSize>>>(balancedPermutations, atomicCounter, numThreadsPerGroup, permLen, numPerms);
    else
        printf("Permutation length above 126 are not supported!\n");
}

void SanityChecks(int * permutations, int permLen, int numPerms)
{
    const int k = permLen / 2;
    const int numDeltas = 2 * k + 1;

    // Check that we have valid permutations
    #ifdef USE_OPENMP
    int * mcounts = reinterpret_cast<int*>(alloca(numDeltas * sizeof(int) * omp_get_max_threads()));
    #pragma omp parallel for
    for(int i = 0; i < numPerms; ++i)
    {
        int * counts = mcounts + omp_get_thread_num() * numDeltas;
    #else
    int * counts = reinterpret_cast<int*>(alloca(numDeltas * sizeof(int)));
    for(int i = 0; i < numPerms; ++i)
    {
    #endif
        memset(counts, 0, numDeltas * sizeof(int));
        int * perm = &permutations[i * permLen];
        for(int j = 0; j < permLen; ++j)
        {
            if(perm[j] < 1 || perm[j] > permLen) // perm[j] is in [1, permLen]
            {
                printf("Invalid value!!!\n");
                break;
            }
            else
            {
                if(++counts[perm[j]-1] > 1)
                {
                    printf("Invalid permutation!!!\n");
                    break;
                }
            }
        }
    }

    // Check that we have balanced permutations
    #pragma omp parallel for
    for(int i = 0; i < numPerms; ++i)
    {
        #ifdef USE_OPENMP
        int * counts = mcounts + omp_get_thread_num() * numDeltas;
        #endif

        memset(counts, 0, numDeltas * sizeof(int));
        int * perm = &permutations[i * permLen];
        for(int j = 1; j <= permLen; ++j)
        {
            int d = j != permLen ? (perm[j] - perm[j-1]) : (perm[0] - perm[permLen-1]);
            if(d < -k || d > k)
            {
                printf("Unbalanced permutation: delta too big!!!\n");
                break;
            }
            else
            {
                if(++counts[d + k] > 1)
                {
                    printf("Unbalanced permutation: non-unique delta!!!\n");
                    break;
                }
            }
        }
    }
}

int main(int argc, char const * argv[])
{
    uint PermLen  = 32; 
    uint NumPerms = 10;

    bool bParsingSuccess = true;
    bool bPrintPerms = true;
    bool bPrintTimings = (argc == 1);
    bool bPerformSanityChecks = (argc == 1);
    for(int i = 1; i < argc && argv[i][0] == '-'; ++i)
    {
        switch(argv[i][1])
        {
            case 'l': { ++i; if(i >= argc) { bParsingSuccess = false; break; } PermLen  = atoi(argv[i]); break; }
            case 'n': { ++i; if(i >= argc) { bParsingSuccess = false; break; } NumPerms = atoi(argv[i]); break; }
            case 't': { bPrintTimings = true; break; }
            case 'c': { bPerformSanityChecks = true; break; }
            case 's': { bPrintPerms = false; break; }
            default: bParsingSuccess = false; break;
        }   
    }

    if(!bParsingSuccess)
    {
        fprintf(stderr, "Failed to parse command line arguments\n");
        fprintf(stderr, "Usage: %s -l <permutation length> -n <number of permutations to generate> [-t] [-c]\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    if(PermLen & 1 || PermLen > 126)
    {
        fprintf(stderr, "Permutation length must be even and at most equal to 126.\n");
        exit(EXIT_FAILURE);
    }


    // Select "best" device
    int numDevices;
    cudaGetDeviceCount(&numDevices);

    int deviceId = 0;
    int maxNumMultiprocessors = 0;
    for(int i = 0; i < numDevices; ++i)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        if(prop.multiProcessorCount > maxNumMultiprocessors)
        {
            maxNumMultiprocessors = prop.multiProcessorCount;
            deviceId = i;
        }
    }
    cudaSetDevice(deviceId);

    const int numSMs = maxNumMultiprocessors;
    const int numGroupsPerSM = 8;
    const int numGroups = numSMs * numGroupsPerSM;
    const int numThreadsPerGroup = 128;

    int * atomicCounter;
    cudaMalloc(&atomicCounter, sizeof(int));
    cudaMemset(atomicCounter, 0, sizeof(int));

    const size_t balancedPermutationsByteSize = NumPerms * PermLen * sizeof(int);

    int * dBalancedPermutations;
    cudaMalloc(&dBalancedPermutations, balancedPermutationsByteSize);

    const size_t sharedMemByteSize = numThreadsPerGroup * PermLen * sizeof(char);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    BalancedPermutations(numGroups, numThreadsPerGroup, sharedMemByteSize, dBalancedPermutations, atomicCounter, PermLen, NumPerms);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsedTime_ms = 0;
    cudaEventElapsedTime(&elapsedTime_ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    if(bPrintTimings)
        printf("Generated %d balanced permutations of length %d in %.3fms (%.3f permutations/s)\n\n", NumPerms, PermLen, elapsedTime_ms, (NumPerms * 1000.f / elapsedTime_ms));

    int * hBalancedPermutations = reinterpret_cast<int*>(malloc(balancedPermutationsByteSize));
    cudaMemcpy(hBalancedPermutations, dBalancedPermutations, balancedPermutationsByteSize, cudaMemcpyDeviceToHost);

    if(bPerformSanityChecks)
        SanityChecks(hBalancedPermutations, PermLen, NumPerms);

    if(bPrintPerms)
    {
        for(uint i = 0; i < NumPerms; ++i)
        {
            int * items = &hBalancedPermutations[i * PermLen];
            printf("%2d", items[0]);
            for(uint j = 1; j < PermLen; ++j)
            {
                printf(", %2d", items[j]);
            }
            if(i < NumPerms-1) printf("\n");
        }
    }

    free(hBalancedPermutations);
    cudaFree(dBalancedPermutations);
    cudaFree(atomicCounter);

    return 0;
}