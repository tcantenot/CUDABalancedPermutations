# Generation of balanced permutations in CUDA

Original idea from Martin Roberts: http://extremelearning.com.au/isotropic-blue-noise-point-sets/

Reference Java implementation from Tommy Ettinger: https://github.com/tommyettinger/sarong/blob/master/src/test/java/sarong/PermutationEtc.java
</br>Reference C/C++/ISPC implementation from Max Tarpini: https://github.com/RomboDev/Miscellaneous/tree/master/MRbluenoisepointsets


</br>Usage: `BalancedPermutations -l <permutations length> -n <number of permutations to generate> [-t] [-c] [-s]`
</br>(-t to display timings, -c to perform sanity checks on the generated balanced permutations, -s to not display the generated permutations)

</br>On my GTX 1070 with CUDA 9.2, it can generate:
 - 1000000 balanced permutations of length 16 in about 250 milliseconds (but only 574573 of them are unique...) (*)
 - 100000 balanced permutations of length 32 in about 2.6 seconds
 - 500 balanced permutations of length 56 in about 2.5 seconds
 - 100 balanced permutations of length 64 in about 2.3 seconds
 - 10 balanced permutations of length 96 in about 41 seconds
 - 2 balanced permutations of length 126 in about 370 seconds


</br>Some limitations:
 - balanced permutations of length up to 126
 - duplicates permutations can be found (*) (although it does not happen with 100000 32-element long permutations (all unique))
    ```
    BalancedPermutations -l 32 -n 100000 | sort | uniq | wc -l
    100000
    ```
 - all in-memory, output only at the end of the generation
 - NVIDIA only GPUs (can be easily ported to OpenCL)
 - command line parsing code is unsafe

</br>(*) An idea to prevent duplicates while keeping high parallelism would be to use to a [bloom](https://en.wikipedia.org/wiki/Bloom_filter) [filter](https://blog.demofox.org/2015/02/08/estimating-set-membership-with-a-bloom-filter/) or a [quotient filter](https://en.wikipedia.org/wiki/Quotient_filter) to quickly determine if a permutation has not already been added.