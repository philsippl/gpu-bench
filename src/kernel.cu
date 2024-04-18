////////////////////////////////
//   U16 or U32 Ring Matmul   //
////////////////////////////////

template <typename T, size_t LIMBS>
__global__ void reduce_ring(int *intermediate, T *output, unsigned int *aSums, int *bSums, size_t n, size_t m, size_t k, size_t offset, size_t chunkSize, size_t chunkIdx)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < chunkSize * m)
    {
        unsigned int as[LIMBS] = {};
        unsigned int bs[LIMBS] = {};

        size_t vIdx = (idx / chunkSize) * n + (idx % chunkSize) + chunkIdx * chunkSize;
        for (int i = 0; i < LIMBS; i++)
        {
            as[i] = aSums[(i * n) + (vIdx % n)];
            bs[i] = bSums[(i * m) + (vIdx / n)] + k * 128;
        }

        T result = intermediate[idx];
        for (int i = 0; i < LIMBS; i++)
        {
            for (int j = 0; j < LIMBS; j++)
            {
                if ((i + j) >= LIMBS)
                    continue;
                result += (((as[i] + bs[j]) << 7) - (k * 16384)) << (8 * (i + j));
            }
        }

        // transpose output
        output[idx / chunkSize + (idx % chunkSize) * m + offset] = result;
    }
}

extern "C" __global__ void reduce_u32(int *intermediate, unsigned int *output, unsigned int *aSums, int *bSums, size_t n, size_t m, size_t k, size_t offset, size_t chunkSize, size_t chunkIdx, unsigned int _p)
{
    reduce_ring<unsigned int, 4>(intermediate, output, aSums, bSums, n, m, k, offset, chunkSize, chunkIdx);
}

extern "C" __global__ void reduce_u16(int *intermediate, unsigned short *output, unsigned int *aSums, int *bSums, size_t n, size_t m, size_t k, size_t offset, size_t chunkSize, size_t chunkIdx, unsigned short _p)
{
    reduce_ring<unsigned short, 2>(intermediate, output, aSums, bSums, n, m, k, offset, chunkSize, chunkIdx);
}

////////////////////////////////
//     PrimeField Matmul      //
////////////////////////////////

template <typename T, size_t LIMBS>
__global__ void reduce_field(int *intermediate, T *output, unsigned int *aSums, int *bSums, size_t n, size_t m, size_t k, size_t offset, size_t chunkSize, size_t chunkIdx, T p)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long pp = p;

    if (idx < chunkSize * m)
    {
        unsigned int as[LIMBS] = {};
        unsigned int bs[LIMBS] = {};

        size_t vIdx = (idx / chunkSize) * n + (idx % chunkSize) + chunkIdx * chunkSize;
        for (int i = 0; i < LIMBS; i++)
        {
            as[i] = aSums[(i * n) + (vIdx % n)];
            bs[i] = bSums[(i * m) + (vIdx / n)] + k * 128;
        }

        int result[LIMBS * LIMBS] = {};
        for (int i = 0; i < LIMBS * LIMBS; i++)
        {
            result[i] = intermediate[idx + chunkSize * m * i];
        }

        for (int i = 0; i < LIMBS; i++)
        {
            for (int j = 0; j < LIMBS; j++)
            {
                size_t result_idx = i * LIMBS + j;
                result[result_idx] += (((as[i] + bs[j]) << 7) - (k * 16384));
            }
        }

        unsigned long long final_result = 0;
        for (int i = 0; i < LIMBS; i++)
        {
            for (int j = 0; j < LIMBS; j++)
            {
                unsigned long long f = (1ULL << (8 * (i + j))) % pp;
                final_result += ((result[i * LIMBS + j] % pp) * f) % pp;
            }
        }

        // transpose output
        output[idx / chunkSize + (idx % chunkSize) * m + offset] = final_result % pp;
    }
}

extern "C" __global__ void reduce_p32(int *intermediate, unsigned int *output, unsigned int *aSums, int *bSums, size_t n, size_t m, size_t k, size_t offset, size_t chunkSize, size_t chunkIdx, unsigned int p)
{
    reduce_field<unsigned int, 4>(intermediate, output, aSums, bSums, n, m, k, offset, chunkSize, chunkIdx, p);
}

extern "C" __global__ void reduce_p16(int *intermediate, unsigned short *output, unsigned int *aSums, int *bSums, size_t n, size_t m, size_t k, size_t offset, size_t chunkSize, size_t chunkIdx, unsigned short p)
{
    reduce_field<unsigned short, 2>(intermediate, output, aSums, bSums, n, m, k, offset, chunkSize, chunkIdx, p);
}
