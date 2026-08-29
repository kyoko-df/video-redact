extern "C" __global__ void mosaic_rgb24(
    const unsigned char* input,
    unsigned char* output,
    unsigned int width,
    unsigned int height,
    unsigned int stride,
    unsigned int left,
    unsigned int top,
    unsigned int right,
    unsigned int bottom,
    unsigned int block_size
) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int roi_width = right - left;
    const unsigned int roi_height = bottom - top;
    const unsigned int pixel_count = roi_width * roi_height;

    if (index >= pixel_count || block_size == 0) {
        return;
    }

    const unsigned int local_x = index % roi_width;
    const unsigned int local_y = index / roi_width;
    const unsigned int x = left + local_x;
    const unsigned int y = top + local_y;

    const unsigned int sample_x = min(
        left + (local_x / block_size) * block_size + block_size / 2,
        right - 1
    );
    const unsigned int sample_y = min(
        top + (local_y / block_size) * block_size + block_size / 2,
        bottom - 1
    );

    if (x >= width || y >= height) {
        return;
    }

    const unsigned int source = sample_y * stride + sample_x * 3;
    const unsigned int destination = y * stride + x * 3;
    output[destination] = input[source];
    output[destination + 1] = input[source + 1];
    output[destination + 2] = input[source + 2];
}

