extern "C" __global__ void mosaic_rgb24(
    unsigned char* frame,
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

    if (block_size == 0) {
        return;
    }

    const unsigned int block_columns = (roi_width - 1) / block_size + 1;
    const unsigned int block_rows = (roi_height - 1) / block_size + 1;
    const unsigned int block_count = block_columns * block_rows;
    if (index >= block_count) {
        return;
    }

    const unsigned int block_x = index % block_columns;
    const unsigned int block_y = index / block_columns;
    const unsigned int block_left = left + block_x * block_size;
    const unsigned int block_top = top + block_y * block_size;
    const unsigned int block_right = block_left + min(block_size, right - block_left);
    const unsigned int block_bottom = block_top + min(block_size, bottom - block_top);
    unsigned long long red = 0;
    unsigned long long green = 0;
    unsigned long long blue = 0;
    unsigned long long count = 0;

    for (unsigned int y = block_top; y < block_bottom; ++y) {
        for (unsigned int x = block_left; x < block_right; ++x) {
            const unsigned int source = y * stride + x * 3;
            red += frame[source];
            green += frame[source + 1];
            blue += frame[source + 2];
            ++count;
        }
    }

    const unsigned char average_red = (unsigned char)(red / count);
    const unsigned char average_green = (unsigned char)(green / count);
    const unsigned char average_blue = (unsigned char)(blue / count);
    for (unsigned int y = block_top; y < block_bottom; ++y) {
        for (unsigned int x = block_left; x < block_right; ++x) {
            const unsigned int destination = y * stride + x * 3;
            frame[destination] = average_red;
            frame[destination + 1] = average_green;
            frame[destination + 2] = average_blue;
        }
    }
}
