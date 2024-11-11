import numpy as np

def conv3d(input_volume, kernel, stride=1, padding=0):
    # Get dimensions of the input volume and kernel
    depth, height, width = input_volume.shape
    kernel_depth, kernel_height, kernel_width = kernel.shape

    # Apply padding to each dimension of the input volume
    # Padding formula: padded_volume = np.pad(input_volume, ((padding, padding), (padding, padding), (padding, padding)), mode='constant')
    padded_volume = np.pad(input_volume, ((padding, padding), (padding, padding), (padding, padding)), mode='constant')
    # Example: if input_volume is 4x4x4 and padding = 1, padded_volume will add a border of zeros around each dimension

    # Calculate the output dimensions
    # Formula for output depth, height, and width:
    output_depth = ((depth + 2 * padding - kernel_depth) // stride) + 1
    output_height = ((height + 2 * padding - kernel_height) // stride) + 1
    output_width = ((width + 2 * padding - kernel_width) // stride) + 1

    # Initialize the output feature map with zeros
    output = np.zeros((output_depth, output_height, output_width))

    # Perform the 3D convolution
    for z in range(output_depth):         # Loop over the depth of the output
        for y in range(output_height):     # Loop over the height of the output
            for x in range(output_width):  # Loop over the width of the output
                # Define the starting point of the current region in the padded volume
                z_start, y_start, x_start = z * stride, y * stride, x * stride

                # Extract the region of the input volume to apply the kernel to
                region = padded_volume[z_start:z_start + kernel_depth,
                                        y_start:y_start + kernel_height,
                                        x_start:x_start + kernel_width]
                # Example for first region (when z=0, y=0, x=0):
                # if padded_volume is 6x6x6, kernel is 3x3x3
                # region = padded_volume[0:3, 0:3, 0:3]

                # Element-wise multiplication and summing
                # Formula for convolution at each (z, y, x) position:
                # output[z, y, x] = np.sum(region * kernel)
                output[z, y, x] = np.sum(region * kernel)

    return output

# Example usage
input_volume = np.array([
    [[1, 2, 0, 1], [2, 3, 1, 0], [0, 1, 2, 3], [1, 0, 2, 1]],
    [[1, 2, 1, 0], [3, 0, 2, 3], [1, 2, 1, 0], [2, 1, 0, 1]],
    [[2, 0, 1, 3], [1, 3, 0, 2], [1, 1, 2, 1], [0, 2, 1, 0]],
    [[0, 1, 2, 1], [1, 0, 1, 3], [2, 3, 0, 1], [1, 1, 0, 2]]
])

kernel = np.array([
    [[1, 0, -1], [1, 0, -1], [1, 0, -1]],
    [[1, 0, -1], [1, 0, -1], [1, 0, -1]],
    [[1, 0, -1], [1, 0, -1], [1, 0, -1]]
])

output = conv3d(input_volume, kernel, stride=1, padding=1)
print("Output of 3D convolution:\n", output)
