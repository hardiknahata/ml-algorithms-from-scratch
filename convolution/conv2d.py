import numpy as np

def conv2d(image, kernel, stride=1, padding=0):
    # Get dimensions of the input image and kernel
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    # Apply padding to the input image
    # Formula for padding: padded_image = np.pad(image, ((p, p), (p, p)), mode='constant')
    # Example: if image = 4x4 and padding = 1, padded_image will add a border of 0s around the original image
    padded_image = np.pad(image, ((padding, padding), (padding, padding)), mode='constant')
    # If image is 4x4 and padding=1:
    # padded_image = [
    #   [0, 0, 0, 0, 0, 0],
    #   [0, 1, 2, 0, 1, 0],
    #   [0, 2, 3, 1, 0, 0],
    #   [0, 0, 1, 2, 3, 0],
    #   [0, 1, 0, 2, 1, 0],
    #   [0, 0, 0, 0, 0, 0]
    # ]

    # Calculate the output dimensions
    # Formula for output height: output_height = ((image_height + 2 * padding - kernel_height) // stride) + 1
    # Formula for output width: output_width = ((image_width + 2 * padding - kernel_width) // stride) + 1
    output_height = ((image_height + 2 * padding - kernel_height) // stride) + 1
    output_width = ((image_width + 2 * padding - kernel_width) // stride) + 1

    # Example calculation for 4x4 image, 3x3 kernel, stride=1, padding=1:
    # output_height = ((4 + 2*1 - 3) // 1) + 1 = 4
    # output_width = ((4 + 2*1 - 3) // 1) + 1 = 4
    # This means the output will be a 4x4 matrix

    # Initialize the output feature map with zeros
    output = np.zeros((output_height, output_width))
    
    # Perform the convolution
    for y in range(output_height):
        for x in range(output_width):
            # Define the starting point of the current region in the padded image
            y_start, x_start = y * stride, x * stride

            # Extract the region of the image to apply the kernel to
            region = padded_image[y_start:y_start + kernel_height, x_start:x_start + kernel_width]
            # Example for first region (when y=0, x=0):
            # if padded_image is 6x6, kernel is 3x3
            # region = padded_image[0:3, 0:3]
            # If kernel is [[1, 0, -1], [1, 0, -1], [1, 0, -1]], region = [[0, 0, 0], [0, 1, 2], [0, 2, 3]]

            # Element-wise multiplication and summing
            # Formula for convolution at each (y, x) position:
            # output[y, x] = np.sum(region * kernel)
            output[y, x] = np.sum(region * kernel)
            # For example, for position (y=0, x=0):
            # region * kernel = [[0*1, 0*0, 0*(-1)], [0*1, 1*0, 2*(-1)], [0*1, 2*0, 3*(-1)]]
            # output[0, 0] = 0 + 0 + 0 + 0 + 0 - 2 + 0 + 0 - 3 = -5

    return output

# Example usage
image = np.array([
    [1, 2, 0, 1],
    [2, 3, 1, 0],
    [0, 1, 2, 3],
    [1, 0, 2, 1]
])

kernel = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
])

output = conv2d(image, kernel, stride=1, padding=1)
print("Output of 2D convolution:\n", output)
