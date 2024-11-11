import numpy as np

def conv1d(input_array, kernel, stride=1, padding=0):
    # Apply padding to the input array
    input_padded = np.pad(input_array, (padding, padding), mode='constant')
    # Example: if input_array = [1, 2, 3, 4, 5, 6, 7, 8, 9] and padding = 1
    # input_padded = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
    
    # Determine the length of the output array
    output_length = ((len(input_array) + 2 * padding - len(kernel)) // stride) + 1
    # Example: len(input_array) = 9, len(kernel) = 3, padding = 1, stride = 1
    # output_length = ((9 + 2*1 - 3) // 1) + 1 = 9
    
    # Initialize the output array
    output = np.zeros(output_length)
    # output = [0. 0. 0. 0. 0. 0. 0. 0. 0.]
    
    # Perform the convolution
    for i in range(output_length):
        # Determine the segment of the input to apply the kernel to
        segment = input_padded[i * stride : i * stride + len(kernel)]
        # Example for each iteration:
        # i = 0: segment = [0, 1, 2]
        # i = 1: segment = [1, 2, 3]
        # i = 2: segment = [2, 3, 4]
        # and so on...
        
        # Element-wise multiplication and summing
        output[i] = np.sum(segment * kernel)
        # Example calculations for each iteration (kernel = [1, 0, -1]):
        # i = 0: output[0] = 0*1 + 1*0 + 2*(-1) = 0 + 0 - 2 = -2
        # i = 1: output[1] = 1*1 + 2*0 + 3*(-1) = 1 + 0 - 3 = -2
        # i = 2: output[2] = 2*1 + 3*0 + 4*(-1) = 2 + 0 - 4 = -2
        # and so on...
    return output

# Example usage
input_array = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
kernel = np.array([1, 0, -1])

output = conv1d(input_array, kernel, stride=1, padding=1)
print("Output of 1D convolution:", output)
# Expected output based on above calculations:
# Output of 1D convolution: [-2. -2. -2. -2. -2. -2. -2. -2.  8.]
