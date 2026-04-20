from modules.layer import Layer
#from cython_modules.maxpool2d import maxpool_forward_cython
import numpy as np

class MaxPool2D(Layer):
    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride

    def forward_original(self, input, training=True):  # input: np.ndarray of shape [B, C, H, W]
        self.input = input
        B, C, H, W = input.shape
        KH, KW = self.kernel_size, self.kernel_size
        SH, SW = self.stride, self.stride

        out_h = (H - KH) // SH + 1
        out_w = (W - KW) // SW + 1

        self.max_indices = np.zeros((B, C, out_h, out_w, 2), dtype=int)
        output = np.zeros((B, C, out_h, out_w),dtype=input.dtype)

        for b in range(B):
            for c in range(C):
                for i in range(out_h):
                    for j in range(out_w):
                        h_start = i * SH
                        h_end = h_start + KH
                        w_start = j * SW
                        w_end = w_start + KW

                        window = input[b, c, h_start:h_end, w_start:w_end]
                        max_idx = np.unravel_index(np.argmax(window), window.shape)
                        max_val = window[max_idx]

                        output[b, c, i, j] = max_val
                        self.max_indices[b, c, i, j] = (h_start + max_idx[0], w_start + max_idx[1])

        return output

    def backward(self, grad_output, learning_rate=None):
        B, C, H, W = self.input.shape
        grad_input = np.zeros_like(self.input, dtype=grad_output.dtype)
        out_h, out_w = grad_output.shape[2], grad_output.shape[3]

        for b in range(B):
            for c in range(C):
                for i in range(out_h):
                    for j in range(out_w):
                        r, s = self.max_indices[b, c, i, j]
                        grad_input[b, c, r, s] += grad_output[b, c, i, j]

        return grad_input

    # --- GENERADA CON IA ---
    def forward(self, input, training=True): 
        self.input = input
        B, C, H, W = input.shape
        KH, KW = self.kernel_size, self.kernel_size
        SH, SW = self.stride, self.stride

        out_h = (H - KH) // SH + 1
        out_w = (W - KW) // SW + 1

        self.max_indices = np.zeros((B, C, out_h, out_w, 2), dtype=int)
        output = np.zeros((B, C, out_h, out_w), dtype=input.dtype)

        h_starts = [i * SH for i in range(out_h)]
        w_starts = [j * SW for j in range(out_w)]

        for b in range(B):
            input_b = input[b]
            output_b = output[b]
            max_indices_b = self.max_indices[b]
            for c in range(C):
                input_bc = input_b[c]
                output_bc = output_b[c]
                max_indices_bc = max_indices_b[c]
                for i, h_start in enumerate(h_starts):
                    h_end = h_start + KH
                    for j, w_start in enumerate(w_starts):
                        w_end = w_start + KW
                        window = input_bc[h_start:h_end, w_start:w_end]
                        flat_idx = np.argmax(window)
                        max_h = flat_idx // KW
                        max_w = flat_idx % KW
                        output_bc[i, j] = window[max_h, max_w]
                        max_indices_bc[i, j] = (h_start + max_h, w_start + max_w)
        return output
