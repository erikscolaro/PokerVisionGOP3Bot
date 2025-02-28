# REMEMBER: tesseract must be installed at system level and the path must be specified in image_text.py

## Computational Analysis: Comparison of Various Template Matching Methods in OpenCV

Given a target image with dimensions $W_T=1920$ and $H_T=1080$ and $M$ template images with dimensions $W_M$ and $H_M$. Let's define the pyramid scaling factor in cv2 as $S=1/2$. The following computational complexities apply:

- For standard _template matching_:

$$\sum_{i=0}^{M} O(W_{Mi} \cdot H_{Mi} \cdot W_T \cdot H_T)$$

- For _template matching_ with _pyramid scaling_, assuming we've already pre-scaled the target image and template images by a scaling factor S:

$$\sum_{i=0}^{M} O(S^4 \cdot W_{Mi} \cdot H_{Mi} \cdot W_T \cdot H_T)$$

- For calculating convolution through two-dimensional Fourier transform, the following steps are taken:
    - FFT2D of the target image: 
    $$O(W_T \cdot H_T \cdot \log_2(W_T \cdot H_T))$$
    - FFT2D of the template images: 
    $$\sum_{i=0}^{M} O(W_{Mi} \cdot H_{Mi} \cdot \log_2(W_{Mi} \cdot H_{Mi}))$$
    - Padding of template images
    - Multiplication of FFT2D of the target image with FFT2D of template images: 
    $$\sum_{i=0}^{M} O(W_T \cdot H_T) = O(M \cdot W_T \cdot H_T)$$
    - IFFT2D of the previous multiplications: 
    $$\sum_{i=0}^{M} O(W_T \cdot H_T \cdot \log_2(W_T \cdot H_T)) = O(M \cdot W_T \cdot H_T \cdot \log_2(W_T \cdot H_T))$$
    - If we assume pre-calculated FFT2D of template images with padding already applied, then the exact complexity equals that of multiplication and inverse transform: 
    $$O(M \cdot W_T \cdot H_T) + O(M \cdot W_T \cdot H_T \cdot \log_2(W_T \cdot H_T))$$
    
    Since we're discussing Big-O notation and assuming $W_T \cdot H_T >> 2$, the first term becomes negligible, making the final Big-O complexity:
    $$O(M \cdot W_T \cdot H_T \cdot \log_2(W_T \cdot H_T))$$

Now let's compare these techniques. Template matching with scaling is always more efficient than without scaling. Rewriting the complexity of template matching with scaling:

$$O(S^4 \cdot W_T \cdot H_T \cdot \sum_{i=0}^{M}(W_{Mi} \cdot H_{Mi}))$$

The Fourier transform approach becomes advantageous compared to scaled template matching only if:

$$S^4 \cdot \sum_{i=0}^{M}(W_{Mi} \cdot H_{Mi}) > M \cdot \log_2(W_T \cdot H_T)$$

Dividing by M, the resulting term $\sum_{i=0}^{M}(W_{Mi}\cdot H_{Mi})/M$ corresponds to the average area of template images. Let's assume the worst case where template images have an average size of 16x16 pixels with the target image dimensions of 1920x1080:

$$S^4 \cdot (\sum_{i=0}^{M}(W_{Mi} \cdot H_{Mi}))/M > \log_2(W_T \cdot H_T)$$

$$S^4 \cdot \text{avg}(\text{Area}(\text{template\_images})) > S^4 \cdot 16 \cdot 16 > \log_2(1920\cdot1080)\\
 \iff S > \sqrt[4]{\log_2(1920\cdot1080) / (16\cdot16)} \approx 0.4$$
 
If 1/S must be a multiple of two, then $S \geq 1/(2^1) = 0.5 > 0.4$, the only acceptable scaling factor in the worst case is a scaling factor of 2.

### Conclusion 

Since we'll use template images larger than 16x16 in any case, and a scaling factor of 2 represents a negligible loss of information in this case (it might even provide benefits in terms of image noise reduction), I will use a _template matching_ technique with a scaling factor of 2 in this project.