# References

## Primary Reference

This implementation is based on the following research paper:

**Towards Benchmarking and Evaluating Deepfake Detection**  
Publisher: IEEE  
Year: 2024

### Key Contributions from the Paper

The paper provides a comprehensive framework for:

1. **Benchmarking Metrics**
   - Area Under Curve (AUC) of ROC plots
   - Floating Point Operations Per Second (FLOPS)
   - Model parameter count
   - Inference time measurements

2. **Robustness Evaluation**
   - Performance under various perturbations
   - Gaussian blur analysis
   - Brightness and contrast adjustments
   - JPEG compression effects

3. **Model Architectures**
   - Meso4: Lightweight CNN for mesoscopic feature detection
   - Xception: Deep architecture with depthwise separable convolutions

4. **Evaluation Methodology**
   - Systematic comparison framework
   - Multiple evaluation dimensions
   - Computational efficiency analysis
   - Real-world robustness testing

## Implementation Details

This project implements the benchmarking and evaluation framework proposed in the paper, including:

- **Model Implementations**: Meso4 and Xception architectures as described
- **Evaluation Pipeline**: Complete benchmarking suite with all proposed metrics
- **Robustness Testing**: Perturbation analysis following the paper's methodology
- **Performance Analysis**: FLOPS calculation and inference time measurement

## Additional Resources

### Face Detection
- **face_recognition library**: Used for robust face detection and landmark extraction
- Based on dlib's state-of-the-art face recognition built with deep learning

### Deep Learning Frameworks
- **TensorFlow/Keras**: Deep learning framework for model implementation
- **ImageIO**: Video frame extraction and processing

### Video Processing
- **FFmpeg**: Video codec support and frame extraction

## Course Information

- **Course**: EE656
- **Institution**: [Your Institution]
- **Date**: November 2025

## Citation

If you use this code in your research, please cite both the original paper and this implementation:

### BibTeX for the Paper
```bibtex
@inproceedings{deepfake_benchmark,
  title={Towards Benchmarking and Evaluating Deepfake Detection},
  author={[Authors]},
  booktitle={IEEE Conference},
  year={2024},
  organization={IEEE}
}
```

### BibTeX for this Implementation
```bibtex
@software{deepfake_detection_ee656,
  title={Deepfake Detection using Meso4 and Xception},
  author={Rudraksh919},
  year={2025},
  url={https://github.com/Rudraksh919/Benchmarking-Deepfake-Detection},
  note={Implementation based on IEEE paper "Towards Benchmarking and Evaluating Deepfake Detection"}
}
```

## Related Work

While this implementation focuses on the IEEE benchmarking paper, the field of deepfake detection builds upon several foundational works:

### Deepfake Detection Architectures
- MesoNet series (Meso4, MesoInception4)
- CNN-based detection methods
- Transfer learning approaches with Xception

### Face Manipulation Detection
- Face forensics research
- GAN-generated image detection
- Video manipulation detection

### Benchmarking Datasets
- FaceForensics++
- Celeb-DF
- DFDC (Deepfake Detection Challenge)
- WildDeepfake

## Acknowledgments

We acknowledge the authors of "Towards Benchmarking and Evaluating Deepfake Detection" for providing a comprehensive evaluation framework that guided this implementation.

---

For questions about the implementation, please open an issue on GitHub.  
For questions about the original paper, please contact the paper's authors through IEEE.
