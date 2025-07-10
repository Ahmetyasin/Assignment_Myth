# Sketch Generation via Diffusion Models 🎨

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> Sequential stroke generation using Denoising Diffusion Probabilistic Models (DDPM) on the Quick, Draw! dataset

## 🚀 Overview

This project implements a novel approach to sketch generation by applying diffusion models to sequential drawing data. Unlike traditional image-based generation, our method generates sketches stroke-by-stroke, mimicking the natural human drawing process.

### Key Features

- **Sequential Generation**: Produces sketches stroke-by-stroke in temporal order
- **Transformer Architecture**: Uses attention mechanisms for capturing long-range stroke dependencies  
- **Multi-Category Support**: Trained separate models for cats, buses, and rabbits
- **Robust Pipeline**: 100% generation success rate with advanced post-processing
- **Comprehensive Evaluation**: Both quantitative (FID/KID) and qualitative assessment

## 📊 Results

| Category | FID Score | KID Score | Generated Samples | Status |
|----------|-----------|-----------|-------------------|---------|
| **Rabbit** | **0.0381** | 0.0227 | 10/10 | ✅ Success |
| **Cat** | 0.0407 | 0.0222 | 10/10 | ✅ Success |
| **Bus** | 0.0418 | 0.0221 | 10/10 | ✅ Success |

- **100% Generation Success Rate** across all categories
- **Low FID/KID scores** indicating good distribution similarity to real sketches
- **Stable training convergence** with final losses around 0.04-0.05

## 🎯 Sample Outputs

### Generated Sketches
![Cat Sketches](results/cat_generated_1.png) ![Bus Sketches](results/bus_generated_1.png) ![Rabbit Sketches](results/rabbit_generated_1.png)

### Training Convergence
![Training Loss](results/cat_training_loss.png)

### Animated Generation
![Generation Process](results/cat_generation.gif)

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.9+
- CUDA (optional, for GPU acceleration)

### Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/sketch-diffusion.git
cd sketch-diffusion

# Install dependencies
pip install torch torchvision matplotlib numpy scipy tqdm pillow

# Install Google Cloud SDK for data download
pip install gsutil
```

## 📋 Usage

### Quick Start
Run the complete pipeline in a Jupyter notebook:

```python
# The main implementation is contained in one comprehensive notebook cell
# Simply run the complete code provided in the notebook
```

### Training Custom Models
```python
# Train a model for a specific category
model, losses = train_category_model(
    category='cat',
    train_sketches=train_data,
    num_epochs=30,
    batch_size=32
)
```

### Generating Sketches
```python
# Generate new sketches
generator = AdvancedSketchGenerator(model, processor, device)
generated_sequences = generator.generate_sketches(num_samples=10)
sketches = [generator.sequence_to_strokes_advanced(seq) for seq in generated_sequences]
```

### Creating Visualizations
```python
# Plot generated sketches
generator.plot_sketch(sketches[0], title="Generated Sketch")

# Create animated GIF
generator.create_animated_gif(sketches[0], "animation.gif")
```

## 🏗️ Architecture

### Model Components

1. **Data Processing**
   - Converts stroke sequences to 5D vectors: `[dx, dy, pen_down, pen_up, end_sketch]`
   - Normalizes coordinates to [-1, 1] range
   - Handles variable-length sequences with padding

2. **Diffusion Model**
   - DDPM with linear beta schedule (1000 timesteps)
   - Transformer-based U-Net for noise prediction
   - Time embeddings and positional encodings

3. **Generation Pipeline**
   - Multi-method stroke extraction with fallbacks
   - Advanced post-processing for continuous-to-discrete conversion
   - Robust coordinate handling and bounds checking

### Technical Innovation

- **Sequential Representation**: Novel application of diffusion to stroke-based data
- **Transformer U-Net**: Attention mechanisms for capturing stroke dependencies
- **Robust Post-processing**: Multiple extraction methods ensure generation success

## 📈 Evaluation

### Quantitative Metrics
- **FID (Fréchet Inception Distance)**: Measures distribution similarity
- **KID (Kernel Inception Distance)**: Robust alternative to FID
- Both metrics calculated on rendered 64x64 binary images

### Qualitative Assessment
- Stroke coherence and naturalness
- Category-specific pattern recognition
- Generation consistency and reliability

## 🔬 Technical Details

### Method Choice Rationale
- **Diffusion Models**: Chosen for stability and high-quality generation
- **Sequential Processing**: Captures temporal drawing dynamics
- **Transformer Architecture**: Better than CNNs for sequence modeling

### Training Configuration
- **Epochs**: 30 per category
- **Batch Size**: 32
- **Learning Rate**: 2e-4 with AdamW optimizer
- **Dataset**: 2000 training + 500 test samples per category

### Key Challenges Solved
- **Continuous-to-Discrete Conversion**: Major technical challenge in sketch generation
- **Variable Length Sequences**: Handled through padding and special tokens
- **Coordinate Stability**: Differential coordinates with proper normalization

## 🚀 Future Improvements

### Immediate Enhancements
- [ ] Extended training with full dataset (50K+ samples)
- [ ] Latent diffusion for computational efficiency
- [ ] Advanced sampling methods (DDIM, DPM-Solver)
- [ ] Better stroke extraction algorithms

### Advanced Features
- [ ] Text-to-sketch generation
- [ ] Multi-category conditioning
- [ ] Style transfer capabilities
- [ ] Interactive sketch completion

## 📚 References

1. Ho, J., Jain, A., & Abbeel, P. (2020). Denoising diffusion probabilistic models. *NeurIPS*.
2. Ha, D., & Eck, D. (2017). A neural representation of sketch drawings. *arXiv:1704.03477*.
3. Vaswani, A., et al. (2017). Attention is all you need. *NeurIPS*.
4. [Quick, Draw! Dataset](https://github.com/googlecreativelab/quickdraw-dataset)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup
```bash
# Fork the repo and create a feature branch
git checkout -b feature/amazing-feature

# Make your changes and test them
python -m pytest tests/

# Commit and push
git commit -m 'Add amazing feature'
git push origin feature/amazing-feature
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Google Creative Lab for the Quick, Draw! dataset
- The diffusion models community for foundational research
- PyTorch team for the excellent deep learning framework

## 📧 Contact

- **Author**: [Ahmet Yasin Aytar]
- **Email**: [aytarahmetyasin@gmail.com]
- **Project Link**: [[https://github.com/yourusername/sketch-diffusion](https://github.com/yourusername/sketch-diffusion](https://github.com/Ahmetyasin/Assignment_Myth))

---
