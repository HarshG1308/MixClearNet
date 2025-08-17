# MixClearNet

MixClearNet is a deep learning-based framework for audio source separation and speech enhancement. This repository contains the implementation of the model, training scripts, evaluation metrics, and a user-friendly web interface for interacting with the model.

## Features
- **Model**: Implements a SepFormer-based architecture for source separation.
- **Training**: Scripts for training the model on the WSJ0 dataset.
- **Evaluation**: Metrics such as SI-SDRi, PESQ, WER, and EER.
- **Visualization**: Generates visualizations of model performance.
- **Web Interface**: A Flask-based interface for uploading audio files and viewing results.

## Repository Structure
```
├── app.py               # Flask application for the web interface
├── data_loader.py       # Data loading utilities
├── Dockerfile           # Docker configuration for containerization
├── evaluation.py        # Evaluation metrics and visualization
├── main.py              # Main script for running the model
├── model.py             # Model architecture
├── README.md            # Project documentation
├── requirements.txt     # Python dependencies
├── results.tex          # LaTeX file for evaluation results
├── sepformer_model.pth  # Pre-trained model weights
├── test_utils.py        # Testing utilities
├── training.py          # Training script
├── utils.py             # Helper functions
├── static/              # Static files (e.g., visualizations)
├── templates/           # HTML templates for the web interface
├── uploads/             # Directory for uploaded audio files
└── wsj0/                # WSJ0 dataset directory
```

## Getting Started

### Prerequisites
- Python 3.8 or higher
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

### Running the Web Interface
1. Start the Flask application:
   ```bash
   python3 app.py
   ```
2. Open your browser and navigate to `http://127.0.0.1:5001/`.
3. Upload an audio file and view the results.

### Training the Model
1. Prepare the WSJ0 dataset and place it in the `wsj0/` directory.
2. Run the training script:
   ```bash
   python3 training.py
   ```

### Evaluation
1. Run the evaluation script:
   ```bash
   python3 evaluation.py
   ```
2. Results will be saved in `results.tex` and visualizations in `static/`.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
- WSJ0 dataset for training and evaluation.
- SepFormer architecture for source separation.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.