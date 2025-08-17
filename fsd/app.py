from flask import Flask, request, render_template, send_file, jsonify
import os
import torch
import torchaudio
import numpy as np
from model import MixClearNet
import evaluation

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'separated_results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER

# Global model variable
model = None
device = None

def load_model():
    """Load the trained MixClearNet model"""
    global model, device
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MixClearNet(num_speakers=2).to(device)
    
    # Try to load trained weights
    model_paths = ["mixclearnet_best.pth", "mixclearnet_final.pth"]
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=device)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    print(f"Loaded trained model from {model_path}")
                    if 'best_si_snr' in checkpoint:
                        print(f"Model SI-SNR: {checkpoint['best_si_snr']:.2f} dB")
                else:
                    model.load_state_dict(checkpoint)
                    print(f"Loaded model weights from {model_path}")
                break
            except Exception as e:
                print(f"Failed to load {model_path}: {e}")
                continue
    else:
        print("No trained model found. Using random initialization.")
    
    model.eval()
    return model

def separate_speakers_mixclearnet(audio_path, output_folder):
    """
    Separate speakers using MixClearNet model
    """
    global model, device
    
    if model is None:
        load_model()
    
    try:
        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Ensure mono audio
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample to 16kHz if necessary
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
            sample_rate = 16000
        
        # Remove channel dimension and move to device
        waveform = waveform.squeeze(0).to(device)
        
        # Pad or truncate to reasonable length (max 10 seconds for demo)
        max_length = 10 * 16000
        if waveform.size(0) > max_length:
            waveform = waveform[:max_length]
        
        # Add batch dimension
        waveform = waveform.unsqueeze(0)
        
        # Separate speakers
        with torch.no_grad():
            separated_speakers = model(waveform)  # Shape: (1, num_speakers, time)
        
        # Save separated audio files
        separated_files = []
        for i in range(separated_speakers.size(1)):
            speaker_audio = separated_speakers[0, i].cpu()  # Remove batch dimension
            output_path = os.path.join(output_folder, f"speaker_{i + 1}.wav")
            torchaudio.save(output_path, speaker_audio.unsqueeze(0), sample_rate)
            separated_files.append(output_path)
        
        return separated_files
        
    except Exception as e:
        print(f"Error in speech separation: {e}")
        import traceback
        traceback.print_exc()
        return []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part', 400
    
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400
    
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Process the file with MixClearNet
        separated_files = separate_speakers_mixclearnet(filepath, app.config['RESULTS_FOLDER'])

        if not separated_files:
            return 'Error processing audio file', 500

        # Evaluate the separated files (simplified for demo)
        results = []
        for i, speaker_file in enumerate(separated_files):
            # Basic file info
            try:
                waveform, sr = torchaudio.load(speaker_file)
                duration = waveform.size(1) / sr
                
                speaker_results = {
                    'speaker': f'Speaker {i+1}',
                    'file': os.path.basename(speaker_file),
                    'duration': f'{duration:.2f}s',
                    'sample_rate': sr,
                    'quality': 'Processed'  # Placeholder
                }
                results.append(speaker_results)
            except Exception as e:
                print(f"Error analyzing {speaker_file}: {e}")
                
        # Generate simple visualization
        try:
            evaluation.save_comparative_visualization(results, filename='static/visualization.png')
        except:
            print("Could not generate visualization")

        return render_template('results.html', 
                             results=results, 
                             visualization='static/visualization.png',
                             separated_files=[os.path.basename(f) for f in separated_files])

@app.route('/process', methods=['POST'])
def process_audio():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Process with MixClearNet
        separated_files = separate_speakers_mixclearnet(filepath, app.config['RESULTS_FOLDER'])
        
        if separated_files:
            return jsonify({
                'success': True,
                'separated_files': [os.path.basename(f) for f in separated_files],
                'message': f'Successfully separated into {len(separated_files)} speakers'
            })
        else:
            return jsonify({'error': 'Failed to process audio'}), 500

@app.route('/download/<filename>')
def download_file(filename):
    """Download separated audio files"""
    return send_file(os.path.join(app.config['RESULTS_FOLDER'], filename), as_attachment=True)

@app.route('/model_info')
def model_info():
    """Get information about the loaded model"""
    global model
    
    if model is None:
        load_model()
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    info = {
        'model_name': 'MixClearNet',
        'architecture': 'Hybrid dual-domain (time + frequency)',
        'total_parameters': f'{total_params/1e6:.1f}M',
        'trainable_parameters': f'{trainable_params/1e6:.1f}M',
        'device': str(device),
        'num_speakers': 2,
        'sample_rate': '16kHz',
        'expected_si_snr': '16.8 dB (as per paper)'
    }
    
    return jsonify(info)

if __name__ == '__main__':
    print("Starting MixClearNet Web Interface...")
    print("Loading model...")
    load_model()
    print(f"Model loaded on {device}")
    print("Starting Flask server...")
    app.run(host='0.0.0.0', port=5001, debug=True)