import torch
from data_loader import load_dataset
from model import MixClearNet
from training import train_model
import os

def get_device():
    """Get the appropriate device for computation."""
    if torch.cuda.is_available():
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        return torch.device("cuda")
    else:
        print("CUDA not available. Falling back to CPU.")
        return torch.device("cpu")

def test_model_architecture():
    """Test the MixClearNet architecture with dummy data."""
    print("Testing MixClearNet architecture...")
    
    device = torch.device("cpu")  # Use CPU for testing
    model = MixClearNet(num_speakers=2)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model created successfully!")
    print(f"Total parameters: {total_params/1e6:.1f}M")
    print(f"Trainable parameters: {trainable_params/1e6:.1f}M")
    
    # Test with dummy input
    batch_size = 2
    sequence_length = 4 * 16000  # 4 seconds at 16kHz
    dummy_input = torch.randn(batch_size, sequence_length)
    
    print(f"Testing with input shape: {dummy_input.shape}")
    
    try:
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"Output shape: {output.shape}")  # Should be (batch, num_speakers, time)
        print("✅ Model architecture test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Model architecture test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=== MixClearNet: Deep Learning Framework for Speech Separation ===")
    print("Cocktail Party Problem Solution")
    print("-" * 60)
    
    device = get_device()
    print(f"Running on device: {device}")
    
    # Test model architecture first
    if not test_model_architecture():
        print("Model architecture test failed. Please check the implementation.")
        return
    
    # Test dataset loading
    print("\n" + "="*60)
    print("Testing dataset loading...")
    
    dataset_name = "si_dt_05"
    print(f"Loading dataset: {dataset_name}")
    
    try:
        # Test with mixture dataset for speech separation
        dataloader = load_dataset(dataset_name, batch_size=4, use_mixtures=True)
        print(f"✅ Dataset loaded successfully: {len(dataloader)} batches")
        
        # Display first few batches
        print("\nProcessing sample batches:")
        for i, (mixed_audio, target_speakers) in enumerate(dataloader):
            print(f"Batch {i + 1}:")
            print(f"  Mixed audio shape: {mixed_audio.shape}")
            print(f"  Target speakers shape: {target_speakers.shape}")
            
            if i >= 2:  # Limit to first 3 batches
                break
                
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        print("Creating demo data instead...")
        
        # Create some demo data for testing
        try:
            from data_loader import create_wsj0_2mix_style_data
            demo_clean_dir = "wsj0/si_dt_05"
            demo_output_dir = "demo_mixtures"
            
            if os.path.exists(demo_clean_dir):
                create_wsj0_2mix_style_data(demo_clean_dir, demo_output_dir, num_mixtures=10)
                print(f"✅ Created demo mixtures in {demo_output_dir}")
            else:
                print(f"❌ No data available at {demo_clean_dir}")
        except Exception as demo_e:
            print(f"❌ Demo data creation failed: {demo_e}")
    
    # Model summary
    print("\n" + "="*60)
    print("MixClearNet Model Summary:")
    print("- Architecture: Hybrid dual-domain (time + frequency)")
    print("- Components: 5 main modules as described in paper")
    print("- Features: Cross-domain attention, temporal modeling")
    print("- Target: State-of-the-art speech separation performance")
    print("- Expected SI-SNR: 16.8 dB (as reported in paper)")
    
    # Check for existing trained models
    print("\n" + "="*60)
    print("Checking for trained models...")
    
    model_files = [
        "mixclearnet_best.pth",
        "mixclearnet_final.pth",
        "sepformer_model.pth"
    ]
    
    for model_file in model_files:
        if os.path.exists(model_file):
            print(f"✅ Found: {model_file}")
            try:
                # Try to load and inspect the model
                checkpoint = torch.load(model_file, map_location='cpu')
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    print(f"   - Type: Training checkpoint")
                    if 'epoch' in checkpoint:
                        print(f"   - Epoch: {checkpoint['epoch']}")
                    if 'best_si_snr' in checkpoint:
                        print(f"   - Best SI-SNR: {checkpoint['best_si_snr']:.2f} dB")
                else:
                    print(f"   - Type: Model weights only")
            except Exception as e:
                print(f"   - Warning: Could not inspect {model_file}: {e}")
        else:
            print(f"❌ Not found: {model_file}")
    
    print("\n" + "="*60)
    print("Setup completed!")
    print("Next steps:")
    print("1. Run 'python training.py' to train the MixClearNet model")
    print("2. Run 'python evaluation.py' to evaluate performance")
    print("3. Run 'python app.py' to start the web interface")
    print("4. Check generated LaTeX tables and plots for paper results")

if __name__ == "__main__":
    main()