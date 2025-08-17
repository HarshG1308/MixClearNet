import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import zipfile
import math

class TimeDomainEncoder(nn.Module):
    """Time-domain encoder with dilated convolutions as described in the paper"""
    def __init__(self, input_channels=1, output_channels=256):
        super(TimeDomainEncoder, self).__init__()
        
        # Input processing layer
        self.input_conv = nn.Conv1d(input_channels, 64, kernel_size=3, padding=1)
        self.input_bn = nn.BatchNorm1d(64)
        
        # Dilated convolution stack with dilation rates [1, 2, 4, 8, 16, 32, 64, 128]
        self.dilated_blocks = nn.ModuleList()
        for i in range(8):
            dilation = 2 ** i
            self.dilated_blocks.append(nn.Sequential(
                nn.Conv1d(64, 64, kernel_size=3, dilation=dilation, padding=dilation),
                nn.BatchNorm1d(64),
                nn.ReLU(),
            ))
        
        # Feature compression
        self.feature_compression = nn.Conv1d(64, output_channels, kernel_size=1)
    
    def forward(self, x):
        # x: (batch, 1, time_steps)
        x = F.relu(self.input_bn(self.input_conv(x)))
        
        # Apply dilated convolutions with residual connections
        for block in self.dilated_blocks:
            residual = x
            x = block(x) + residual
        
        # Feature compression
        x = self.feature_compression(x)
        return x.transpose(1, 2)  # (batch, time_steps, 256)

class SpectralProcessingModule(nn.Module):
    """Spectral processing module for magnitude and phase"""
    def __init__(self, n_fft=512, hop_length=128):
        super(SpectralProcessingModule, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        # Magnitude processing path - 3 residual CNN blocks
        self.mag_blocks = nn.ModuleList()
        for i in range(3):
            in_channels = 1 if i == 0 else 128
            self.mag_blocks.append(nn.Sequential(
                nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
            ))
        
        # Phase processing path - 2 lightweight CNN blocks
        self.phase_blocks = nn.ModuleList()
        for i in range(2):
            in_channels = 1 if i == 0 else 64
            self.phase_blocks.append(nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
            ))
    
    def forward(self, x):
        # x: (batch, time_steps)
        # Apply window to ensure proper STFT computation
        window = torch.hann_window(self.n_fft, device=x.device)
        
        # STFT
        stft = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length, 
                         window=window, return_complex=True, normalized=True)
        
        magnitude = torch.abs(stft).unsqueeze(1)  # (batch, 1, freq, time)
        phase = torch.angle(stft).unsqueeze(1)    # (batch, 1, freq, time)
        
        # Log magnitude spectrogram
        log_magnitude = torch.log(magnitude + 1e-8)
        
        # Process magnitude
        mag_features = log_magnitude
        for i, block in enumerate(self.mag_blocks):
            residual = mag_features if i > 0 else None
            mag_features = block(mag_features)
            if residual is not None:
                mag_features = mag_features + residual
            mag_features = F.relu(mag_features)
        
        # Process phase
        phase_features = phase
        for block in self.phase_blocks:
            phase_features = block(phase_features)
        
        # Concatenate magnitude and phase features
        spectral_features = torch.cat([mag_features, phase_features], dim=1)  # (batch, 192, freq, time)
        
        return spectral_features, stft

class CrossDomainFusion(nn.Module):
    """Cross-domain feature fusion with attention mechanism"""
    def __init__(self, time_dim=256, spec_dim=192, hidden_dim=512):
        super(CrossDomainFusion, self).__init__()
        
        # Dimension alignment
        self.time_upsample = nn.ConvTranspose1d(time_dim, time_dim, kernel_size=4, stride=2, padding=1)
        
        # Feature projection
        self.time_proj = nn.Linear(time_dim, hidden_dim)
        self.spec_proj = nn.Linear(spec_dim, hidden_dim)
        
        # Cross-attention
        self.sqrt_dim = math.sqrt(hidden_dim)
    
    def forward(self, time_features, spec_features):
        # time_features: (batch, time_steps, 256)
        # spec_features: (batch, 192, freq, time)
        
        # Upsample time features to match spectral resolution
        time_features = time_features.transpose(1, 2)  # (batch, 256, time_steps)
        time_features = self.time_upsample(time_features)  # Upsample
        time_features = time_features.transpose(1, 2)  # (batch, time_steps', 256)
        
        # Flatten spectral features
        batch, channels, freq, time = spec_features.shape
        spec_features = spec_features.permute(0, 3, 1, 2).flatten(2)  # (batch, time, channels*freq)
        
        # Ensure both features have the same time dimension
        min_time = min(time_features.size(1), spec_features.size(1))
        time_features = time_features[:, :min_time, :]
        spec_features = spec_features[:, :min_time, :]
        
        # Project to common space
        H_time = self.time_proj(time_features)  # (batch, time_steps', 512)
        H_spec = self.spec_proj(spec_features)  # (batch, time, 512)
        
        # Cross-attention
        A_time = F.softmax(torch.bmm(H_time, H_spec.transpose(1, 2)) / self.sqrt_dim, dim=-1)
        A_spec = F.softmax(torch.bmm(H_spec, H_time.transpose(1, 2)) / self.sqrt_dim, dim=-1)
        
        # Fused features
        fused_time = torch.bmm(A_time, H_spec)
        fused_spec = torch.bmm(A_spec, H_time)
        
        fused_features = torch.cat([fused_time, fused_spec], dim=-1)  # (batch, time, 1024)
        
        return fused_features

class TemporalModelingModule(nn.Module):
    """Bi-directional LSTM with temporal attention"""
    def __init__(self, input_dim=1024, hidden_dim=512, num_layers=2):
        super(TemporalModelingModule, self).__init__()
        
        # Bi-LSTM stack
        self.bilstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                             batch_first=True, bidirectional=True)
        
        # Temporal attention
        self.attention_dim = hidden_dim * 2  # bidirectional
        self.W_q = nn.Linear(self.attention_dim, self.attention_dim)
        self.W_k = nn.Linear(self.attention_dim, self.attention_dim)
        self.W_v = nn.Linear(self.attention_dim, self.attention_dim)
        
        self.sqrt_dim = math.sqrt(self.attention_dim)
    
    def forward(self, x):
        # x: (batch, time, 1024)
        
        # Bi-LSTM
        lstm_out, _ = self.bilstm(x)  # (batch, time, 1024)
        
        # Self-attention
        Q = self.W_q(lstm_out)
        K = self.W_k(lstm_out)
        V = self.W_v(lstm_out)
        
        attention_weights = F.softmax(torch.bmm(Q, K.transpose(1, 2)) / self.sqrt_dim, dim=-1)
        attended_features = torch.bmm(attention_weights, V)
        
        return attended_features

class SpeakerSeparationModule(nn.Module):
    """Speaker separation module with mask generation"""
    def __init__(self, input_dim=1024, num_speakers=2, freq_bins=257):
        super(SpeakerSeparationModule, self).__init__()
        self.num_speakers = num_speakers
        self.freq_bins = freq_bins
        
        # Speaker-specific pathways
        self.speaker_pathways = nn.ModuleList()
        for _ in range(num_speakers):
            pathway = nn.Sequential(
                nn.Linear(input_dim, 1024),
                nn.LayerNorm(1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Linear(512, freq_bins),
                nn.Sigmoid()
            )
            self.speaker_pathways.append(pathway)
    
    def forward(self, x):
        # x: (batch, time, 1024)
        
        masks = []
        for pathway in self.speaker_pathways:
            mask = pathway(x)  # (batch, time, freq_bins)
            masks.append(mask)
        
        # Mask refinement - soft normalization
        masks = torch.stack(masks, dim=1)  # (batch, num_speakers, time, freq)
        epsilon = 0.01
        refined_masks = (masks + epsilon) / (masks.sum(dim=1, keepdim=True) + self.num_speakers * epsilon)
        
        return refined_masks

class ReconstructionDecoder(nn.Module):
    """Reconstruction decoder with multi-resolution output"""
    def __init__(self, n_fft=512, hop_length=128):
        super(ReconstructionDecoder, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        # Learned weighting parameters for each speaker
        self.alpha = nn.Parameter(torch.ones(2) * 0.5)  # For 2 speakers
    
    def forward(self, masks, original_stft):
        # masks: (batch, num_speakers, time, freq)
        # original_stft: complex spectrogram
        
        separated_signals = []
        window = torch.hann_window(self.n_fft, device=masks.device)
        
        for i in range(masks.size(1)):
            mask = masks[:, i]  # (batch, time, freq)
            
            # Transpose mask to match STFT dimensions: (batch, freq, time)
            mask = mask.transpose(1, 2)
            
            # Apply mask to original complex spectrogram
            masked_stft = mask * original_stft
            
            # ISTFT reconstruction
            istft_signal = torch.istft(masked_stft, n_fft=self.n_fft, 
                                     hop_length=self.hop_length, window=window, normalized=True)
            
            separated_signals.append(istft_signal)
        
        return torch.stack(separated_signals, dim=1)

class MixClearNet(nn.Module):
    """Complete MixClearNet architecture as described in the paper"""
    def __init__(self, num_speakers=2, n_fft=512, hop_length=128):
        super(MixClearNet, self).__init__()
        
        self.num_speakers = num_speakers
        self.time_encoder = TimeDomainEncoder()
        self.spectral_processor = SpectralProcessingModule(n_fft, hop_length)
        self.cross_domain_fusion = CrossDomainFusion()
        self.temporal_modeling = TemporalModelingModule()
        self.speaker_separation = SpeakerSeparationModule(num_speakers=num_speakers)
        self.reconstruction_decoder = ReconstructionDecoder(n_fft, hop_length)
        
        # Composite loss weights
        self.loss_weights = {
            'si_snr': 1.0,
            'l1': 0.1,
            'perceptual': 0.05,
            'consistency': 0.02
        }
    
    def forward(self, x):
        # x: (batch, time_steps) - raw audio waveform
        
        # Ensure input is the right shape
        if x.dim() == 3 and x.size(1) == 1:
            x = x.squeeze(1)  # Remove channel dimension if present
        
        # Time-domain encoding
        time_features = self.time_encoder(x.unsqueeze(1))  # Add channel dim
        
        # Spectral processing
        spec_features, original_stft = self.spectral_processor(x)
        
        # Cross-domain fusion
        fused_features = self.cross_domain_fusion(time_features, spec_features)
        
        # Temporal modeling
        attended_features = self.temporal_modeling(fused_features)
        
        # Speaker separation
        masks = self.speaker_separation(attended_features)
        
        # Reconstruction
        separated_signals = self.reconstruction_decoder(masks, original_stft)
        
        return separated_signals
    
    def compute_si_snr_loss(self, predicted, target):
        """Compute SI-SNR loss"""
        # Ensure same length
        min_len = min(predicted.size(-1), target.size(-1))
        predicted = predicted[..., :min_len]
        target = target[..., :min_len]
        
        # Zero-mean
        predicted = predicted - torch.mean(predicted, dim=-1, keepdim=True)
        target = target - torch.mean(target, dim=-1, keepdim=True)
        
        # SI-SNR computation
        s_target = torch.sum(predicted * target, dim=-1, keepdim=True) * target / (torch.sum(target**2, dim=-1, keepdim=True) + 1e-8)
        e_noise = predicted - s_target
        
        si_snr = 10 * torch.log10((torch.sum(s_target**2, dim=-1) + 1e-8) / (torch.sum(e_noise**2, dim=-1) + 1e-8))
        
        return -torch.mean(si_snr)  # Negative because we want to maximize SI-SNR
    
    def compute_composite_loss(self, predicted, target):
        """Compute composite loss as described in the paper"""
        # SI-SNR loss (primary term)
        si_snr_loss = 0
        for i in range(predicted.size(1)):  # For each speaker
            si_snr_loss += self.compute_si_snr_loss(predicted[:, i], target[:, i])
        si_snr_loss /= predicted.size(1)
        
        # L1 loss
        l1_loss = F.l1_loss(predicted, target)
        
        # Simplified perceptual loss
        perceptual_loss = F.mse_loss(predicted, target)
        
        # Consistency loss
        consistency_loss = F.l1_loss(predicted, target)
        
        total_loss = (self.loss_weights['si_snr'] * si_snr_loss + 
                     self.loss_weights['l1'] * l1_loss + 
                     self.loss_weights['perceptual'] * perceptual_loss + 
                     self.loss_weights['consistency'] * consistency_loss)
        
        return total_loss

def separate_speakers(audio_path, output_folder):
    """
    Separates speakers from an audio file using the SepFormer model.

    Args:
        audio_path (str): Path to the input audio file.
        output_folder (str): Folder to save separated audio files.

    Returns:
        list: Paths to the separated audio files.
    """
    # Extract the model weights if the file is a zip archive
    model_weights_path = 'sepformer_model/data.pkl'  # Updated to match the actual file in the archive
    if zipfile.is_zipfile('sepformer_model.pth'):
        with zipfile.ZipFile('sepformer_model.pth', 'r') as zip_ref:
            zip_ref.extract(model_weights_path)

    # Load the pre-trained model
    model = torch.load(model_weights_path, weights_only=False)  # Explicitly set weights_only to False
    model.eval()

    # Load the audio file
    waveform, sample_rate = torchaudio.load(audio_path)

    # Process the audio with the model
    with torch.no_grad():
        separated_sources = model(waveform)

    # Save the separated audio files
    separated_files = []
    for i, source in enumerate(separated_sources):
        output_path = f"{output_folder}/speaker_{i + 1}.wav"
        torchaudio.save(output_path, source.unsqueeze(0), sample_rate)
        separated_files.append(output_path)

    return separated_files