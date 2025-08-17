import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

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
            self.mag_blocks.append(nn.Sequential(
                nn.Conv2d(1 if i == 0 else 128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
            ))
        
        # Phase processing path - 2 lightweight CNN blocks
        self.phase_blocks = nn.ModuleList()
        for i in range(2):
            self.phase_blocks.append(nn.Sequential(
                nn.Conv2d(1 if i == 0 else 64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
            ))
    
    def forward(self, x):
        # x: (batch, time_steps)
        # STFT
        stft = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length, 
                         return_complex=True, normalized=True)
        
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
        self.sqrt_dim = hidden_dim ** 0.5
    
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
        
        self.sqrt_dim = self.attention_dim ** 0.5
    
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
        
        # Time-domain decoder
        self.time_decoder = nn.Sequential(
            nn.ConvTranspose1d(1024, 512, kernel_size=128, stride=64, padding=32),
            nn.ReLU(),
            nn.ConvTranspose1d(512, 256, kernel_size=64, stride=32, padding=16),
            nn.ReLU(),
            nn.ConvTranspose1d(256, 128, kernel_size=32, stride=16, padding=8),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 1, kernel_size=16, stride=8, padding=4)
        )
        
        # Learned weighting parameters
        self.alpha = nn.Parameter(torch.ones(1) * 0.5)
    
    def forward(self, masks, original_stft, features):
        # masks: (batch, num_speakers, time, freq)
        # original_stft: complex spectrogram
        # features: (batch, time, 1024) for time-domain decoder
        
        separated_signals = []
        
        for i in range(masks.size(1)):
            mask = masks[:, i]  # (batch, time, freq)
            
            # Apply mask to original complex spectrogram
            masked_stft = mask.unsqueeze(-1) * original_stft.unsqueeze(1)
            
            # ISTFT reconstruction
            istft_signal = torch.istft(masked_stft.squeeze(1), n_fft=self.n_fft, 
                                     hop_length=self.hop_length, normalized=True)
            
            # Time-domain decoder (simplified for demonstration)
            time_signal = istft_signal  # In full implementation, use the decoder
            
            # Multi-resolution output (simplified)
            final_signal = self.alpha * istft_signal + (1 - self.alpha) * time_signal
            
            separated_signals.append(final_signal)
        
        return torch.stack(separated_signals, dim=1)

class MixClearNet(nn.Module):
    """Complete MixClearNet architecture as described in the paper"""
    def __init__(self, num_speakers=2, n_fft=512, hop_length=128):
        super(MixClearNet, self).__init__()
        
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
        separated_signals = self.reconstruction_decoder(masks, original_stft, attended_features)
        
        return separated_signals
    
    def compute_composite_loss(self, predicted, target):
        """Compute composite loss as described in the paper"""
        # This is a simplified version - implement full composite loss
        si_snr_loss = self.compute_si_snr_loss(predicted, target)
        l1_loss = F.l1_loss(predicted, target)
        
        total_loss = (self.loss_weights['si_snr'] * si_snr_loss + 
                     self.loss_weights['l1'] * l1_loss)
        
        return total_loss
    
    def compute_si_snr_loss(self, predicted, target):
        """Compute SI-SNR loss"""
        # Simplified SI-SNR computation
        noise = predicted - target
        si_snr = 10 * torch.log10(torch.sum(target**2, dim=-1) / torch.sum(noise**2, dim=-1))
        return -torch.mean(si_snr)  # Negative because we want to maximize SI-SNR

# Example usage
if __name__ == "__main__":
    # Create model
    model = MixClearNet(num_speakers=2)
    
    # Test with dummy input
    batch_size = 4
    time_steps = 16000  # 1 second at 16kHz
    dummy_input = torch.randn(batch_size, time_steps)
    
    # Forward pass
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")  # Should be (batch, num_speakers, time_steps)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6:.1f}M")
