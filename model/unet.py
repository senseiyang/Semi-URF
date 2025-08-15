from __future__ import division, print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import ptwt


class ConvBlock(nn.Module):
    """
    two convolution layers with batch norm and leaky relu
    """

    def __init__(self, in_channels, out_channels, dropout_p):
        super(ConvBlock, self).__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.conv_conv(x)


class DownBlock(nn.Module):
    """
    Downsampling followed by ConvBlock
    """

    def __init__(self, in_channels, out_channels, dropout_p):
        super(DownBlock, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels, dropout_p)

        )

    def forward(self, x):
        return self.maxpool_conv(x)


class CrossAttention(nn.Module):
    """
     CrossAttention Block - Channel Context Attention Module
    """

    def __init__(self, F_g, F_x):
        super().__init__()
        # MLP for processing feature x, flattening features and performing linear transformation
        self.mlp_x = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(F_x, F_x, kernel_size=1, bias=False),
        )

        # MLP for processing guidance feature g, flattening features and performing linear transformation
        self.mlp_g = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(F_g, F_x, kernel_size=1, bias=False),
        )

        # ReLU activation function, inplace=True performs operation directly on the original tensor, saving memory
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # Implementation of channel attention mechanism

        # Global average pooling on feature x to get the average value for each channel
        channel_att_x = self.mlp_x(x)

        # Similarly, global average pooling on guidance feature g
        channel_att_g = self.mlp_g(g)

        # Average the channel attention weights of the two features
        channel_att_sum = (channel_att_x + channel_att_g) / 2.0

        # Use sigmoid function to normalize attention weights to the range of 0-1
        scale = torch.sigmoid(channel_att_sum)

        # Multiply attention weights with original features to achieve channel weighting
        x_after_channel = x * scale

        # Process weighted features through ReLU activation function
        out = self.relu(x_after_channel)

        return out


class FastWT(nn.Module):
    """
    fast wavelet transform
    """

    def __init__(self, mode='high'):
        """
        Initialize wavelet transform module
        Args:
            mode: 'high' extracts high-frequency information, 'low' extracts low-frequency information
        """
        super(FastWT, self).__init__()
        self.mode = mode
        self.wavelet = 'haar'  # Use Haar wavelet basis

    def forward(self, x):
        """
        Forward propagation - processes all channels using batch processing
        Args:
            x: Input feature map [B, C, H, W]
        Returns:
            Transformed feature map [B, C, H, W]
        """
        B, C, H, W = x.shape

        # Ensure input dimensions are even
        if H % 2 != 0 or W % 2 != 0:
            pad_h = 0 if H % 2 == 0 else 1
            pad_w = 0 if W % 2 == 0 else 1
            x = F.pad(x, (0, pad_w, 0, pad_h))
            B, C, H, W = x.shape

        # Reshape tensor for batch processing
        # Merge batch and channel dimensions for processing all channels at once
        x_reshaped = x.view(B * C, 1, H, W)

        # Perform wavelet transform using ptwt - batch processing mode
        try:
            # Attempt direct batch processing
            coeffs = ptwt.wavedec2(x_reshaped, wavelet=self.wavelet, level=1)

            if self.mode == 'low':
                # Keep only low-frequency information, set high-frequency information to zero
                modified_coeffs = [coeffs[0], (torch.zeros_like(coeffs[1][0]),
                                               torch.zeros_like(coeffs[1][1]),
                                               torch.zeros_like(coeffs[1][2]))]
            else:  # 'high'
                # Keep only high-frequency information, set low-frequency information to zero
                modified_coeffs = [torch.zeros_like(coeffs[0]), coeffs[1]]

            # Reconstruct the signal using inverse wavelet transform
            reconstructed = ptwt.waverec2(modified_coeffs, wavelet=self.wavelet)

            # Ensure reconstructed dimensions match original input
            if reconstructed.shape[-2:] != (H, W):
                reconstructed = F.interpolate(reconstructed.unsqueeze(1),
                                              size=(H, W),
                                              mode='bilinear',
                                              align_corners=False).squeeze(1)

            # Reshape back to original shape
            output = reconstructed.view(B, C, H, W)

        except Exception as e:
            # If batch processing fails, fall back to chunked processing
            print(f"Batch wavelet transform failed, falling back to chunked processing: {e}")

            # Use torch.chunk to process input in multiple batches
            chunks = torch.chunk(x_reshaped, chunks=min(B * C, 10), dim=0)
            results = []

            for chunk in chunks:
                coeffs = ptwt.wavedec2(chunk, wavelet=self.wavelet, level=1)

                if self.mode == 'low':
                    modified_coeffs = [coeffs[0], (torch.zeros_like(coeffs[1][0]),
                                                   torch.zeros_like(coeffs[1][1]),
                                                   torch.zeros_like(coeffs[1][2]))]
                else:  # 'high'
                    modified_coeffs = [torch.zeros_like(coeffs[0]), coeffs[1]]

                reconstructed = ptwt.waverec2(modified_coeffs, wavelet=self.wavelet)

                if reconstructed.shape[-2:] != (H, W):
                    reconstructed = F.interpolate(reconstructed.unsqueeze(1),
                                                  size=(H, W),
                                                  mode='bilinear',
                                                  align_corners=False).squeeze(1)

                results.append(reconstructed)

            # Concatenate results
            reconstructed = torch.cat(results, dim=0)
            output = reconstructed.view(B, C, H, W)

        return output


class FEFF(nn.Module):
    """
    Frequency-Enhanced Feature Fusion
    """

    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p, bilinear=True):
        super(FEFF, self).__init__()
        self.bilinear = bilinear

        # Use optimized wavelet transform
        self.wavelet_high = FastWT(mode='high')
        self.wavelet_low = FastWT(mode='low')

        # Upsampling and channel adjustment
        if bilinear:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels1, in_channels2, kernel_size=2, stride=2)

        # Use CrossAttention for feature fusion
        self.cca = CrossAttention(F_g=in_channels2, F_x=in_channels2)

        # Final convolution block
        self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1, x2):
        # x1: Decoder features [B, 2*C, H, W]
        # x2: Encoder features [B, C, 2*H, 2*W]

        # 1. Feature extraction using wavelet transform
        x2_high = self.wavelet_high(x2)  # Extract high-frequency information
        x2_enhanced = x2 + x2_high  # Residual connection

        x1_low = self.wavelet_low(x1)  # Extract low-frequency information
        x1_enhanced = x1 + x1_low  # Residual connection

        # 2. Upsample decoder features
        if self.bilinear:
            x1_enhanced = self.conv1x1(x1_enhanced)
        x1_up = self.up(x1_enhanced)  # [B, C, 2*H, 2*W]

        # 3. Feature fusion using CrossAttention
        x2_att = self.cca(g=x1_up, x=x2_enhanced)

        # 4. Feature fusion
        x = torch.cat([x2_att, x1_up], dim=1)  # [B, 2*C, H, W]

        # 5. Final convolution
        return self.conv(x)  # [B, C, 2H, 2W]


class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        self.dropout = self.params['dropout']
        assert (len(self.ft_chns) == 5)
        self.in_conv = ConvBlock(
            self.in_chns, self.ft_chns[0], self.dropout[0])
        self.down1 = DownBlock(
            self.ft_chns[0], self.ft_chns[1], self.dropout[1])
        self.down2 = DownBlock(
            self.ft_chns[1], self.ft_chns[2], self.dropout[2])
        self.down3 = DownBlock(
            self.ft_chns[2], self.ft_chns[3], self.dropout[3])
        self.down4 = DownBlock(
            self.ft_chns[3], self.ft_chns[4], self.dropout[4])

    def forward(self, x):
        x0 = self.in_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        return [x0, x1, x2, x3, x4]


class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        assert (len(self.ft_chns) == 5)

        # Upsampling module using wavelet transform
        self.up1 = FEFF(
            self.ft_chns[4], self.ft_chns[3], self.ft_chns[3],
            dropout_p=0.0, bilinear=self.bilinear)

        self.up2 = FEFF(
            self.ft_chns[3], self.ft_chns[2], self.ft_chns[2],
            dropout_p=0.0, bilinear=self.bilinear)

        self.up3 = FEFF(
            self.ft_chns[2], self.ft_chns[1], self.ft_chns[1],
            dropout_p=0.0, bilinear=self.bilinear)

        self.up4 = FEFF(
            self.ft_chns[1], self.ft_chns[0], self.ft_chns[0],
            dropout_p=0.0, bilinear=self.bilinear)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class,
                                  kernel_size=3, padding=1)

    def forward(self, feature):
        '''
        x0: torch.Size([16, 16, 256, 256])
        x1: torch.Size([16, 32, 128, 128])
        x2: torch.Size([16, 64, 64, 64])
        x3: torch.Size([16, 128, 32, 32])
        x4: torch.Size([16, 256, 16, 16])
        up1 output: torch.Size([16, 128, 32, 32])
        up2 output: torch.Size([16, 64, 64, 64])
        up3 output: torch.Size([16, 32, 128, 128])
        up4 output: torch.Size([16, 16, 256, 256])
        Final output: torch.Size([16, 4, 256, 256])
        '''

        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)
        output = self.out_conv(x)
        return output


class FEUNet(nn.Module):
    def __init__(self, in_chns, class_num):
        super(FEUNet, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}

        self.encoder = Encoder(params)
        self.decoder = Decoder(params)

    def forward(self, x):
        feature = self.encoder(x)
        output = self.decoder(feature)
        return output
