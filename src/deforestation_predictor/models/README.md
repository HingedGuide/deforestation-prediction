# Overview
This library contains Pytorch implementations of several Deep Learning (DL) model architectures. 

## Models included
| Model      | Type        | Dimensionality  | Key Characteristic                                      |
|------------|-------------|-----------------|---------------------------------------------------------|
| ResUNet    | CNN         | 2D              | Flattens time into channels                             |
| ResUNet3D  | CNN         | 3D              | Uses volumetric convolutions to presever temporal depth |
| ConvLSTM3D | Hybrid      | 3D + RNN        | 3D encoder with a Convolutional LSTM bridge             | 
| ViViT      | Transformer | Attention-based | Factorized spatial and temporal attention               |

# Architecture details
**1. Residual Blocks (`ResidualBlock` & `ResidualBlcok3D`)** \
These are the fundamental building blocks of the ResUNet variants. They use skip connections to tackle the vanishing gradient problem. 
* 2D Version: Standard 3x3 convolutions.
* 3D Version: 3x3x3 convolutions, this is configured so that it downsamples teh spatial dimension while preserving the temporal dimension

**2. ResUNet & ResUNet3D** \
The ResUNet follows a standard Encoder-Decoder structure with skip connectinos between the corresponding levels
* ResUNet (2D): Collapses the temporal dimension T into the channel dimension C. This means that the input shape become [Batch, Channels x Time, Height, Width]
* ResUNet3D: Processes the input as a 5D tensor [Batch, Channels, Time, Height, Width], which should allow it to capture the spatio-temporal features simultaneously.

**3. ConvLSTM3D (Hybrid model)**\
This model combines the feature extraction of the 3D CNNs with the sequential memory that LSTMs have.
* Encoder: 3D Residual blocks
* Bridge: A `ConvLSTMCell` that processes the temporal sequence of the deepest feature maps
* Decoder: A standard 2D Decoder that produces the segmentation map based on the final hidden state of the LSTM

**4. ViViT (Video Vision Transfromer)** \ 
A factorized transformer approach for video segmentation
* Tubelet embedding: The video is divided into 3D patches (tubelets) and projected into an embedding space
* Factorize attention: To reduce the computational complexity, the model first applies spatal attentino (within the frames) and then the temproal attention (across frames)
* Progressive decoder: A series of transposed convolutions upsample the transformer tokens back to the original image resolution

# Usage
* Python 3.8+
* PyTorch 1.10+

All models except the 2D ResUNet expect a 5D tensor as input

| Dimension | Description                                                     |
|-----------|-----------------------------------------------------------------|
| B         | Batch size                                                      |
| C         | Input Channels (basically all the static and dynamic variables) |
| T         | Time Steps                                                      |
| H         | Height                                                          |
| W         | Width                                                           |

The output is a 4D tensor [Batch, Channels, Height, Width] which represents the pixel-wise class logits for the final time step or the sequence average.
