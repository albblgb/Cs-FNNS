# Cs-FNNS
The official code for 'Cover-separable Fixed Neural Network Steganography via Deep Generative Models'

## Dependencies and Installation
- Python 3.8.13, PyTorch = 1.11.0
- Run the following commands in your terminal:

  `conda env create  -f env.yaml`

   `conda activate FNNS`


## Get Started
- Regarding resistance against detection,

   Run `python Cs-FNNS_AntiDetect.py` 

- Regarding resistance against JPEG compression,

   Change the code in `config.py`:  `line14:  secret_image_size = '128'`
  
   Run `python Cs-FNNS-JPEG.py`

- Regarding hiding multiple secret images for different receiver,
  
  Run `python Cs-FNNS_MUsers.py`

- Results will be saved in the "./result" folder.
    
## Others
- Pretrained SRNet model is large, so we upload it to [Google Drive](https://drive.google.com/drive/folders/1nZuBFTH0-bb9umTOO54Axap453N2mfa6?usp=sharing). Download it and place it in the specified location (refer to line 20 in the config.py file). After that, you can uncomment the part related to SRNet, bringing its gradient signals into the SPS optimization.

