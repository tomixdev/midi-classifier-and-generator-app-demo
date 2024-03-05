'''classify_midi_image.py'''
"""
=======================================================================================================================
-------------------------------Importing Libraries and miscellaneous Code Settings-------------------------------------
=======================================================================================================================
"""


import warnings
from torchvision.transforms.functional import InterpolationMode
import os
from PIL import Image
import torchvision
import params
import torchvision.transforms as transforms
import torch
import japanize_matplotlib
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from matplotlib import style
plt.style.use(['science', 'ieee'])

np.set_printoptions(suppress=True, precision=5)

if torch.cuda.is_available():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # !sudo nvidia-smi
    print(f"{device = }")
# elif torch.backends.mps.is_available():
#     device = torch.device("mps")
#     print (f"{device = }")
else:
    device = torch.device('cpu')
    print("No GPU device found. Running on CPU.")

plt.rcParams['font.size'] = 14
plt.rcParams['figure.figsize'] = (6, 6)
plt.rcParams['axes.grid'] = True
plt.style.use(['science', 'ieee'])
plt.rcParams["font.family"] = "Times New Roman"

"""
=======================================================================================================================
-------------------------------Load Pre-trained Model------------------------------------------------------------------
=======================================================================================================================
"""

# load the model
model = torchvision.models.resnet18(pretrained=True).to(device)
# Replace the last fully connected layer with a new one for grayscale image classification
num_classes = params.NUM_CLASSES_FOR_MIDI_IMG_CLASSIFIER
model.fc = torch.nn.Linear(512, num_classes, device=device)
model.load_state_dict(torch.load(params.PATH_TO_CLASSIFIER_MODEL, map_location=device))
model.to(device)
# set to evaluation mode
model.eval()


"""
=======================================================================================================================
-------------------------------Midi Classifier Function----------------------------------------------------------------
=======================================================================================================================
"""


def classify_midi_image(midi_img_path: str) -> int:
    assert isinstance(midi_img_path, str), \
        "The path should be a string" + f" | {midi_img_path = }"
    assert os.path.exists(midi_img_path), \
        "The file does not exist" + f" | {midi_img_path = }"
    assert midi_img_path.endswith(".png") or midi_img_path.endswith(".jpg"),\
        "The file should be a png or jpg file" + f" | {midi_img_path = }"

    # Define the transformation
    transform = transforms.Compose([
        # Convert image to RGB, removing alpha channel
        transforms.Lambda(lambda img: img.convert('RGB')),
        # Resize to the input size of the model
        transforms.Resize((224, 224), interpolation=InterpolationMode.NEAREST),
        transforms.ToTensor(),  # Convert to tensor
    ])

    # Transoform the image
    img = Image.open(midi_img_path)
    img = transform(img).unsqueeze(0)  # Add a batch dimension

    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)

    predicted_class = predicted.item()
    print(f"{predicted_class = }")
    return predicted_class


# tester code
if __name__ == "__main__":
    import os
    import midi_preprocessing
    from datetime import datetime

    midi_file_full_path = "sample_midi_files/Debussy-Claude-Images-6th(id=151dac4ecb3e1d62e628768359a67adff8a2c483).mid"
    # convert this to full path
    midi_file_full_path = midi_preprocessing._format_a_midi_file_name(midi_file_full_path)
    midi_file_full_path = os.path.abspath(midi_file_full_path)

    # get current time with milliseconds
    now = datetime.now()
    img_path = midi_preprocessing.convert_midi_file_to_bw_piano_roll_img_with_pitch_as_row(
        original_midi_file_full_path=midi_file_full_path,
        midi_img_path=params.TMP_MIDI_IMG_FOLDER + f"{now.strftime('%Y%m%d%H%M%S%f')}.png",
    )

    img_path = midi_preprocessing.resize_image_to_square_shape(img_path)
    predicted_class = classify_midi_image(midi_img_path=img_path)
