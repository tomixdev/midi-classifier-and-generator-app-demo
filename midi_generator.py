import params

import PIL.Image as Image
import copy
import pretty_midi

import torch
import torch.nn as nn

import datetime
import scienceplots

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
# plt.style.use('science')
plt.style.use(['science', 'ieee'])
plt.rcParams['font.size'] = 14
plt.rcParams['figure.figsize'] = (6, 6)
plt.rcParams['axes.grid'] = True
np.set_printoptions(suppress=True, precision=5)

########################################################################################################################
# Device Setting
if torch.cuda.is_available():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"{device = }")
# elif torch.backends.mps.is_available():
#     device = torch.device("mps")
#     print (f"{device = }")
else:
    device = torch.device('cpu')
    print("No GPU device found. Running on CPU.")
########################################################################################################################

########################################################################################################################
# Parameters
NUM_CLASSES = params.NUM_CLASSES_FOR_MIDI_GENERATOR
# Length and width of the image that is fed to the discriminator. Images are resized to a square shape.
RESIZED_IMAGE_SIZE = 112
A_NUM = int(round(RESIZED_IMAGE_SIZE / 4))
########################################################################################################################


class TwoConvBlock_2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, padding="same")
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.rl = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, padding="same")
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.rl(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.rl(x)
        return x


class Generator(nn.Module):  # 生成器
    def __init__(self):
        super().__init__()
        self.l = nn.Linear(100 + NUM_CLASSES, A_NUM * A_NUM)
        self.dropout = nn.Dropout(0.2)
        self.TCB1 = TwoConvBlock_2D(1, 512)
        self.TCB2 = TwoConvBlock_2D(512, 256)
        self.TCB3 = TwoConvBlock_2D(256, 128)
        self.UC1 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.UC2 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(128, 1, kernel_size=2, padding="same")

    def forward(self, x, y):
        print("---------------------- Generator Forward ----------------------------------")
        y = torch.nn.functional.one_hot(
            y.long(), num_classes=NUM_CLASSES).to(torch.float32)
        x = torch.cat([x, y], dim=1)
        x = self.dropout(x)
        x = self.l(x)
        x = torch.reshape(x, (-1, 1, A_NUM, A_NUM))
        x = self.TCB1(x)
        x = self.UC1(x)
        x = self.TCB2(x)
        x = self.UC2(x)
        x = self.TCB3(x)
        x = self.conv1(x)
        x = torch.sigmoid(x)

        return x


def convert_greyscale_img_to_midi_and_save(img_path,
                                           midi_path,
                                           midi_velocity_threshold=25,
                                           duraion_per_grid_in_s=0.5,
                                           lowest_midi_pitch=21,
                                           highest_midi_pitch=108
                                           ):
    """
    :param img_path: str
    :param midi_path: str
    :param lowest_midi_pitch: int
    :param highest_midi_pitch: int
    :return:
    """
    assert isinstance(img_path, str) and img_path.endswith(".png") or img_path.endswith('jpg'), \
        f"{type(img_path) = } | {img_path = } "
    assert isinstance(midi_path, str) and midi_path.endswith(".mid"), \
        f"{type(midi_path) = } | {midi_path = } "
    assert isinstance(duraion_per_grid_in_s, float), type(
        duraion_per_grid_in_s)
    assert isinstance(midi_velocity_threshold, int), type(
        midi_velocity_threshold)
    assert isinstance(lowest_midi_pitch, int), type(lowest_midi_pitch)
    assert isinstance(highest_midi_pitch, int), type(highest_midi_pitch)

    ############################################################################################
    # convert image to numpy array -------------------------------------------------------------
    ############################################################################################
    img = Image.open(img_path)

    # remove alpha channel
    img = img.convert('RGB')

    # convert to numpy array
    img = np.asarray(img)

    # show img
    plt.imshow(img)
    # print("--------- Basic Information about Image ------------ ")
    # print(f"{img.shape = }")
    # print(f"{np.max(img) = }")
    # print(f"{np.min(img) = }")
    # print(f"{np.mean(img) = }")

    ############################################################################################
    # pixel value is between 0-255. convert to velocities 0-127 --------------------------------
    ############################################################################################
    img = img / 255.0 * 127.0

    ############################################################################################
    # Filter pixel value (= velocity values) that are too small (i.e. set small pixel values to zero) ----------------
    ############################################################################################
    for row_num in range(img.shape[0]):
        for col_num in range(img.shape[1]):
            velocity = img[row_num, col_num, 0]
            if velocity < midi_velocity_threshold:
                img[row_num, col_num, 0] = 0
                img[row_num, col_num, 1] = 0
                img[row_num, col_num, 2] = 0

    # cut zeros in upper and lower sides and create a new image --------------------------------
    the_first_row_num_with_nonzero = None
    for i in range(img.shape[0]):
        if np.max(img[i, :, 0]) > 0:
            the_first_row_num_with_nonzero = i
            break

    the_last_row_num_with_nonzero = None
    i = img.shape[0] - 1
    while i >= 0:
        if np.max(img[i, :, 0]) > 0:
            the_last_row_num_with_nonzero = i
            break
        i -= 1

    # print(f"{the_first_row_num_with_nonzero = }")
    # print(f"{the_last_row_num_with_nonzero = }")

    zero_cut_image = copy.deepcopy(
        img[the_first_row_num_with_nonzero:the_last_row_num_with_nonzero, :, :])
    # print(f"{zero_cut_image.shape = }")

    #############################################################################################
    # scale velocity values to 0-127 ------------------------------------------------------------
    ############################################################################################
    original_min = np.min(zero_cut_image[:, :, 0])
    original_max = np.max(zero_cut_image[:, :, 0])
    zero_cut_image = (zero_cut_image - original_min) / \
        (original_max - original_min) * 127.0

    ############################################################################################
    # create a list of note tuples --------------------------------------------------------------
    ############################################################################################
    # (midi_pos_in_s, midi_pitch, midi_velocity, note_duration_in_s)
    list_of_note_tuples = []
    for row_num in range(zero_cut_image.shape[0]):
        for col_num in range(zero_cut_image.shape[1]):
            # note timing
            a_midi_pos_in_s = col_num * duraion_per_grid_in_s

            # pitch
            # a_midi_pitch = int(127 - row_num*(127.0/zero_cut_image.shape[0]))
            n_of_available_pitches = highest_midi_pitch - lowest_midi_pitch + 1
            a_midi_pitch = n_of_available_pitches - row_num * (
                n_of_available_pitches / zero_cut_image.shape[0]) + lowest_midi_pitch
            a_midi_pitch = int(a_midi_pitch)

            # velocity
            a_midi_velocity = float(
                round(zero_cut_image[row_num, col_num, 0], 3))

            #
            note_duration_in_s = duraion_per_grid_in_s
            while row_num < zero_cut_image.shape[0] - 1 and zero_cut_image[row_num + 1, col_num, 0] > 0:
                note_duration_in_s += duraion_per_grid_in_s
                row_num += 1

            if a_midi_velocity == 0:
                continue
            else:
                a_note_tuple = (a_midi_pos_in_s, a_midi_pitch,
                                a_midi_velocity, note_duration_in_s)
                list_of_note_tuples.append(a_note_tuple)

    # print(f"{list_of_note_tuples = }")

    ############################################################################################
    # create a midi file and save
    ############################################################################################
    a_midi = pretty_midi.PrettyMIDI()
    an_instrument = pretty_midi.Instrument(program=0)
    for a_note_tuple in list_of_note_tuples:
        try:
            a_note = pretty_midi.Note(
                pitch=a_note_tuple[1],
                velocity=round(a_note_tuple[2]),
                start=a_note_tuple[0],
                end=a_note_tuple[0] + a_note_tuple[3],
            )
            an_instrument.notes.append(a_note)
        except Exception as e:
            print(e)
            print(f"Not Created {midi_path}")

    a_midi.instruments.append(an_instrument)
    a_midi.write(midi_path)


def generate_midi_img(generated_img_path, label=1, randomly_choose_the_most_condensed_img=True):
    assert isinstance(label, int), f"{label = } | {type(label) = }"
    assert label in params.CLASSIFICATION_CLASS_LABELS.keys(), f"{label = }"
    assert isinstance(generated_img_path, str), f"{generated_img_path = }"
    assert generated_img_path.endswith(".png") or generated_img_path.endswith(
        ".jpeg"), f"{generated_img_path = }"

    model = Generator()
    model.load_state_dict(torch.load(params.PATH_TO_GENERATOR_MODEL))
    model.eval()
    model.to(device)

    if randomly_choose_the_most_condensed_img:
        generated_img_list = []
        pixel_value_sum_list = []
        label_int_value = label
        for i in range(20):
            label = np.array([label_int_value])
            with torch.no_grad():
                noise = torch.randn((1, 100), dtype=torch.float32).to(device)
                label = torch.from_numpy(label)
                generated_img = model(noise, label)

            # save image to a file
            generated_img = generated_img.squeeze(0).detach().cpu().numpy()
            generated_img = generated_img * 255
            generated_img = generated_img.astype(np.uint8)
            generated_img = generated_img.reshape(
                generated_img.shape[1], generated_img.shape[2])
            generated_img_list.append(generated_img)
            pixel_value_sum_list.append(np.sum(generated_img))

        # pick the most condensed image
        generated_img = generated_img_list[np.argmax(pixel_value_sum_list)]
        generated_img = Image.fromarray(generated_img)
        generated_img.save(generated_img_path)
    else:
        label = np.array([label])
        with torch.no_grad():
            noise = torch.randn((1, 100), dtype=torch.float32).to(device)
            label = torch.from_numpy(label)
            generated_img = model(noise, label)

        # save image to a file
        generated_img = generated_img.squeeze(0).detach().cpu().numpy()
        generated_img = generated_img * 255
        generated_img = generated_img.astype(np.uint8)
        generated_img = generated_img.reshape(
            generated_img.shape[1], generated_img.shape[2])
        generated_img = Image.fromarray(generated_img)
        generated_img.save(generated_img_path)


# tester code
if __name__ == "__main__":
    now = datetime.datetime.now()
    new_midi_img_path = params.TMP_MIDI_IMG_FOLDER + \
        f"{now.strftime('%Y%m%d%H%M%S%f')}.png"
    new_midi_path = params.TMP_MIDI_IMG_FOLDER + \
        f"{now.strftime('%Y%m%d%H%M%S%f')}.mid"

    generate_midi_img(generated_img_path=new_midi_img_path, label=1)
    convert_greyscale_img_to_midi_and_save(
        img_path=new_midi_img_path, midi_path=new_midi_path)
