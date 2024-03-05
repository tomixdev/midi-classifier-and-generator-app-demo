import sys
import pandas as pd

import params
import util as u
import pretty_midi
import numpy as np
import matplotlib.pyplot as plt
import torch
import re
import os
import traceback


def convert_midi_file_to_bw_piano_roll_img_with_pitch_as_row(original_midi_file_full_path, midi_img_path):
    assert isinstance(original_midi_file_full_path, str), \
        "original_midi_file_full_path must be a string but is {}".format(
            type(original_midi_file_full_path)) + f" | {original_midi_file_full_path = }"
    u.assert_file_exists(original_midi_file_full_path)

    assert isinstance(midi_img_path, str), "midi_img_path must be a string but is {}".format(
        type(midi_img_path))
    assert midi_img_path.endswith(
        ".png") or midi_img_path.endswith(".jpeg"), midi_img_path

    midi_df = _get_midi_df(original_midi_file_full_path)

    n_row = 128
    n_col = (midi_df.end_col_num.max() -
             midi_df.start_col_num.min() + 1).astype(int)

    # TODO: No need to use torch. Just use numpy
    midi_img_tensor = torch.zeros(n_row, n_col)

    # fill in the image
    for _, a_midi_df_row in midi_df.iterrows():
        img_row_num = n_row - int(a_midi_df_row.pitch)
        img_col_num_start = int(a_midi_df_row.start_col_num)
        img_col_num_end = int(a_midi_df_row.end_col_num)

        # scale value to 0 and 1
        pixel_val = a_midi_df_row.velocity / 127.0
        midi_img_tensor[img_row_num,
                        img_col_num_start:img_col_num_end] = pixel_val

    # convert midi_img_path to full path
    midi_img_path = os.path.abspath(midi_img_path)

    plt.imsave(midi_img_path, arr=midi_img_tensor.numpy(),
               vmin=0., vmax=1., cmap="gray", format='png')
    print(f"\n Generated {midi_img_path} \n")
    return midi_img_path


def resize_image_to_square_shape(full_midi_img_path, side_length=224*1):
    assert isinstance(full_midi_img_path, str)
    u.assert_file_exists(full_midi_img_path)

    assert isinstance(
        side_length, int), f"{side_length = } is not an integer..."
    assert side_length > 0, f"{side_length = } is not a positive integer..."

    # resize the image using transform
    img = plt.imread(full_midi_img_path)
    img = torch.tensor(img)
    img = img.permute(2, 0, 1)
    img = torch.nn.functional.interpolate(img.unsqueeze(
        0), size=side_length, mode="bilinear", align_corners=False)
    img = img.squeeze(0).permute(1, 2, 0).numpy()

    # remove .png or .jpeg from the file name
    full_midi_img_path = full_midi_img_path.replace(".png", "")
    full_midi_img_path = full_midi_img_path.replace(".jpeg", "")

    # save the image
    img = np.ascontiguousarray(img)

    # if a value in img is greater than 1, set it to 1
    img[img > 1.0000000] = 1.0

    new_midi_img_path = full_midi_img_path + f"_resized_to_{side_length}.png"

    plt.imsave(new_midi_img_path, arr=img, vmin=0.,
               vmax=1., cmap="gray", format='png')
    print(f"\n Generated {new_midi_img_path} \n")
    return new_midi_img_path


def _get_midi_df(midi_file_path):
    assert isinstance(
        midi_file_path, str), f"{midi_file_path = } is not a string..."
    u.assert_file_exists(midi_file_path)

    midi_data = pretty_midi.PrettyMIDI(midi_file_path)

    # get grid length ------------------------------------------------------------
    tempo = None
    if midi_data.get_tempo_changes()[1].size == 1:
        tempo = midi_data.get_tempo_changes()[1]
    else:
        raise NotImplementedError(
            f"The midi has two changes multiple tempi. T"
            f"his situation is not assumed! Error in the following midi file: \n {midi_file_path = }"
        )

    grid_width_in_s = (60.0 / tempo / midi_data.resolution).item()

    # make pandas dataframe -------------------------------------------------------
    midi_df = pd.DataFrame({})

    for instrument in midi_data.instruments:
        # flatten all instruments
        for note in instrument.notes:
            start = note.start
            end = note.end
            pitch = note.pitch
            velocity = note.velocity
            duration = end - start

            program_num = instrument.program

            # round to the nearest integer number
            # TODO: I might need to add or subtract 1
            start_col_num = int(np.round(start / grid_width_in_s).item())
            # TODO: I might need to add or subtract 1
            end_col_num = int(np.round(end / grid_width_in_s).item())

            new_row = pd.DataFrame({
                # midi note starts at the beginning of this grid
                "start_col_num": [start_col_num],
                "start_time_in_s": [start],
                # midi note starts at the END of this grid
                "end_col_num": [end_col_num],
                "end_time_in_s": [end],
                "program_num": [program_num],
                "pitch": [pitch],
                "velocity": [velocity],
                "duration_in_s": [duration]
            })

            # append an element to dataframe
            midi_df = pd.concat([midi_df, new_row], ignore_index=True)

    try:
        # Cut the beginning head ----------------------------------------------------------
        min_col_num = midi_df["start_col_num"].min()
        midi_df["start_col_num"] = midi_df["start_col_num"] - min_col_num
        midi_df["end_col_num"] = midi_df["end_col_num"] - min_col_num

        min_start_time_in_s = midi_df["start_time_in_s"].min()
        midi_df["start_time_in_s"] = midi_df["start_time_in_s"] - \
            min_start_time_in_s
        midi_df["end_time_in_s"] = midi_df["end_time_in_s"] - \
            min_start_time_in_s
    except KeyError as e:
        print("\n")
        print(f"Error occured for {midi_file_path} !!!!!!!!!!!")
        print("\n")
        traceback.print_exc()
        sys.exit(1)

    return midi_df


def _format_a_midi_file_name(a_midi_file_full_path):
    current_midi_full_file_path = a_midi_file_full_path

    u.assert_file_exists(current_midi_full_file_path)

    # compute hash value of the file
    id_str = "id=" + u.hash_midi(current_midi_full_file_path)

    new_file_name = current_midi_full_file_path.split(
        "/")[-1].replace("_", "-")
    # replace empty space with '-'
    new_file_name = re.sub(r"\s+", "-", new_file_name)
    new_file_name = new_file_name.replace(".mid", "")
    new_file_name = new_file_name.replace(".midi", "")
    new_file_name = new_file_name.replace(",", "")
    new_file_name = new_file_name.replace(".", "")

    if id_str in current_midi_full_file_path and new_file_name in current_midi_full_file_path:
        return current_midi_full_file_path

    if "id=" in current_midi_full_file_path and id_str not in current_midi_full_file_path:
        new_file_name = new_file_name.split("(", 1)[0]
        new_midi_file_full_path = current_midi_full_file_path.rsplit(
            "/", 1)[0] + "/" + new_file_name + "(" + id_str + ")" + ".mid"
    elif "id=" in current_midi_full_file_path and id_str in current_midi_full_file_path:
        new_midi_file_full_path = current_midi_full_file_path.rsplit(
            "/", 1)[0] + "/" + new_file_name + ".mid"
        print(new_midi_file_full_path)
        sys.exit()
    elif "id=" not in current_midi_full_file_path:  # replace file name
        new_midi_file_full_path = current_midi_full_file_path.rsplit(
            "/", 1)[0] + "/" + new_file_name + "(" + id_str + ")" + ".mid"
    else:
        raise NotImplementedError()

    # rename the file with its hash value
    os.rename(current_midi_full_file_path, new_midi_file_full_path)
    return new_midi_file_full_path


# tester code
if __name__ == '__main__':
    from datetime import datetime

    midi_file_full_path = \
        "sample_midi_files/Chopin-Frédéric-Ballade-No2-Op38-NZfNAVQ6H4o(id=181d18db337f201a6980d47a7594c97838ba3937).mid"
    # convert this to full path
    midi_file_full_path = os.path.abspath(midi_file_full_path)
    midi_file_full_path = _format_a_midi_file_name(midi_file_full_path)

    # get current time with milliseconds
    now = datetime.now()

    img_path = convert_midi_file_to_bw_piano_roll_img_with_pitch_as_row(
        original_midi_file_full_path=midi_file_full_path,
        midi_img_path=params.TMP_MIDI_IMG_FOLDER +
        f"{now.strftime('%Y%m%d%H%M%S%f')}.png",
    )

    img_path = resize_image_to_square_shape(img_path)
