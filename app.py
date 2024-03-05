# to run the app locally: streamlit run app.py

from streamlit_option_menu import option_menu
import midi_generator
import midi_image_classifier
import time
from datetime import datetime
import midi_preprocessing
import tempfile

import params

import password_protection

import streamlit as st

st.set_page_config(page_title=params.PAGE_TITLE,
                   page_icon=params.PAGE_ICON,
                   layout='wide')

########################################################################################################################
# ----------------------------------------------Main Page---------------------------------------------------------------
########################################################################################################################

# Sidebar Navigation
with st.sidebar:
    selected_page = option_menu(
        menu_title="Navigation",
        options=["About This Page", "Midi Classifier", "Midi Generator"],
        icons=["house", "book", "book"],
        default_index=0
    )

if selected_page == "About This Page":
    with open("app_general_explanation.md", "r", encoding="utf-8") as f:
        st.markdown(f.read(), unsafe_allow_html=True)

if selected_page == 'Midi Classifier':
    with st.container():
        # st.write("--------------")

        # file drop
        st.write("# Midi Classifier")
        uploaded_file = st.file_uploader(
            "Choose a midi file",
            accept_multiple_files=False,
            type=["mid", "midi"],
        )

        if uploaded_file is not None:
            current_file_details = (
                uploaded_file.name, uploaded_file.size) if uploaded_file else None

            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                # write the uploaded file's contents into the temporary file
                tmp_file.write(uploaded_file.getvalue())
                uploaded_file_path = tmp_file.name

                with st.spinner("Analyzing midi... Converting midi file to image..."):
                    # sleep intentionally to show the message
                    time.sleep(1)

                    try:
                        # get current time with milliseconds
                        now = datetime.now()
                        now = now.strftime("%Y%m%d%H%M%S%f")

                        midi_img_path = midi_preprocessing.convert_midi_file_to_bw_piano_roll_img_with_pitch_as_row(
                            original_midi_file_full_path=uploaded_file_path,
                            midi_img_path=params.TMP_MIDI_IMG_FOLDER + now + ".png",
                        )
                        midi_img_path = midi_preprocessing.resize_image_to_square_shape(midi_img_path)

                        # display image
                        # Create three columns
                        # Adjust the ratio as needed for better centering
                        col1, col2, col3 = st.columns([1, 3, 1])
                        # Display the image in the middle column
                        with col2:
                            st.image(midi_img_path, caption='midi file converted to an image', use_column_width=True)

                    except Exception as e:
                        st.error(
                            "Midi file format is wrong. Try with another midi file.")
                        # raise e

                with st.spinner("Classifying generated midi..."):
                    time.sleep(2)
                    try:
                        classification_result_int = midi_image_classifier.classify_midi_image(
                            midi_img_path=midi_img_path
                        )
                        classification_result_str = params.CLASSIFICATION_CLASS_LABELS[classification_result_int]

                        # display the explanation of the result
                        st.write("#### Classification Result")
                        if classification_result_int == 1:
                            st.success(classification_result_str)
                        elif classification_result_int == 0:
                            st.error(classification_result_str)
                    except:
                        st.error("Classification somehow did not work. Sorry.")
        else:
            pass

if selected_page == 'Midi Generator':
    with st.container():
        # st.write("--------------")
        st.write("# Midi Generator")

        new_midi_img_path = None
        new_midi_path = None
        if st.button("Generate Midi"):
            with st.spinner("Generating Midi..."):
                time.sleep(2)
                now = datetime.now()
                new_midi_img_path = params.TMP_MIDI_IMG_FOLDER + \
                    f"{now.strftime('%Y%m%d%H%M%S%f')}.png"
                new_midi_path = params.TMP_MIDI_IMG_FOLDER + \
                    f"{now.strftime('%Y%m%d%H%M%S%f')}.mid"

                midi_generator.generate_midi_img(
                    generated_img_path=new_midi_img_path
                )
                midi_generator.convert_greyscale_img_to_midi_and_save(
                    img_path=new_midi_img_path,
                    midi_path=new_midi_path
                )

                # display image
                # Create three columns
                # Adjust the ratio as needed for better centering
                col1, col2, col3 = st.columns([1, 3, 1])
                # Display the image in the middle column
                with col2:
                    st.image(
                        new_midi_img_path, caption='generated midi image', use_column_width=True)

                st.success("Midi file generated successfully.")
