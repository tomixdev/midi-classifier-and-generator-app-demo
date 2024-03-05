PAGE_TITLE = 'long midi classifier demo'
PAGE_ICON = ":musical_note:"

TMP_MIDI_IMG_FOLDER = "./tmp_midi_img_files/"

NUM_CLASSES_FOR_MIDI_IMG_CLASSIFIER = 2
PATH_TO_CLASSIFIER_MODEL = "./pretrained_models/classification_20230713_030922_model_trained_on_cuda_device.pth"
CLASSIFICATION_CLASS_LABELS = {
    0: "Incomplete Music Segment",
    1: "Complete Music Segment"
}
NUM_CLASSES_FOR_MIDI_GENERATOR = 2

PATH_TO_GENERATOR_MODEL = "./pretrained_models/modelG_90_epochs_20230915_065222.pth"
