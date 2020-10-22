import json
# pylint: disable=redefined-outer-name, unused-argument
import os
import string
import numpy as np
import torch

from mozilla_voice_tts.tts.utils.generic_utils import setup_model
from mozilla_voice_tts.tts.utils.synthesis import synthesis
from mozilla_voice_tts.tts.utils.text.symbols import make_symbols, phonemes, symbols
from mozilla_voice_tts.utils.audio import AudioProcessor
from mozilla_voice_tts.utils.io import load_config
from mozilla_voice_tts.vocoder.utils.generic_utils import setup_generator


def tts(model, vocoder_model, text, CONFIG, use_cuda, ap, use_gl, speaker_fileid, speaker_embedding=None, gst_style=None):
    t_1 = time.time()
    waveform, _, _, mel_postnet_spec, _, _ = synthesis(model, text, CONFIG, use_cuda, ap, speaker_fileid, gst_style, False, CONFIG.enable_eos_bos_chars, use_gl, speaker_embedding=speaker_embedding)
    if CONFIG.model == "Tacotron" and not use_gl:
        mel_postnet_spec = ap.out_linear_to_mel(mel_postnet_spec.T).T
    if not use_gl:
        waveform = vocoder_model.inference(torch.FloatTensor(mel_postnet_spec.T).unsqueeze(0))
    if use_cuda and not use_gl:
        waveform = waveform.cpu()
    if not use_gl:
        waveform = waveform.numpy()
    waveform = waveform.squeeze()
    rtf = (time.time() - t_1) / (len(waveform) / ap.sample_rate)
    tps = (time.time() - t_1) / len(waveform)
    print(" > Run-time: {}".format(time.time() - t_1))
    print(" > Real-time factor: {}".format(rtf))
    print(" > Time per step: {}".format(tps))
    return waveform

TEXT = ''
OUT_PATH = 'tests-audios/'
# create output path
os.makedirs(OUT_PATH, exist_ok=True)

SPEAKER_FILEID = None # if None use the first embedding from speakers.json

# model vars
MODEL_PATH = r'TTS-checkpoint\best_model.pth.tar'
CONFIG_PATH = r'TTS-checkpoint\config.json'
SPEAKER_JSON = r'TTS-checkpoint\speakers.json'

mVOCODER_PATH = r'multiband-melgan\checkpoint_1450000.pth.tar'
mVOCODER_CONFIG_PATH = r'multiband-melgan\config.json'

USE_CUDA = False

# load the config
C = load_config(CONFIG_PATH)
C.forward_attn_mask = True

# load the audio processor
ap = AudioProcessor(**C.audio)

# if the vocabulary was passed, replace the default
if 'characters' in C.keys():
    symbols, phonemes = make_symbols(**C.characters)

speaker_embedding = None
speaker_embedding_dim = None
num_speakers = 0
# load speakers
if SPEAKER_JSON != '':
    speaker_mapping = json.load(open(SPEAKER_JSON, 'r'))
    num_speakers = len(speaker_mapping)
    if C.use_external_speaker_embedding_file:
        if SPEAKER_FILEID is not None:
            speaker_embedding = speaker_mapping[SPEAKER_FILEID]['embedding']
        else: # if speaker_fileid is not specificated use the first sample in speakers.json
            choise_speaker = list(speaker_mapping.keys())[0]
            print(" Speaker: ",choise_speaker.split('_')[0],'was chosen automatically', "(this speaker seen in training)")
            speaker_embedding = speaker_mapping[choise_speaker]['embedding']
        speaker_embedding_dim = len(speaker_embedding)

# load the model
num_chars = len(phonemes) if C.use_phonemes else len(symbols)
model = setup_model(num_chars, num_speakers, C, speaker_embedding_dim)
cp = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
model.load_state_dict(cp['model'])
model.eval()

if USE_CUDA:
    model.cuda()

model.decoder.set_r(cp['r'])

def load_vocoder(path, config):
    # load vocoder model
    if path != "":
        VC = load_config(config)
        vocoder_model = setup_generator(VC)
        vocoder_model.load_state_dict(torch.load(path, map_location="cpu")["model"])
        vocoder_model.remove_weight_norm()
        vocoder_model.eval()
    else:
        vocoder_model = None
        VC = None

    # synthesize voice
    use_griffin_lim = path== ""
    return vocoder_model, use_griffin_lim

if not C.use_external_speaker_embedding_file:
    if SPEAKER_FILEID.isdigit():
        SPEAKER_FILEID = int(SPEAKER_FILEID)
    else:
        SPEAKER_FILEID = None
else:
    SPEAKER_FILEID = None


def synth_own_sample(TEXT, vocoder_model, use_griffin_lim, speaker_embedding, file_add="", noisy=None):

    print(" > Text: {}".format(TEXT))
    wav = tts(model, vocoder_model, TEXT, C, USE_CUDA, ap, use_griffin_lim, SPEAKER_FILEID,
              speaker_embedding=speaker_embedding)
    wav = nr.reduce_noise(audio_clip=wav, noise_clip=noisy, verbose=False, prop_decrease=1)

    # save the results
    file_name = TEXT.replace(" ", "_")
    file_name = file_name.translate(
        str.maketrans('', '', string.punctuation.replace('_', ''))) + f'_{file_add}.wav'
    out_path = os.path.join(OUT_PATH, file_name)
    print(" > Saving output to {}".format(out_path))
    ap.save_wav(wav, out_path)
    return out_path

def load_speaker_embedding(file_list):
    SE_MODEL_RUN_PATH = "GE2E-SpeakerEncoder"
    SE_MODEL_PATH = os.path.join(SE_MODEL_RUN_PATH, "best_model.pth.tar")
    SE_CONFIG_PATH = os.path.join(SE_MODEL_RUN_PATH, "config.json")
    USE_CUDA = False

    from mozilla_voice_tts.utils.audio import AudioProcessor
    from mozilla_voice_tts.speaker_encoder.model import SpeakerEncoder
    se_config = load_config(SE_CONFIG_PATH)
    se_ap = AudioProcessor(**se_config['audio'])

    se_model = SpeakerEncoder(**se_config.model)
    se_model.load_state_dict(torch.load(SE_MODEL_PATH, map_location=torch.device('cpu'))['model'])
    se_model.eval()
    if USE_CUDA:
        se_model.cuda()

    # select one or more wav files
    #file_list = [r"C:\Users\annak\OneDrive - Loctimize\Dokumente\Studium\Bachelorarbeit\tts\clean_231_v1.wav"]

    # extract embedding from wav files
    speaker_embeddings = []
    for name in file_list:
        if '.wav' in name:
            mel_spec = se_ap.melspectrogram(se_ap.load_wav(name, sr=se_ap.sample_rate)).T
            mel_spec = torch.FloatTensor(mel_spec[None, :, :])
            if USE_CUDA:
                mel_spec = mel_spec.cuda()
            embedd = se_model.compute_embedding(mel_spec).cpu().detach().numpy().reshape(-1)
            speaker_embeddings.append(embedd)
        else:
            print("You need upload Wav files, others files is not supported !!")

    # takes the average of the embedings samples of the announcers
    speaker_embedding = np.mean(np.array(speaker_embeddings), axis=0).tolist()
    return speaker_embedding


import noisereduce as nr
import librosa, time
import playsound
noisy, raten = librosa.load("noise mulmel.wav")

file_list231 = [os.path.join(r"modified_speaker", file) for file in os.listdir(r"modified_speaker")]

speaker_embed231 = load_speaker_embedding(file_list231)

mm_vc, ugl2 = load_vocoder(mVOCODER_PATH, mVOCODER_CONFIG_PATH)

while True:
    text = input("Text input: ")
    if text == "quit":
        break
    sentences = text.split(".")
    for sentence in sentences:
        sentence = sentence.strip()
        out = synth_own_sample(sentence, mm_vc, ugl2, speaker_embed231, file_add="", noisy=noisy)
        print(out)
        playsound.playsound(out)
    text = "quit"
    break


