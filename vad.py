#write API to check vad of audio file
#input: path to audio file
#output: True if audio file has voice, False if audio file has no voice
#example: print(check_vad(path))
import numpy as np
from flask import Flask
import werkzeug
werkzeug.cached_property = werkzeug.utils.cached_property
from flask_restx import Api, Resource
import os
from pydub import AudioSegment

os.environ["no_proxy"] = "localhost, 127.0.0.1, ::1"
app = Flask(__name__)
api = Api(app, version='1.0', title='Check_vad', description='API for voice activity detection')

# VAD_SR = 16000
# model, utils = torch.hub.load(
#     repo_or_dir="snakers4/silero-vad", model="silero_vad", onnx=False
# )
# (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

# #check vad
# upload_parser_vad = api.parser()
# upload_parser_vad.add_argument('file_path', type=str, required=True, 
#                            help='Path to audio file', 
#                            default='my_audio.wav')
# upload_parser_vad.add_argument('vad_threshold', type=np.float32, required=True, 
#                            help='Threshold for vad', 
#                            default=0.4)
# @api.route('/check_vad') 
# class check_vad(Resource):
#     @api.expect(upload_parser_vad)
#     def post(self):
#         try:
#             args = upload_parser_vad.parse_args()
#             file_path = args['file_path']
#             vad_threshold = args['vad_threshold']
#             audio, _ = librosa.load(file_path, sr=VAD_SR)
#             wav = torch.tensor(audio)
#             t = get_speech_timestamps(wav, model, sampling_rate=VAD_SR, threshold=vad_threshold)
#             if len(t) ==0:
#                 return {"message": "No voice"}, 200
#             else:
#                 return {"message": "Has voice"}, 200
#         except Exception as e:
#                 return {"message": str(e)}, 500



#strim silence
upload_parser_strim = api.parser()
upload_parser_strim.add_argument('input_file', type=str, required=True, 
                           help='Path to input audio file', 
                           default='input_audio.wav')
# upload_parser_strim.add_argument('output_file', type=str, required=True,
#                             help='Path to output audio file', 
#                             default='output_audio.wav')
upload_parser_strim.add_argument('silence_threshold', type=np.float32, required=True,
                            help='Threshold for silence', 
                            default=-40)
upload_parser_strim.add_argument('chunk_size', type=np.int16, required=True,
                            help='Chunk size for silence', 
                            default=5)

def milliseconds_until_sound(sound, silence_threshold_in_decibels=-40.0, chunk_size=5):
    trim_ms = 0  # ms
    sound_strim = sound[0:0]
    assert chunk_size > 0  # to avoid infinite loop
    while  trim_ms < len(sound):
        if sound[trim_ms:trim_ms+chunk_size].dBFS > silence_threshold_in_decibels:
            sound_strim+=sound[trim_ms:trim_ms+chunk_size]
        trim_ms += chunk_size
    return sound_strim

@api.route('/strim_silence')
class strim_silence(Resource):
    @api.expect(upload_parser_strim)
    def post(self):
        try:
            args = upload_parser_strim.parse_args()
            input_file = args['input_file']
            # output_file = args['output_file']
            silence_threshold = args['silence_threshold']
            chunk_size = args['chunk_size']
            audio = AudioSegment.from_file(input_file)
            # non_silent_audio = audio.strip_silence(silence_thresh=silence_threshold)
            audio_strim = milliseconds_until_sound(audio, silence_threshold_in_decibels=silence_threshold, chunk_size=chunk_size)
            # print(len(audio_strim), len(non_silent_audio))
            # audio_strim.export("_" + output_file, format="wav")
            # non_silent_audio.export(output_file, format="wav")
            return {"non_silent_seconds": len(audio_strim)/1000}, 200
        except Exception as e:
                return {"message": str(e)}, 500
        

if __name__ == "__main__":
    app.run(debug=True)


