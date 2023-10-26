#write API to check vad of audio file
#input: path to audio file
#output: True if audio file has voice, False if audio file has no voice
#example: print(check_vad(path))
import librosa
import torch
import numpy as np
from flask import Flask
import werkzeug
werkzeug.cached_property = werkzeug.utils.cached_property
from flask_restx import Api, Resource
import os
import tracemalloc
from pydub import AudioSegment

os.environ["no_proxy"] = "localhost, 127.0.0.1, ::1"
tracemalloc.start()
app = Flask(__name__)
api = Api(app, version='1.0', title='Check_vad', description='API for voice activity detection')

VAD_SR = 16000
model, utils = torch.hub.load(
    repo_or_dir="snakers4/silero-vad", model="silero_vad", onnx=False
)
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

#check vad
upload_parser_vad = api.parser()
upload_parser_vad.add_argument('file_path', type=str, required=True, 
                           help='Path to audio file', 
                           default='my_audio.wav')
upload_parser_vad.add_argument('vad_threshold', type=np.float32, required=True, 
                           help='Threshold for vad', 
                           default=0.4)
@api.route('/check_vad') 
class check_vad(Resource):
    @api.expect(upload_parser_vad)
    def post(self):
        try:
            args = upload_parser_vad.parse_args()
            file_path = args['file_path']
            vad_threshold = args['vad_threshold']
            audio, _ = librosa.load(file_path, sr=VAD_SR)
            wav = torch.tensor(audio)
            t = get_speech_timestamps(wav, model, sampling_rate=VAD_SR, threshold=vad_threshold)
            if len(t) ==0:
                return {"message": "No voice"}, 200
            else:
                return {"message": "Has voice"}, 200
        except Exception as e:
                return {"message": str(e)}, 500



#strip silence
upload_parser_strip = api.parser()
upload_parser_strip.add_argument('input_file', type=str, required=True, 
                           help='Path to input audio file', 
                           default='input_audio.wav')
upload_parser_strip.add_argument('output_file', type=str, required=True,
                            help='Path to output audio file', 
                            default='output_audio.wav')
upload_parser_strip.add_argument('silence_threshold', type=np.float32, required=True,
                            help='Threshold for silence', 
                            default=-50)
@api.route('/strip_silence')
class strip_silence(Resource):
    @api.expect(upload_parser_strip)
    def post(self):
        try:
            args = upload_parser_strip.parse_args()
            input_file = args['input_file']
            output_file = args['output_file']
            silence_threshold = args['silence_threshold']
            audio = AudioSegment.from_file(input_file)
            non_silent_audio = audio.strip_silence(silence_thresh=silence_threshold)
            non_silent_audio.export(output_file, format="wav")
            return {"message": "Strip silence successfully"}, 200
        except Exception as e:
                return {"message": str(e)}, 500
        

if __name__ == "__main__":
    
    app.run(debug=True)
    Current,_ = tracemalloc.get_traced_memory()
    print(Current/1024**2)
    tracemalloc.stop()


