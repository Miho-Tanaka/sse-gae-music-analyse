# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# [START gae_flex_quickstart]
import logging
import pandas as pd
import matplotlib.pyplot as plt

from flask import Flask
import librosa
import librosa.display
import numpy as np
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis


app = Flask(__name__)


@app.route('/')
def hello():
    print('hello')

def monauralize(data):
    #モノラル化
    try:
        if data.shape[1] == 2:
            res = 0.5 * (data.T[0] + data.T[1])
    except:
        res = data
    return res
###
def cos_sim(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
###

#
def librosa_chroma(file_path="https://storage.cloud.google.com/smartse-music/burg5sec.wav", sr=44100):
    import librosa

    # 読み込み(sr:サンプリングレート)
    y, sr = librosa.load(file_path, sr=sr)

    # 楽音成分とパーカッシブ成分に分けます
    y_harmonic, y_percussive = librosa.effects.hpss(y)

    #### CENSからクロマグラムを計算
    chroma_cens = librosa.feature.chroma_cens(y=y_harmonic, sr=sr)
    
    # パワースペクトログラムからクロマグラムを計算
    chroma_stft = librosa.feature.chroma_stft(y=y_harmonic, sr=sr, n_chroma=12, n_fft=4096)

    # Constant-Qからクロマグラムを計算
    chroma_cq = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)    

    # プロットします
    plt.figure(figsize=(12,4))
    librosa.display.specshow(chroma_cens, sr=sr, x_axis='time', y_axis='chroma', vmin=0, vmax=1)
    plt.title('Chromagram')
    plt.colorbar()
    plt.tight_layout()
    plt.show()
    
    return chroma_cens

from sklearn.metrics.pairwise import cosine_similarity
# assume cens_correct and cens_sample are ndarray
# whose shape are both (12, window_num)
#print(cosine_similarity([[1,1,1]],[[1,1,1]])[0])
def cosine_similarity_scores(cens_correct, cens_sample):
  return np.diag(cosine_similarity(cens_correct.values.T,cens_sample.values.T))

# correct_path = 'https://storage.cloud.google.com/smartse-music/burg5sec.wav'
correct_path = 'burgFast.wav'

# sample_path = 'https://storage.cloud.google.com/smartse-music/' + filename
sample_path = 'burgFast.wav'
def compare(path_correct,path_sample):
  cens_correct = pd.DataFrame(librosa_chroma(file_path=path_correct, sr=44100))
  cens_sample = pd.DataFrame(librosa_chroma(file_path=path_sample, sr=44100))
  #np.savetxt('test_burg.csv', cosine_similarity_scores(cens_correct,cens_sample), delimiter=',')
  return cosine_similarity_scores(cens_correct,cens_sample)

score = compare(correct_path,sample_path)
plt.bar(range(len(score)),score)

"""##DTW"""

# Commented out IPython magic to ensure Python compatibility.
# git clone https://github.com/Miho-Tanaka/dtaidistance_sse
# %cd dtaidistance_sse

# yes | pip uninstall dtaidistance
# python3 setup.py install

# path_correct = 'https://storage.cloud.google.com/smartse-music/burg5sec.wav'
path_correct = 'burgFast.wav'

# path_sample = 'https://storage.cloud.google.com/smartse-music/burg5sec.wav'
path_sample = 'burgFast.wav'

#from collections import OrderedDict
import librosa
import librosa.display
def librosa_chroma(file_path="", sr=44100):
    import librosa

    # 読み込み(sr:サンプリングレート)
    y, sr = librosa.load(file_path, sr=sr)

    # 楽音成分とパーカッシブ成分に分けます
    y_harmonic, y_percussive = librosa.effects.hpss(y)

    # クロマグラムを計算します
    C = librosa.feature.chroma_cens(y=y_harmonic, sr=sr)
    return C

cens_correct = librosa_chroma(file_path=path_correct, sr=44100)
cens_sample = librosa_chroma(file_path=path_sample, sr=44100)
x = cens_correct
y = cens_sample

# from dtaidistance import dtw
# from dtaidistance import dtw_visualisation as dtwvis
import numpy as np
path = dtw.warping_path(x.T,y.T)
dtwvis.plot_warping([z[0] for z in x.T], [z[0] for z in y.T], path)

dtwvis.plot_warping([z[2] for z in x.T], [z[2] for z in y.T], path)

plt.plot(np.array(path).T[0], np.array(path).T[1], 'k')
plt.show()
    # return 'Done!!'

@app.errorhandler(500)
def server_error(e):
    logging.exception('An error occurred during a request.')
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500

if __name__ == '__main__':
    # This is used when running locally. Gunicorn is used to run the
    # application on Google App Engine. See entrypoint in app.yaml.
    app.run(host='127.0.0.1', port=8080, debug=True)
# [END gae_flex_quickstart]
