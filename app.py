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

import logging
import pandas as pd
import matplotlib.pyplot as plt

from flask import Flask, request
import librosa
import librosa.display
import numpy as np
from fastdtw import fastdtw
from math import acos
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis
from sklearn.metrics.pairwise import cosine_similarity
import urllib.request
import sys

app = Flask(__name__)
 


def download(download_file_path):


    print(download_file_path)
    param= download_file_path.rsplit("/", 1)

    file_url = param[0]
    file_name = param[1]

    print(param)
    print(file_url)
    print(file_name)


    opener = urllib.request.URLopener()
    opener.addheader('User-Agent', 'whatever')
    filename, headers = opener.retrieve(download_file_path, file_name)

    print(filename)

    # headers = {

    #         "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:47.0) Gecko/20100101 Firefox/47.0"
    #         }


    # urllib.request.urlretrieve(download_file_path ,file_name)


# def monauralize(data):
#     #モノラル化
#     try:
#         if data.shape[1] == 2:
#             res = 0.5 * (data.T[0] + data.T[1])
#     except:
#         res = data
#     return res
# ###
# def cos_sim(vec1, vec2):
#     return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
# ###
# #
def librosa_chroma(file_path="", sr=44100):
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

    # # プロットします
    # plt.figure(figsize=(12,4))
    # librosa.display.specshow(chroma_cens, sr=sr, x_axis='time', y_axis='chroma', vmin=0, vmax=1)
    # plt.title('Chromagram')
    # plt.colorbar()
    # plt.tight_layout()
    # plt.show()
    
    return chroma_cens


def compare(path_correct,path_sample):
  cens_correct = pd.DataFrame(librosa_chroma(file_path=path_correct, sr=44100))
  cens_sample = pd.DataFrame(librosa_chroma(file_path=path_sample, sr=44100))
  #np.savetxt('test_burg.csv', cosine_similarity_scores(cens_correct,cens_sample), delimiter=',')
  return cosine_similarity_scores(cens_correct,cens_sample)


# assume cens_correct and cens_sample are ndarray
# whose shape are both (12, window_num)
def cosine_similarity_scores(cens_correct, cens_sample):
  return np.diag(cosine_similarity(cens_correct.values.T,cens_sample.values.T))

def rythm_deviation_cos_sim(path,cens_correct,cens_sample):
  music_len = path[-1][0]+1
  max_scores = -np.ones((music_len,))
  min_scores = np.ones((music_len,))
  idx_path = 0
  for idx in range(music_len):
    while idx == path[idx_path][0]:
      score = cosine_similarity(np.array([cens_correct[idx]]),np.array([cens_sample[path[idx_path][1]]]))[0][0]
      max_scores[idx] = max(max_scores[idx],score)
      min_scores[idx] = min(min_scores[idx],score)
      idx_path+=1
      if idx_path == len(path):
        break
  return max_scores,min_scores

 # =====
@app.route('/analyze')
def hello():
    print('hello')
    url = request.args.get('url')

    correct_path = 'burg5sec.wav'
    # sample_path = 'burg5sec.wav'
    # sample_path = 'burgFast.wav'


    #ファイルは`app.py`と同一ディレクトリに保存される
    download(url)
    print('download done')

    #URLからファイル名を取得
    param= url.rsplit("/", 1)
    print(param[1])
    sample_path = param[1]


    print(correct_path)
    print('compare with')
    print(sample_path)


    # 単純にコサイン類似度取得
    score = compare(correct_path, sample_path)
    cos_sim = sum(score)
    print(cos_sim)

    
# """##DTW"""

    print('DTW------')
    cens_correct = librosa_chroma(file_path=correct_path, sr=44100)
    cens_sample = librosa_chroma(file_path=sample_path, sr=44100)
    x = cens_correct
    y = cens_sample
    print(cens_correct)
    print(cens_sample)
    #path = dtw.warping_path(x.T,y.T)
    _,path = fastdtw(x.T, y.T, dist=(lambda x,y:acos(min(1,max(-1,cosine_similarity([x],[y])[0][0])))))

    max_scores,min_scores = rythm_deviation_cos_sim(path, x.T, y.T)


    # sum dtw_cos_sim
    dtw_cos_sim = sum(min_scores)
    print(dtw_cos_sim)


    len_correct = len(x.T)
    len_sample = len(y.T)

    print('Done')
    str_cos_sim           = 'コサイン類似度は{:.5f}({:.5f}%) / '.format(cos_sim, cos_sim/len_correct * 100)
    str_dtw_cos_sim = 'DTW後のコサイン類似度は{:.5f}({:.5f}%) / '.format(dtw_cos_sim, dtw_cos_sim / len_correct * 100)

    evaluate = ''
    speed_ratio = len_correct / len_sample
    # 判定基準
    # 1.DTW後のコサイン類似度が90%以上なら合格。そうでなければ不合格
    # 2.合格のうち、演奏の長さに応じて「速い」「遅い」「ただしい」を分岐
    if dtw_cos_sim / len_correct * 100 > 90:
        if speed_ratio > 1.1:
            evaluate = '速く弾きましたね'
        elif speed_ratio < 0.9:
            evaluate = 'ゆっくり弾きましたね'
        else:
            evaluate = 'じょうずに弾きましたね'
    else:
        evaluate = 'もうすこしがんばろう'


    str_result = str_cos_sim + str_dtw_cos_sim + evaluate
    print(str_result)

    return str_result

@app.route('/')
def index():
    return 'INDEX'

@app.errorhandler(500)
def server_error(e):
    logging.exception('An error occurred during a request.')
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500

if __name__ == '__main__':
    # # This is used when running locally. Gunicorn is used to run the
    # # application on Google App Engine. See entrypoint in app.yaml.
    # app.run(host='127.0.0.1', port=8080, debug=True)
    app.run(host='0.0.0.0',port=8080, debug=True)
# [END gae_flex_quickstart]
