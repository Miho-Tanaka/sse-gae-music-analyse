FROM python:3
ADD . /root

RUN pip3 install --upgrade pip


RUN pip3 install --no-cache-dir pandas --no-build-isolation

RUN pip3 install flask 
RUN pip install google-cloud-storage
RUN pip install sklearn
RUN pip install librosa
RUN pip install matplotlib
RUN git clone https://github.com/Miho-Tanaka/dtaidistance_sse
RUN pip uninstall dtaidistance && cd dtaidistance_sse && python3 ./setup.py install
RUN pip install dtw
RUN pip install fastdtw
#RUN pip install tqdm

RUN apt-get -y update 
RUN apt-get install -y libsndfile1

# RUN pip install pysoundfile

# RUN python -m pip list

# EXPOSE 8080

WORKDIR /root

CMD python app.py

# how to run with dockerfile
## docker build -t sse/gae-music .
## docker run -p 8080:8080 -it sse/gae-music
