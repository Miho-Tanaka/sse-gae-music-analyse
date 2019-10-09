FROM python:3
ADD . /root

RUN pip3 install --upgrade pip
RUN pip install -r requirements.txt

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
