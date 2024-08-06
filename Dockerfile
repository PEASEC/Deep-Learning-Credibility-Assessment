FROM python:3.7.9
WORKDIR /learner

COPY requirements.txt /learner/requirements.txt
RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt

COPY src/ /learner/src/
COPY resources/ /learner/resources/
COPY prebuild/ /learner/prebuild/
COPY rest.py /learner/rest.py

ENTRYPOINT python ./rest.py