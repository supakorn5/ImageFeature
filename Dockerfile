FROM python:3.11.4

# 
WORKDIR /ImageFeature

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 -y

# 
COPY ./requirements.txt /ImageFeature/requirements.txt

# 
RUN pip install --no-cache-dir --upgrade -r /ImageFeature/requirements.txt

# 
COPY ./app /ImageFeature/app


ENV PYTHONPATH "${PYTHONPATH}:/ImageFeature"

# a
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]