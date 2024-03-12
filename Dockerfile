FROM python:3.8
WORKDIR /app
COPY . /app
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libsm6 libxext6 libxrender-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    opencv-python-headless \
    numpy \
    pygame \
    tensorflow \
    flask

EXPOSE 5005

CMD ["python", "app.py"]
 
