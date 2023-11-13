FROM python:3.11-slim-bullseye

RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install "git+https://github.com/MO-RISE/keelson.git@0.1.0-pre.12#subdirectory=brefv/python"

RUN pip3 install --no-cache-dir -r requirements.txt

CMD [ python3 ./src/main.py -o 10.10.7.63 -r test -e ted -i os2 -s 0  -f fid ]
