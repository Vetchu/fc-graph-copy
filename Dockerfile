FROM python:3.8-buster
RUN pip3 install --upgrade pip

RUN apt-get update && apt-get install -y \
    supervisor nginx

COPY server_config/supervisord.conf /supervisord.conf
COPY server_config/nginx /etc/nginx/sites-available/default
COPY server_config/docker-entrypoint.sh /entrypoint.sh

# RUN pip3 install torch==1.9.0+cpu torchvision==0.10.0+cpu torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
COPY requirements.txt /app/requirements.txt
RUN pip3 install -r ./app/requirements.txt

COPY . /app

EXPOSE 9000 9001

ENTRYPOINT ["sh", "/entrypoint.sh"]
