FROM python:3.11
WORKDIR /app
COPY . /app
RUN pip install flask
RUN pip install tensorflow
RUN pip install keras
RUN pip install numpy
RUN pip install os-sys
RUN pip install socket
EXPOSE 8080
CMD ["python3", "main.py"]