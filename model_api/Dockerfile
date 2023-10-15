FROM python:3.10

RUN mkdir -p /app/
WORKDIR /app/

COPY ./ /app/

RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8000

CMD ["python", "model_api.py"]