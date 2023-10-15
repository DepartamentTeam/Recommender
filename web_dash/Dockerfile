FROM python:3.10

RUN mkdir -p /app/
WORKDIR /app/

COPY ./ /app/

RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8050

CMD ["python", "app.py"]