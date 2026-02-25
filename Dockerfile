FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y default-jre wget

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app/ ./app/

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
