FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY engine.py .
COPY slack_notifier.py .
COPY cost_model.py .

EXPOSE 8001

CMD ["python", "-m", "uvicorn", "engine:app", "--host", "0.0.0.0", "--port", "8001"]
