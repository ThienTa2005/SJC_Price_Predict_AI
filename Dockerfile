FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PORT=8501 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import os,urllib.request; urllib.request.urlopen('http://localhost:'+os.environ.get('PORT','8501')+'/_stcore/health')"
CMD ["python", "start.py"]

