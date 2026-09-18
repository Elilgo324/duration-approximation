FROM python@sha256:e9d4cbc0c16887e69c08b6451fd556e644fb96814ffcc5da840cd4cb78cfd3e1
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt
COPY duration_approximation/ duration_approximation/
COPY tests/ tests/
RUN PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests
ENTRYPOINT ["python", "-m", "duration_approximation.experiment"]
CMD ["--mode", "smoke", "--output", "/results/smoke"]
