FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN python - <<'PY'
from pathlib import Path
raw = Path('/app/requirements.txt').read_bytes()
for enc in ('utf-16', 'utf-8'):
    try:
        txt = raw.decode(enc)
        break
    except UnicodeDecodeError:
        txt = None
if txt is None:
    raise RuntimeError('Unable to decode requirements.txt')
Path('/app/requirements-ci.txt').write_text(txt, encoding='utf-8')
PY

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements-ci.txt

COPY . /app

EXPOSE 8000

CMD ["uvicorn", "Deploy.api:app", "--host", "0.0.0.0", "--port", "8000"]
