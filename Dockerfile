FROM python:alpine3.7

ENV PYTHONPATH=/app

WORKDIR /app

# Build system dependencies first for layer caching
COPY Pip* ./
RUN apk --no-cache add git curl openssh-client gcc g++ linux-headers && \
  pip install --no-cache-dir pipenv && \
  pipenv install --deploy --system

# Install app code
COPY src /app/src
COPY conf /app/conf
COPY start.sh .

EXPOSE 8080
CMD ["/app/start.sh"]
