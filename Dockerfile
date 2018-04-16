FROM gcr.io/v1-dev-main/datascience-base:python3-alpine

ENV PYTHONPATH=/app
WORKDIR /app

# Python dependencies
COPY Pip* ./
RUN pipenv install --deploy --system && \
  apk del .buildeps

# Install app code
COPY src /app/src
COPY conf /app/conf
COPY start.sh .

EXPOSE 8080
CMD ["/app/start.sh"]
