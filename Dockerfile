FROM gcr.io/v1-dev-main/datascience-base-tf2

ENV PYTHONPATH=/app
WORKDIR /app

# Install app code
COPY src /app/src
COPY conf /app/conf
COPY start.sh .

EXPOSE 8080
RUN ["chmod", "a+rX", "/app/start.sh"]
CMD ["/app/start.sh"]
