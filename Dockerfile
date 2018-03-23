FROM python:alpine3.7

ENV PYTHONPATH=/app

WORKDIR /app

# Build system dependencies first for layer caching
COPY Pip* ./
RUN echo "http://dl-8.alpinelinux.org/alpine/edge/community" >> /etc/apk/repositories && \
	apk --no-cache --update-cache add gcc gfortran python python-dev py-pip build-base wget freetype-dev libpng-dev openblas-dev && \
	ln -s /usr/include/locale.h /usr/include/xlocale.h && \
	pip install --no-cache-dir -U numpy scipy pandas

# Install app dependencies
RUN apk add --no-cache --virtual=.buildeps git curl openssh-client g++ linux-headers musl-dev && \
	pip install --no-cache-dir pipenv && \
    pipenv install --deploy --system 

# Install app code
COPY src /app/src
COPY conf /app/conf
COPY start.sh .

EXPOSE 8080
CMD ["/app/start.sh"]
