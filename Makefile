SERVICE := marketing_opp_scoring
BASE_IMAGE := gcr.io/v1-dev-main/datascience-base-tf2
TAG := $(shell date +%Y%m%d-%H%M)
INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include /usr/include/hdf5/serial/
LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu/hdf5/serial/


ifeq ($(ENV),)
$(error ENV needs to be defined. e.g.: make deploy ENV=dev)
endif

build:
	gcloud auth configure-docker --account=$$(gcloud auth list | sed -n 's/^\*?[[:space:]]+\(.*volusion.com\).*$$/\1/p')
	docker pull $(BASE_IMAGE)
	docker build -t $(SERVICE) . 

build-base:
	gcloud auth configure-docker --account=$$(gcloud auth list | sed -n 's/^\*?[[:space:]]+\(.*volusion.com\).*$$/\1/p')
	docker build -t $(BASE_IMAGE) -f Dockerfile.base .
	docker push $(BASE_IMAGE)

run: build 
	docker run -it -p 8080:8080 $(SERVICE)

run_local: build
	docker run -it -p 8080:8080 -v "$(HOME)/.config/gcloud":/root/.config/gcloud $(SERVICE)

deploy: build
	docker tag $(SERVICE) gcr.io/v1-dev-main/$(SERVICE):$(TAG)
	docker push gcr.io/v1-dev-main/$(SERVICE):$(TAG)
	gcloud app deploy app.$(ENV).yaml cron.yaml --project=v1-$(ENV)-main --version=$(shell date +%Y%m%d-%H%M) --image-url=gcr.io/v1-dev-main/$(SERVICE):$(TAG) --verbosity=debug

logs:
	gcloud app logs tail \
		--service=$(SERVICE) \
		--version=`gcloud app instances list --project=v1-$(ENV)-main | grep $(SERVICE) | awk '{print $$2}'` \
		--project=v1-$(ENV)-main \
		| grep -v '/health-check'

