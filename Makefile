SERVICE := marketing-opp-scoring
BASE_IMAGE := gcr.io/v1-dev-main/datascience-base-tf2
TAG := $(shell date +%Y%m%d-%H%M)
ACCOUNT :=$(shell gcloud config list account --format "value(core.account)")

ifeq ($(ENV),)
$(error ENV needs to be defined. e.g.: make deploy ENV=dev)
endif

auth:
	gcloud auth configure-docker --account=$(ACCOUNT)

build: auth
	docker pull $(BASE_IMAGE)
	docker build -t $(SERVICE) . 

build-base: auth
	docker build -t $(BASE_IMAGE) -f Dockerfile.base .
	docker push $(BASE_IMAGE)

run: build 
	docker run -it -p 8080:8080 $(SERVICE)

run_local: build
	docker run -it -p 8080:8080 -v "$(HOME)/.config/gcloud":/root/.config/gcloud $(SERVICE)

deploy: build
	docker tag $(SERVICE) gcr.io/v1-dev-main/$(SERVICE):$(TAG)
	docker push gcr.io/v1-dev-main/$(SERVICE):$(TAG)
	gcloud app deploy app.$(ENV).yaml cron.yaml \
		--project=v1-$(ENV)-main \
		--version=$(TAG) \
		--image-url=gcr.io/v1-dev-main/$(SERVICE):$(TAG) \
		--verbosity=debug

logs:
	gcloud app logs tail \
		--service=$(SERVICE) \
		--version=`gcloud app instances list --project=v1-$(ENV)-main | grep $(SERVICE) | awk '{print $$2}'` \
		--project=v1-$(ENV)-main \
		| grep -v '/health-check'


