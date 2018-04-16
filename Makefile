SERVICE := marketing_lead_scoring
BASE_IMAGE := gcr.io/v1-dev-main/datascience-base:python3-alpine

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

deploy:
	gcloud app deploy app.$(ENV).yaml cron.yaml --project=v1-$(ENV)-main --version=$(shell date +%Y%m%d-%H%M)

logs:
	gcloud app logs tail \
		--service=$(SERVICE) \
		--version=`gcloud app instances list --project=v1-$(ENV)-main | grep $(SERVICE) | awk '{print $$2}'` \
		--project=v1-$(ENV)-main \
		| grep -v '/health-check'
