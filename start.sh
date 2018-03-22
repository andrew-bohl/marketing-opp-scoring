#!/bin/sh

function KmsDecrypt() {
  local KEYRING="volusion-bi"
  local KEY="marketing_lead_scoring"

  local CIPHERTEXT="${1}"
  local PLAINTEXT=$(echo ${CIPHERTEXT} | \
    gcloud kms decrypt \
      --ciphertext-file=- \
      --plaintext-file=- \
      --keyring ${KEYRING} \
      --key ${KEY} \
      --location global
    )

  echo "${PLAINTEXT}"
}

# Get configs
ENV=${ENV:-"dev"}
cd conf
source ./base.env
source ./${ENV}.env
cd -

echo "Env variables:"
env | sort

# Start flask web application
twistd -n web --wsgi src.main.app --port tcp:8080
