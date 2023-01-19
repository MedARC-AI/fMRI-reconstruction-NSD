NAME ?= laion-fmri
REG ?= jimgoo6/laion-fmri

build:
	docker build -t ${NAME}:latest -f Dockerfile .

push:
	docker tag ${NAME}:latest ${REG}:latest
	docker push ${REG}:latest
