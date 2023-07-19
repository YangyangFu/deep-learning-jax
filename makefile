IMA_NAME = tf
HOST = yangyangfu

build:
	docker build --no-cache --rm -t ${HOST}/${IMA_NAME} .