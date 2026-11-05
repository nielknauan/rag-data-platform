.PHONY: install test run stream incremental eval ask maintain all docker-build docker-run clean

export PYTHONPATH := $(CURDIR)/src
export LAKEHOUSE_ROOT ?= $(CURDIR)/lakehouse
Q ?= Quais direitos o titular dos dados tem?

install:
	pip install -r requirements.txt

test:            ## testes unitários (embedder determinístico, sem baixar modelo)
	EMBEDDER=hash pytest -q

run:             ## batch: landing -> bronze -> silver -> chunks -> embeddings -> índice
	python -m rag_lakehouse.pipeline run

stream:          ## streaming contínuo: solte arquivos em lakehouse/landing/docs
	python -m rag_lakehouse.pipeline stream

incremental:     ## simula chegada de documento novo e reprocessa
	cp data/sample/incoming/*.md $(LAKEHOUSE_ROOT)/landing/docs/
	python -m rag_lakehouse.pipeline run

eval:
	python -m rag_lakehouse.pipeline eval --min-hit-rate 0.8

ask:
	python -m rag_lakehouse.pipeline ask "$(Q)"

maintain:        ## OPTIMIZE + VACUUM
	python -m rag_lakehouse.pipeline maintain

all: test run eval

docker-build:
	docker build -t rag-lakehouse .

docker-run:
	docker run --rm -v $(CURDIR)/lakehouse:/app/lakehouse rag-lakehouse

clean:
	rm -rf lakehouse spark-warehouse metastore_db
