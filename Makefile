.PHONY: clean
clean:
	find . \( -name __pycache__ \
		-o -name "*.pyc" \
		-o -name .pytest_cache \
		-o -path "./build" \
		-o -path "./dist" \
		-o -path "./sgmrfmix.egg-info" \
		! -path ".venv/*" \
	\) -exec rm -rf {} +


.PHONY: install-dev
install-dev:
	pip install -e .[dev]

.PHONY: build
build:
	mkdir -p build && \
	cd build && \
	cmake -DPYTHON_EXECUTABLE:FILEPATH=`which python` .. && \
	make && \
	cp *.so ../

.PHONY: test
test:
	pytest test

.PHONY: docker-run-bash
docker-run-bash:
	docker run --rm -v `pwd`:/io -ti quay.io/pypa/manylinux1_x86_64 bash