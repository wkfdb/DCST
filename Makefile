pre:
	python -m pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
	cd thirdparty && unzip mmdetection.zip
	cd thirdparty/mmdetection && python -m pip install -e .
install:
	make pre
	python -m pip install -e .
clean:
	rm -rf thirdparty
	rm -r ssod.egg-info

