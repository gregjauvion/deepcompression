# deepcompression

This library has been built to run some experiments related to data compression using deep learning methods.

The tests have been performed on 3 datasets:
* [ERA5](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview), which gives hourly weather historical data over a regular grid
* [ImageNet](https://www.image-net.org/)
* [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)

The package deepcompression.data contains the PyTorch objects to load and process those datasets.
The raw datasets are supposed to be downloaded in the tmp/ folder. They are all open datasets.

This work is based on:
* The paper [End-to-End Optimized Image Compression](https://arxiv.org/abs/1611.01704) by Ballé et al., introducing the data compression framework we use
* The python library [CompressAI](https://github.com/InterDigitalInc/CompressAI), in particular for its implementation of the Generalized Divisive Normalization layer in deepcompression.model.gdn.py
* The python library [Constriction](https://bamler-lab.github.io/constriction/) for its highly optimized entropy coder implementations

The package deepcompression.model implements the PyTorch modules, loss functions, training and evaluation loops. 

The package deepcompression.codec implements a codec that enables to compress / decompress data easily and efficiently
once the PyTorch model has been trained.

The package deepcompression.scripts contains some experiments performed using this library on the different datasets.
The compression methods based on deep learning are compared to the jpeg encoder whose OpenCV implementation is used
through the package deepcompression.benchmark.