# deepcompression

This library has been built to run some experiments related to data compression using deep learning methods.

The tests have been performed on 3 datasets:
* ERA5, which gives hourly weather historical data over a regular grid (https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview)
* ImageNet (https://www.image-net.org/)
* Div2K (https://data.vision.ee.ethz.ch/cvl/DIV2K/)

The package deepcompression.data contains the PyTorch objects to load and process those datasets.
The raw datasets are supposed to be downloaded and in the tmp/ folder. They are all open datasets.

This work is based on:
* The paper End-to-End Optimized Image Compression by Ballé et al.
* CompressAI library (https://github.com/InterDigitalInc/CompressAI), in particular for the implementation of the Generalized Divisive Normalization layer in deepcompression.model.gdn.py
* The python Constriction library (https://bamler-lab.github.io/constriction/) for highly optimized entropy coding

The package deepcompression.model implements the PyTorch modules, loss functions, training and evaluation loops needed.

The package deepcompression.codec implements a codec that enables to compress / decompress data easily and efficiently.

The compression methods based on deep learning are compared to the jpeg encoder whose OpenCV implementation can be used
through the package deepcompression.benchmark.

The package deepcompression.scripts gives some experiments performed using this library.