# NWLON_WebCOOS_Synchronizer

This is a package to create time-synchronized visualizations of [NOAA CO-OPS water level data](https://tidesandcurrents.noaa.gov/inundationdb/) and [WebCOOS webcamera imagery](https://webcoos.org/), including from pan-tilt-zoom (PTZ) cameras.


## Table of Contents
- [Installation](#installation)
- [WebCOOS API Key](#key)
- [Test](#test)
- [Usage](#usage)
- [Disclaimer](#disclaimer)
- [License](#license)


## Installation

For general use:

```bash
pip install git+https://github.com/NOAA-CO-OPS/NWLON_WebCOOS_Synchronizer.git#egg=nwlon_webcoos_synchronizer
```

For development:

```bash
git clone https://github.com/NOAA-CO-OPS/NWLON_WebCOOS_Synchronizer.git
cd NWLON_WebCOOS_Synchronizer
conda env create -f environment.yml
conda activate nwlon_webcoos_synchronizer
```


## WebCOOS API Key

To use the tool, you need a WebCOOS API Key. You can register for one [here](https://webcoos.org/docs/doc/access/)



## Test

Unit and integration tests are included. 

To run the tests, follow the "For development" installation instructions. Then, set your WebCOOS API key as an environment variable (replace 'your_key' with your actual WebCOOS API key, in single quotes):

Linux:
```bash
export API_KEY='your_key'
```

Windows:
```bash
setx API_KEY 'your_key'
```

Then you can run the tests with:

```bash
pytest -v
```


## Usage

Demo notebookes are provided in /demos to showcase package functionality.


## Disclaimer
#### NOAA Open Source Disclaimer:

This repository is a scientific product and is not official communication of the National Oceanic and Atmospheric Administration, or the United States Department of Commerce. All NOAA GitHub project code is provided on an ?as is? basis and the user assumes responsibility for its use. Any claims against the Department of Commerce or Department of Commerce bureaus stemming from the use of this GitHub project will be governed by all applicable Federal law. Any reference to specific commercial products, processes, or services by service mark, trademark, manufacturer, or otherwise, does not constitute or imply their endorsement, recommendation or favoring by the Department of Commerce. The Department of Commerce seal and logo, or the seal and logo of a DOC bureau, shall not be used in any manner to imply endorsement of any commercial product or activity by DOC or the United States Government.


## License

Software code created by U.S. Government employees is not subject to copyright in the United States (17 U.S.C. �105). The United States/Department of Commerce reserve all rights to seek and obtain copyright protection in countries other than the United States for Software authored in its entirety by the Department of Commerce. To this end, the Department of Commerce hereby grants to Recipient a royalty-free, nonexclusive license to use, copy, and create derivative works of the Software outside of the United States.
