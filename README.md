PIAA
====

[![Build Status](https://travis-ci.org/panoptes/PIAA.svg?branch=master)](https://travis-ci.org/panoptes/PIAA)

The PANOPTES Image Analysis Algorithm (PIAA) repository contains the data processing pipeline and algorithms for the images taken by PANOPTES observatories. PIAA currently handles storing the raw data, processing intermediate data and storing final data products. Eventually it will also analyze final data products and store results. 

For a detailed explanation of the system, design choices and alternatives, see the [Design Doc](https://docs.google.com/document/d/1GefBo-vYa6jKhT8LO7Z-fd7rppOMjD7AIczNJLBhS-o/edit#heading=h.xgjl2srtytjt).

### Development 

#### Setting up

Clone this repository to a directory named PIAA on your local machine and run
~~~
pip install -r requirements.txt
~~~
to install the dependencies. Set up everything as in the [Coding in PANOPTES wiki](https://github.com/panoptes/POCS/wiki/Coding-in-PANOPTES).