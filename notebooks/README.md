# PIAA Notebooks

This folder contains a number of jupyter notebooks that are used in various ways, either to explore the data or to help generate assets (e.g. figures for papers).

The notebooks are organized into subfolders depending on their functionality or what aspects of the system they involve.

### Assets

* **[Superpixels-And-Stamps](assets/Superpixels-And-Stamps.ipynb):** Generates an image of the superpixel as an example to be used in papers, etc.

  ![Bayer array demo](assets/bayer-demo.png)
  
* **[Observing-Run-Stats-and-FOV-Plot](assets/Observing-Run-Stats-and-FOV-Plot.ipynb):**  Aggregates the total number of images and exposure time for each sequence that has more than 30 minutes of exposure time. Also plots the FOV for all sequences. Meant to be informative but probably not exact. Also generates a latex table with same information.

  ![FOV Plot](assets/panoptes_observations_overview.png)
  
* **[RMS-Explore](assets/RMS-Explore.ipynb):**  Create histograms of the RMS achienved for each source in the observation, split along color channels. _Note: This notebook uses the stored rms values inside an HDF5 file and requires that the file exist before-hand. The HDF5 file generation is changing somewhat so this notebook will inevitably be updated to support that. As such, not much work is going into it right now and this may not work._

### Camera

* **[Bias-Frame-And-Readnoise](camera/Bias-Frame-And-Readnoise.ipynb):** Generate a master bias frame and examine some of its properties. _Note: this notebook will generate a master bias frame from a combination of 100 individual bias frames. This can take a considerable amount of memory and probably should't be run locally._

* **[Flat-Frame-And-Gain](camera/Flat-Frame-And-Gain.ipynb):** Generate a master flat frame and examine some of its properties. Also compute a simple system gain.

  ![Flat Histogram](assets/flat-hist-colors.png)