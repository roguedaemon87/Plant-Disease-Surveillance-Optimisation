This folder should contain the raw GIS shapefiles used for beta parameterisation.

They are too large to host in this repository and are deposited at
<https://doi.org/10.17632/fw6mk5hsmx.1>, as `beta_parameterisation_shapefiles.zip`. Extract that archive into
this folder.

They are:
- Road network shapefile for northern DRC
- Production/host distribution shapefile for northern DRC

The road network derives from OpenStreetMap, (c) OpenStreetMap contributors,
and is available under the Open Database Licence (ODbL) rather than the licence
covering the rest of the deposit. The host distribution is built from the
production layer alone, so the road network does not affect the estimated
transmission rate.

Note:
Beta parameterisation is OPTIONAL and not required to reproduce the main
manuscript results, which use precomputed host distributions.
