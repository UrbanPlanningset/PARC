# Expected Local Data Layout

Data are intentionally not included in the anonymous source archive.

The core scripts expect a structure similar to:

```text
data/
  candidate_tiles/
    tiles12.csv
    fullscale_tiles.csv
  ig/
    scenario_manifest.csv
    <site_id>/
      dsm.tif
      dem.tif
      landcover_baseline.tif
      scenarios/
        baseline/
          summary/
            tmrt_day_mean.tif
            utci_day_mean.tif
            utci_night_mean.tif
        <scenario_id>/
          cdsm_tree_canopy.tif
          landcover.tif
          summary/
            tmrt_day_mean.tif
            utci_day_mean.tif
            utci_night_mean.tif
```

Candidate-tile CSV files must provide at least `site_id` and `city`. Additional
columns used during raster preparation depend on the selected GIS source and
should be inspected through each script's `--help` output.

SOLWEIG inputs and outputs must share a coordinate reference system and nominal
5 m grid resolution. The surrogate-training loader crops small shape drift to
the common geographic extent.
