# Data

- `covid_tracking_2021`
  - Link: https://data.humdata.org/dataset/novel-coronavirus-2019-ncov-cases
  - Note: We used the file `time_series_covid19_confirmed_global_iso3_regions.csv`


# Dinari README

### Datasets for Sampling in Dirichlet Process Mixture Models for Clustering Streaming Data

In this folder there are several datasets which we have used in our paper.
All datasets are in CSV format, where the last column is the label, the real datasets with points in `R^D` are all normalized by reducing the mean and dividing by the std, and are exactly as we used in the paper. None of the datasets require additional normalization.

All the datasets are meant to be used in batches of size `1000`, the various concept drifts are inserted when taking in account such batch size. This is the size which we have used in the paper.

#### R^D data -

* `2d_moving_clusters.csv` - Contains `10^7` points, sampled from 20 clusters, with incremental concept drift - On each batch the mean of each of the clusters move a random (small) length in some random direction, the means move independent of each other.

* `forest_normalized.csv` - This is the CoverType data (https://www.sciencedirect.com/science/article/abs/pii/S0168169999000460), using the numerical 10 features, which describes a 30x30 sqft of forest. Normalized and with no concept drift added.

* `imagenet_short.csv` - A subset of the original ImageNet(https://www.image-net.org/) data, containing 100 classes in 125K samples. SWAV (https://arxiv.org/abs/2006.09882) was used to extract features, and PCA to reduce dimensionality to 64. Data is normalized with recurring concept drifts, e.g. classes appear, disappear and reappear as the batches progress.

* `imagenet_full.csv` - The full ImageNet (ILSRCV2012) train data, containing 1000 classes in 1.25MIL samples. SWAV (https://arxiv.org/abs/2006.09882) was used to extract features, and PCA to reduce dimensionality to 128. Data is normalized with recurring concept drifts, e.g. classes appear, disappear and reappear as the batches progress.

### Counts Data - 

* `multinomial.csv` - `10^7` points sampled from 100 different multinomials distributions, with gradual concept drift inserted (classes change overtime).

* `20newsgroup.csv` - The 20newsgroup dataset(http://qwone.com/~jason/20Newsgroups/), using only the most common 1k words (not including words such as `if`, `for`, `and` etc... ), each point is the counts of those words in a document.