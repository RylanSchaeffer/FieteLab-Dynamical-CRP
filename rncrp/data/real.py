import itertools
import numpy as np
import pickle
import os
import pandas as pd
import torch
import torchvision
import scipy as sp
from scipy import stats
import sklearn as sk
from sklearn.datasets import fetch_openml
# import umap
# import umap.plot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List, Tuple

# import rncrp.helpers.morph_envir_preprocessing as pp
# import rncrp.helpers.morph_envir_utilities as u
# import rncrp.helpers.PlaceCellAnalysis as pc


def load_dataset(dataset_name: str,
                 dataset_kwargs: Dict = None,
                 data_dir: str = 'data',
                 ) -> Dict[str, np.ndarray]:

    if dataset_kwargs is None:
        dataset_kwargs = dict()

    if dataset_name == 'ames_housing_2011':
        load_dataset_fn = load_dataset_ames_housing_2011
    elif dataset_name == 'boston_housing_1993':
        load_dataset_fn = load_dataset_boston_housing_1993
    elif dataset_name == 'cancer_gene_expression_2016':
        load_dataset_fn = load_dataset_cancer_gene_expression_2016
    elif dataset_name == 'covid_hospital_treatment_2020':
        load_dataset_fn = load_dataset_covid_hospital_treatment_2020
    elif dataset_name == 'diabetes_hospitals_2014':
        load_dataset_fn = load_dataset_diabetes_hospitals_2014
    elif dataset_name == 'electric_grid_stability_2016':
        load_dataset_fn = load_dataset_electric_grid_stability_2016
    elif dataset_name == 'wisconsin_breast_cancer_1995':
        load_dataset_fn = load_dataset_wisconsin_breast_cancer_1995
    else:
        raise NotImplementedError

    dataset_dict = load_dataset_fn(
        data_dir=data_dir,
        **dataset_kwargs)

    return dataset_dict


def load_dataset_ames_housing_2011(data_dir: str = 'data',
                                   **kwargs,
                                   ) -> Dict[str, pd.DataFrame]:

    ames_housing_bunch = fetch_openml(name="house_prices", as_frame=True)

    # Shape: (1460, 81)
    observations = ames_housing_bunch.frame

    # Convert sale price to target.
    labels = observations[['SalePrice']]
    # Also drop the pointless `Id` column.
    observations.drop(columns=['SalePrice', 'Id'], inplace=True)

    # Convert None to string "None"
    observations.replace(to_replace=[None], value="None", inplace=True)

    # Convert categorical columns to dummies
    columns_to_convert_to_one_hot = observations.columns[
        observations.dtypes.eq('object')]

    observations = pd.get_dummies(
        observations,
        drop_first=True,
        columns=columns_to_convert_to_one_hot)

    # Exclude any row containing any NaN
    obs_rows_with_nan = observations.isna().any(axis=1)
    print(f'Dropping {obs_rows_with_nan.sum()}/{len(obs_rows_with_nan)} rows containing NaNs')
    label_rows_with_nan = observations.isna().any(axis=1)
    rows_without_nan = ~(obs_rows_with_nan | label_rows_with_nan)
    observations = observations[rows_without_nan]
    labels = labels[rows_without_nan]

    observations['TimeSold'] = observations['YrSold'] + observations['MoSold'] / 12.

    # Reset indices, check matching shape
    assert observations.shape[0] == labels.shape[0]
    observations.reset_index(drop=True, inplace=True)
    labels.reset_index(drop=True, inplace=True)

    dataset_dict = dict(
        observations=observations,
        labels=labels,
    )

    return dataset_dict


def load_dataset_boston_housing_1993(data_dir: str = 'data',
                                     **kwargs,
                                     ) -> Dict[str, np.ndarray]:
    """
    Properties:
      - dtype: Real, Binary
      - Samples: 506
      - Dimensions: 12
      - Link: https://www.kaggle.com/arslanali4343/real-estate-dataset
    """
    dataset_dir = os.path.join(data_dir,
                               'boston_housing_1993')
    observations_path = os.path.join(dataset_dir, 'data.csv')

    data = pd.read_csv(observations_path, index_col=False)
    observations = data.loc[:, ~data.columns.isin(['MEDV'])]

    # MEDV Median value of owner-occupied homes in $1000's
    # Without rounding, there are 231 classes. With rounding, there are 48.
    labels = data['MEDV'].round().astype('category').cat.codes

    dataset_dict = dict(
        observations=observations.values,
        labels=labels.values,
    )

    return dataset_dict



def transform_site_csv_to_array(site_df,
                                duration: str = 'annual',
                                end_year: int = 2020,
                                use_zscores: bool = False):
    df = site_df.copy()
    df["year"] = df.DATE.apply(lambda x: int(x[:4]))
    df = df[(df.year >= 1946) & (df.year <= end_year)]

    if duration == 'annual':
        iterable = list(range(1946, end_year+1))
        index = pd.Index(iterable, name="year")
        outdf = pd.DataFrame(columns=index).T

        outdf.loc[:, "YEAR"] = iterable
        outdf.loc[:, "LATITUDE"] = np.ones(len(outdf.index)) * df.LATITUDE.iloc[0]
        outdf.loc[:, "LONGITUDE"] = np.ones(len(outdf.index)) * df.LONGITUDE.iloc[0]
        outdf.loc[:, "ELEVATION"] = np.ones(len(outdf.index)) * df.ELEVATION.iloc[0]
        outdf.loc[:, "TMAX"] = df.groupby(["year"]).TMAX.mean().reset_index().set_index(["year"])
        outdf.loc[:, "TMIN"] = df.groupby(["year"]).TMIN.mean().reset_index().set_index(["year"])
        outdf.loc[:, "PRCP"] = df.groupby(["year"]).PRCP.mean().reset_index().set_index(["year"])

        site_metadata = outdf.YEAR.to_numpy()

    elif duration == 'monthly':
        df["month"] = df.DATE.apply(lambda x: int(x[5:7]))

        year_iterable = list(range(1946, end_year+1))
        month_iterable = list(range(1, 13))
        index = pd.MultiIndex.from_product([year_iterable, month_iterable], names=["year", "month"])
        outdf = pd.DataFrame(columns=index).T

        outdf.loc[:, "YEAR"] = np.repeat(year_iterable, 12)
        outdf.loc[:, "MONTH"] = month_iterable * len(year_iterable)
        outdf.loc[:, "LATITUDE"] = np.ones(len(outdf.index)) * df.LATITUDE.iloc[0]
        outdf.loc[:, "LONGITUDE"] = np.ones(len(outdf.index)) * df.LONGITUDE.iloc[0]
        outdf.loc[:, "ELEVATION"] = np.ones(len(outdf.index)) * df.ELEVATION.iloc[0]
        outdf.loc[:, "TMAX"] = df.groupby(["year", "month"]).TMAX.mean().reset_index().set_index(["year", "month"])
        outdf.loc[:, "TMIN"] = df.groupby(["year", "month"]).TMIN.mean().reset_index().set_index(["year", "month"])
        outdf.loc[:, "PRCP"] = df.groupby(["year", "month"]).PRCP.mean().reset_index().set_index(["year", "month"])

        site_metadata = np.vstack((outdf.YEAR.to_numpy(),
                                   outdf.MONTH.to_numpy()))
    else:
        raise ValueError('Impermissible computation interval:', duration)

    # Finish concatenating site metadata
    site_metadata = np.vstack((site_metadata,
                               outdf.LATITUDE.to_numpy(),
                               outdf.LONGITUDE.to_numpy(),
                               outdf.ELEVATION.to_numpy()))

    # Concatenate climate metrics into 3 x (# years or # months) array
    site_metrics = np.vstack((outdf.TMAX.to_numpy(),
                              outdf.TMIN.to_numpy(),
                              outdf.PRCP.to_numpy()))

    # Convert climate metrics to z-scores if want to look more specifically at climate variability of each site
    if use_zscores:
        site_zscores = stats.zscore(site_metrics, axis=1, nan_policy='raise')
        site_array = np.vstack((site_metadata, site_zscores))

    # Otherwise, study overall climate by using raw data values
    else:
        site_array = np.vstack((site_metadata, site_metrics))

    return site_array.T

def create_climate_data(qualifying_sites_path: str = '/om2/user/gkml/FieteLab-Recursive-Nonstationary-CRP/exp2_climate/qualifying_sites_2020.txt',
                        duration: str = 'annual',
                        end_year: int = 2020,
                        use_zscores: bool = False):
    dataset = None
    with open(qualifying_sites_path) as file:
        for site_csv_path in file:
            if '.csv' in site_csv_path:
                try:
                    df = pd.read_csv(site_csv_path.strip(), low_memory=False)
                    site_array = transform_site_csv_to_array(df, duration, end_year, use_zscores)
                    if type(dataset) is not np.ndarray:
                        dataset = site_array
                    else:
                        dataset = np.vstack((dataset,site_array))
                except:
                    print("Invalid File: ", site_csv_path)

    return dataset

def load_dataset_climate(qualifying_sites_path: str = '/om2/user/gkml/FieteLab-Recursive-Nonstationary-CRP/exp2_climate/qualifying_sites_',
                         end_year: int = 2020,
                         use_zscores: bool = False):
    qualifying_sites_dir = qualifying_sites_path + str(end_year) + '.txt'
    annual_data = create_climate_data(qualifying_sites_dir, 'annual', end_year, use_zscores)
    monthly_data = create_climate_data(qualifying_sites_dir, 'monthly', end_year, use_zscores)

    true_cluster_labels = None ## TODO: OBTAIN GROUND TRUTH CLUSTERS

<<<<<<< HEAD
    labels = None

    dataset_dict = dict(
        observations=annual_data, #or monthly data?
        labels=labels,
    )

    return dataset_dict
=======
    climate_data_results = dict(
        monthly_data=monthly_data,
        annual_data=annual_data,
        cluster_assignments=true_cluster_labels)
    return climate_data_results
>>>>>>> dc99b45 (Modified climate dataset; climate runs almost ready)


def load_dataset_covid_tracking_2021(data_dir: str = 'data',
                                     **kwargs,
                                     ) -> Dict[str, np.ndarray]:
    dataset_dir = os.path.join(data_dir, 'covid_tracking_2021')
    data_path = os.path.join(
        dataset_dir, 'time_series_covid19_confirmed_global_iso3_regions.csv')

    raw_data = pd.read_csv(data_path,
                           header=[0, 1],  # first two lines are both headers
                           )
    # array of e.g. ('1/22/20', 'Unnamed: 4_level_1')
    date_columns = raw_data.columns.values[4:]

    # I don't know how to access MultiIndexes
    # TODO: Figure out a more elegant way of doing this
    unmelted_data = pd.DataFrame({
        'Latitude': raw_data['Lat']['#geo+lat'].values,
        'Longitude': raw_data['Long']['#geo+lon'].values,
        'Country': raw_data['Country/Region']['#country+name'].values,
        'Province': raw_data['Province/State']['#adm1+name'].values,
    })
    for date_column in date_columns:
        # Take only the first element of the tuple
        # e.g. ('1/22/20', 'Unnamed: 4_level_1') becomes '1/22/20'
        unmelted_data[date_column[0]] = raw_data[date_column]

    nondate_columns = unmelted_data.columns.values[:4]
    data = unmelted_data.melt(id_vars=nondate_columns,
                              var_name='Date',
                              value_name='Num Cases')

    observations = unmelted_data.loc[:, ~unmelted_data.columns.isin(['id', 'diagnosis'])]
    labels = unmelted_data['diagnosis'].astype('category').cat.codes

    dataset_dict = dict(
        observations=observations.values,
        labels=labels.values,
    )

    return dataset_dict


def load_dataset_diabetes_hospitals_2014(data_dir: str = 'data',
                                         **kwargs,
                                         ) -> Dict[str, np.ndarray]:
    dataset_dir = os.path.join(data_dir,
                               'diabetes_hospitals_2014')
    data_path = os.path.join(dataset_dir, 'diabetic_data.csv')

    data = pd.read_csv(data_path, index_col=False, na_values=['?'])

    # drop unwanted columns e.g. patient number, encounter id
    unwanted_columns = ['encounter_id',
                        'patient_nbr',
                        'weight',  # most are empty
                        'payer_code',
                        'medical_specialty',
                        ]

    data = data.loc[:, ~data.columns.isin(unwanted_columns)]

    label_columns = [
        # 'admission_type_id',
        #                       'discharge_disposition_id',
        #                       'admission_source_id',
        'readmitted',
    ]

    labels = data[label_columns]
    labels = pd.get_dummies(
        labels,
        drop_first=True,
        columns=label_columns)

    observations = data.loc[:, ~data.columns.isin(label_columns)]

    # These three columns are numeric but occasionally have strings
    # Forcibly coerce
    observations[['diag_1', 'diag_2', 'diag_3']] = observations[['diag_1', 'diag_2', 'diag_3']].apply(
        pd.to_numeric, errors='coerce', downcast='float'
    )

    columns_to_convert_to_one_hot = observations.columns[
        observations.dtypes.eq('object')]

    observations = pd.get_dummies(
        observations,
        drop_first=True,
        columns=columns_to_convert_to_one_hot)

    # Drop rows with NaNs
    obs_rows_with_nan = observations.isna().any(axis=1)
    print(f'Dropping {obs_rows_with_nan.sum()}/{len(obs_rows_with_nan)} rows containing NaNs')
    label_rows_with_nan = observations.isna().any(axis=1)
    rows_without_nan = ~(obs_rows_with_nan | label_rows_with_nan)
    observations = observations[rows_without_nan]
    labels = labels[rows_without_nan]

    dataset_dict = dict(
        observations=observations.values.astype(np.float32),
        labels=labels.values.astype(np.float32),
    )

    return dataset_dict


def load_dataset_electric_grid_stability_2016(data_dir: str = 'data',
                                              **kwargs,
                                              ) -> Dict[str, np.ndarray]:
    dataset_dir = os.path.join(data_dir,
                               'electric_grid_stability_2016')
    data_path = os.path.join(dataset_dir, 'smart_grid_stability_augmented.csv')

    data = pd.read_csv(data_path, index_col=False)
    observations = data.loc[:, ~data.columns.isin(['stab', 'stabf'])]

    # Rather than using binary 'stabf' as the class, use deciles (arbitrarily chosen)
    labels = pd.qcut(data['stab'], 10).astype('category').cat.codes

    dataset_dict = dict(
        observations=observations.values,
        labels=labels.values,
    )

    return dataset_dict


def load_dataset_moseq(data_dir: str = 'data'):
    """
    Load MoSeq data from https://github.com/dattalab/moseq-drugs.

    Note: You need to download the data from https://doi.org/10.5281/zenodo.3951698.
    Unfortunately, the data appears to have been pickled using Python2.7 and now
    NumPy and joblib can't read it. I used Python2.7 and Joblib to dump the data
    to disk using the following code:


    moseq_dir = os.path.join('data/moseq_drugs')
    file_names = [
        'dataset',
        'fingerprints',
        'syllablelabels'
    ]
    data = dict()
    for file_name in file_names:
        data[file_name] = np.load(os.path.join(moseq_dir, file_name + '.pkl'),
                                  allow_pickle=True)
    joblib.dump(value=data,
                filename=os.path.join(moseq_dir, 'moseq_drugs_data.joblib'))
    """

    moseq_dir = os.path.join(data_dir, 'moseq_drugs')
    data = joblib.load(filename=os.path.join(moseq_dir, 'moseq_drugs_data.joblib'))
    dataset = data['dataset']
    fingerprints = data['fingerprints']
    syllablelabels = data['syllablelabels']

    moseq_dataset_results = dict(
        dataset=dataset,
        fingerprints=fingerprints,
        syllablelabels=syllablelabels)

    return moseq_dataset_results


def load_dataset_newsgroup(data_dir: str = 'data',
                           num_data: int = None,
                           num_features: int = 500,
                           tf_or_tfidf_or_counts: str = 'tfidf'):
    assert tf_or_tfidf_or_counts in {'tf', 'tfidf', 'counts'}

    # categories = ['soc.religion.christian', 'comp.graphics', 'sci.med']
    categories = None  # set to None for all categories

    twenty_train = sklearn.datasets.fetch_20newsgroups(
        data_home=data_dir,
        subset='train',  # can switch to 'test'
        categories=categories,
        shuffle=True,
        random_state=0)

    class_names = np.array(twenty_train.target_names)
    true_cluster_labels = twenty_train.target
    true_cluster_label_strs = class_names[true_cluster_labels]
    observations = twenty_train.data

    if num_data is None:
        num_data = len(class_names)
    observations = observations[:num_data]
    true_cluster_labels = true_cluster_labels[:num_data]
    true_cluster_label_strs = true_cluster_label_strs[:num_data]

    if tf_or_tfidf_or_counts == 'tf':
        feature_extractor = sklearn.feature_extraction.text.TfidfVectorizer(
            max_features=num_features,  # Lin 2013 used 5000
            sublinear_tf=False,
            use_idf=False,
        )
        observations_transformed = feature_extractor.fit_transform(observations)

    elif tf_or_tfidf_or_counts == 'tfidf':
        # convert documents' word counts to tf-idf (Term Frequency times Inverse Document Frequency)
        # equivalent to CountVectorizer() + TfidfTransformer()
        # for more info, see
        # https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
        # https://scikit-learn.org/stable/auto_examples/text/plot_document_clustering.html#sphx-glr-auto-examples-text-plot-document-clustering-py
        feature_extractor = sklearn.feature_extraction.text.TfidfVectorizer(
            max_features=num_features,  # Lin 2013 used 5000
            sublinear_tf=False,
            use_idf=True,
        )
        observations_transformed = feature_extractor.fit_transform(observations)
    elif tf_or_tfidf_or_counts == 'counts':
        feature_extractor = sklearn.feature_extraction.text.CountVectorizer(
            max_features=num_features)
        observations_transformed = feature_extractor.fit_transform(observations)
    else:
        raise ValueError

    # quoting from Lin NeurIPS 2013:
    # We pruned the vocabulary to 5000 words by removing stop words and
    # those with low TF-IDF scores, and obtained 150 topics by running LDA [3]
    # on a subset of 20K documents. We held out 10K documents for testing and use the
    # remaining to train the DPMM. We compared SVA,SVA-PM, and TVF.

    # possible likelihoods for TF-IDF data
    # https://stats.stackexchange.com/questions/271923/how-to-use-tfidf-vectors-with-multinomial-naive-bayes
    # https://stackoverflow.com/questions/43237286/how-can-we-use-tfidf-vectors-with-multinomial-naive-bayes
    newsgroup_dataset_results = dict(
        observations_transformed=observations_transformed.toarray(),  # remove .toarray() to keep CSR matrix
        true_cluster_label_strs=true_cluster_label_strs,
        assigned_table_seq=true_cluster_labels,
        feature_extractor=feature_extractor,
        feature_names=feature_extractor.get_feature_names(),
    )

    return newsgroup_dataset_results


def load_dataset_mnist(data_dir: str = 'data',
                       num_data: int = None,
                       center_crop: bool = False,
                       avg_pool: bool = False,
                       feature_extractor_method: str = 'pca'):
    assert feature_extractor_method in {'pca', None}
    transforms = [torchvision.transforms.ToTensor()]
    if center_crop:
        transforms.append(torchvision.transforms.CenterCrop((80, 80)))
    if avg_pool:
        transforms.append(torchvision.transforms.Lambda(
            lambda x: torch.nn.functional.avg_pool2d(x, kernel_size=9, stride=3)))
        raise NotImplementedError

    mnist_dataset = torchvision.datasets.MNIST(
        root=data_dir,
        download=True,
        transform=torchvision.transforms.Compose(transforms))

    if num_data is None:
        num_data = mnist_dataset.data.shape[0]
    indices = np.random.choice(np.arange(mnist_dataset.data.shape[0]),
                               size=num_data,
                               replace=False)
    observations = mnist_dataset.data[indices, :, :].numpy()
    labels = mnist_dataset.targets[indices].numpy()

    if feature_extractor_method == 'pca':
        from sklearn.decomposition import PCA
        image_height = mnist_dataset.data.shape[1]
        image_width = mnist_dataset.data.shape[2]
        reshaped_images = np.reshape(observations, newshape=(num_data, image_height * image_width))
        pca = PCA(n_components=50)
        pca_latents = pca.fit_transform(reshaped_images)
        image_features = np.reshape(pca.inverse_transform(pca_latents),
                                    newshape=(num_data, image_height, image_width))
        feature_extractor = pca
    elif feature_extractor_method is None:
        image_features = observations.reshape(observations.shape[0], -1)
        feature_extractor = None
    else:
        raise ValueError(f'Impermissible feature method: {feature_extractor_method}')

    # # visualize observations if curious
    # import matplotlib.pyplot as plt
    # for idx in range(10):
    #     plt.imshow(image_features[idx], cmap='gray')
    #     plt.show()

    mnist_dataset_results = dict(
        observations=observations,
        labels=labels,
        feature_extractor_method=feature_extractor_method,
        feature_extractor=feature_extractor,
        image_features=image_features,
    )

    return mnist_dataset_results


def load_dataset_omniglot(data_dir: str = 'data',
                          num_data: int = None,
                          center_crop: bool = True,
                          avg_pool: bool = False,
                          feature_extractor_method: str = 'pca',
                          shuffle=True):
    """

    """

    assert feature_extractor_method in {'pca', 'cnn', 'vae', 'vae_old', None}

    # https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
    transforms = [torchvision.transforms.ToTensor()]
    if center_crop:
        transforms.append(torchvision.transforms.CenterCrop((80, 80)))
    if avg_pool:
        transforms.append(torchvision.transforms.Lambda(
            lambda x: torch.nn.functional.avg_pool2d(x, kernel_size=9, stride=3)))

    omniglot_dataset = torchvision.datasets.Omniglot(
        root=data_dir,
        download=True,
        transform=torchvision.transforms.Compose(transforms))

    # truncate dataset for now

    if num_data is None:
        num_data = len(omniglot_dataset._flat_character_images)
    omniglot_dataset._flat_character_images = omniglot_dataset._flat_character_images[:num_data]
    dataset_size = len(omniglot_dataset._flat_character_images)

    omniglot_dataloader = torch.utils.data.DataLoader(
        dataset=omniglot_dataset,
        batch_size=1,
        shuffle=False)

    images, labels = [], []
    for image, label in omniglot_dataloader:
        labels.append(label)
        images.append(image[0, 0, :, :])
        # uncomment to deterministically append the first image
        # images.append(omniglot_dataset[0][0][0, :, :])
    images = torch.stack(images).numpy()

    # These might be swapped, but I think height = width for omniglot.
    _, image_height, image_width = images.shape
    labels = np.array(labels)

<<<<<<< HEAD
    if feature_extractor_method == 'pca':
        from sklearn.decomposition import PCA
        reshaped_images = np.reshape(images, newshape=(dataset_size, image_height * image_width))
        pca = PCA(n_components=20)
        pca_latents = pca.fit_transform(reshaped_images)
        image_features = pca.inverse_transform(pca_latents)
        # image_features = np.reshape(pca.inverse_transform(pca_latents),
        #                             newshape=(dataset_size, image_height, image_width))
        feature_extractor = pca
    elif feature_extractor_method == 'cnn':
        # # for some reason, omniglot uses 1 for background and 0 for stroke
        # # whereas MNIST uses 0 for background and 1 for stroke
        # # for consistency, we'll invert omniglot
        # images = 1. - images
        #
        # from utils.omniglot_feature_extraction import cnn_load
        # lenet = cnn_load()
        #
        # from skimage.transform import resize
        # downsized_images = np.stack([resize(image, output_shape=(28, 28))
        #                              for image in images])
        #
        # # import matplotlib.pyplot as plt
        # # plt.imshow(downsized_images[0], cmap='gray')
        # # plt.title('Test Downsized Omniglot')
        # # plt.show()
        #
        # # add channel dimension for CNN
        # reshaped_images = np.expand_dims(downsized_images, axis=1)
        #
        # # make sure dropout is turned off
        # lenet.eval()
        # image_features = lenet(torch.from_numpy(reshaped_images)).detach().numpy()
        #
        # feature_extractor = lenet
=======
    # if feature_extractor_method == 'pca':
    #     from sklearn.decomposition import PCA
    #     reshaped_images = np.reshape(images, newshape=(dataset_size, image_height * image_width))
    #     pca = PCA(n_components=20)
    #     pca_latents = pca.fit_transform(reshaped_images)
    #     image_features = pca.inverse_transform(pca_latents)
    #     # image_features = np.reshape(pca.inverse_transform(pca_latents),
    #     #                             newshape=(dataset_size, image_height, image_width))
    #     feature_extractor = pca

    # if feature_extractor_method == 'vae_new':
    #     vae_data = np.load(os.path.join(os.getcwd(),
    #                                     'data/omniglot_vae/omniglot_data.npz'))
    #     labels = vae_data['targets']

    #     # Order samples by label
    #     indices_for_sorting_labels = np.random.choice(np.arange(len(labels)),
    #                                                 size=num_data,
    #                                                 replace=False)
    #     labels = labels[indices_for_sorting_labels][:num_data]
    #     images = vae_data['images'][indices_for_sorting_labels][:num_data, :, :]
    #     image_features = vae_data['latents'][indices_for_sorting_labels][:num_data, :]
    #     feature_extractor = None

    if feature_extractor_method == 'vae':

        from omniglot_vae_feature_extractor import vae_load
        vae = vae_load(omniglot_dataset=omniglot_dataset)

        # convert to Tensor and add channel
        torch_images = torch.unsqueeze(torch.from_numpy(images), dim=1)
        vae.eval()
        # define the features as the VAE means
        vae_result = vae(torch_images)
        torch_image_features = vae_result['mu']
        image_features = torch_image_features.detach().numpy()

        feature_extractor = vae
>>>>>>> dc99b45 (Modified climate dataset; climate runs almost ready)

        raise NotImplementedError
    elif feature_extractor_method == 'vae':
        vae_data = np.load(os.path.join(os.getcwd(),
                                        'data/omniglot_vae/omniglot_data.npz'))
        labels = vae_data['targets']
        # indices_to_sort_labels = np.argsort(labels)
        indices_to_sort_labels = np.random.choice(
            np.arange(len(labels)),
            size=num_data,
            replace=False)
        # make sure labels are sorted so we get multiple instances of the same class
        labels = labels[indices_to_sort_labels][:num_data]
        images = vae_data['images'][indices_to_sort_labels][:num_data, :, :]
        image_features = vae_data['latents'][indices_to_sort_labels][:num_data, :]
        feature_extractor = None
    elif feature_extractor_method is None:
        image_features = np.reshape(
            images,
            newshape=(dataset_size, image_height * image_width))
        feature_extractor = None
    else:
        raise ValueError(f'Impermissible feature method: {feature_extractor_method}')

    # # visualize images if curious
    # import matplotlib.pyplot as plt
    # for idx in range(10):
    #     plt.imshow(image_features[idx], cmap='gray')
    #     plt.show()

    if shuffle:
        random_indices = np.random.choice(
            np.arange(num_data),
            size=num_data,
            replace=False)
        images = images[random_indices]
        labels = labels[random_indices]
        image_features = image_features[random_indices]

    omniglot_dataset_results = dict(
        images=images,
        labels=labels,
        feature_extractor_method=feature_extractor_method,
        feature_extractor=feature_extractor,
        image_features=image_features,
    )

    return omniglot_dataset_results


def load_dataset_reddit(num_data: int,
                        num_features: int,
                        tf_or_tfidf_or_counts='tfidf',
                        data_dir='data'):
    # TODO: rewrite this function to preprocess data similar to newsgroup
    os.makedirs(data_dir, exist_ok=True)

    # possible other alternative datasets:
    #   https://www.tensorflow.org/datasets/catalog/cnn_dailymail
    #   https://www.tensorflow.org/datasets/catalog/newsroom (also in sklearn)

    # Import here to avoid slowdown
    import tensorflow as tf
    import tensorflow_datasets as tfds

    # useful overview: https://www.tensorflow.org/datasets/overview
    # take only subset of data for speed: https://www.tensorflow.org/datasets/splits
    # specific dataset: https://www.tensorflow.org/datasets/catalog/reddit
    reddit_dataset, reddit_dataset_info = tfds.load(
        'reddit',
        split='train',  # [:1%]',
        shuffle_files=False,
        download=True,
        with_info=True,
        data_dir=data_dir)
    assert isinstance(reddit_dataset, tf.data.Dataset)
    # reddit_dataframe = pd.DataFrame(reddit_dataset.take(10))
    reddit_dataframe = tfds.as_dataframe(
        ds=reddit_dataset.take(num_data),
        ds_info=reddit_dataset_info)
    reddit_dataframe = pd.DataFrame(reddit_dataframe)

    true_cluster_label_strs = reddit_dataframe['subreddit'].values
    true_cluster_labels = reddit_dataframe['subreddit'].astype('category').cat.codes.values

    documents_text = reddit_dataframe['normalizedBody'].values

    # convert documents' word counts to tf-idf (Term Frequency times Inverse Document Frequency)
    # equivalent to CountVectorizer() + TfidfTransformer()
    # for more info, see
    # https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
    # https://scikit-learn.org/stable/auto_examples/text/plot_document_clustering.html#sphx-glr-auto-examples-text-plot-document-clustering-py
    tfidf_vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(
        max_features=5000,
        sublinear_tf=False)
    observations_transformed = tfidf_vectorizer.fit_transform(documents_text)

    # quoting from Lin NeurIPS 2013:
    # We pruned the vocabulary to 5000 words by removing stop words and
    # those with low TF-IDF scores, and obtained 150 topics by running LDA [3]
    # on a subset of 20K documents. We held out 10K documents for testing and use the
    # remaining to train the DPMM. We compared SVA,SVA-PM, and TVF.

    # possible likelihoods for TF-IDF data
    # https://stats.stackexchange.com/questions/271923/how-to-use-tfidf-vectors-with-multinomial-naive-bayes
    # https://stackoverflow.com/questions/43237286/how-can-we-use-tfidf-vectors-with-multinomial-naive-bayes
    reddit_dataset_results = dict(
        observations_transformed=observations_transformed.toarray(),  # remove .toarray() to keep CSR matrix
        true_cluster_label_strs=true_cluster_label_strs,
        assigned_table_seq=true_cluster_labels,
        tfidf_vectorizer=tfidf_vectorizer,
        feature_names=tfidf_vectorizer.get_feature_names(),
    )

    return reddit_dataset_results


def load_dataset_template(data_dir: str = 'data',
                          **kwargs,
                          ) -> Dict[str, np.ndarray]:
    dataset_dir = os.path.join(data_dir,
                               'wisconsin_breast_cancer_1995')
    data_path = os.path.join(dataset_dir, 'data.csv')

    data = pd.read_csv(data_path, index_col=False)
    observations = data.loc[:, ~data.columns.isin(['id', 'diagnosis'])]
    labels = data['diagnosis'].astype('category').cat.codes

    dataset_dict = dict(
        observations=observations.values,
        labels=labels.values,
    )

    return dataset_dict


def load_dataset_wisconsin_breast_cancer_1995(data_dir: str = 'data',
                                              **kwargs,
                                              ) -> Dict[str, np.ndarray]:
    """
    Properties:
      - dtype: Real
      - Samples: 569
      - Dimensions: 32
      - Link: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29
      - Data: https://www.kaggle.com/uciml/breast-cancer-wisconsin-data/version/2

    :param data_dir:
    :return:
    """
    dataset_dir = os.path.join(data_dir,
                               'wisconsin_breast_cancer_1995')
    data_path = os.path.join(dataset_dir, 'data.csv')
    data = pd.read_csv(data_path, index_col=False)

    observations = data.loc[:, ~data.columns.isin(['id', 'diagnosis'])]
    labels = data['diagnosis'].astype('category').cat.codes

    dataset_dict = dict(
        observations=observations.values,
        labels=labels.values,
    )

    return dataset_dict


def load_dataset_yale(num_data: int,
                      data_dir='data'):
    npzfile = np.load(os.path.join('data', 'yale_faces', 'yale_faces.npz'))
    data = dict(npzfile)
    train_data = data['train_data']
    test_data = data['test_data']
    # authors suggest withholding pixels from testing set with 0.3% probability
    # these are those pixels
    test_mask = data['test_mask']

    # image size is 32x32. Reshape or no?

    yale_dataset_results = dict(
        train_data=train_data,
        test_data=test_data,
        test_mask=test_mask)

    return yale_dataset_results


def transform_site_csv_to_array(site_df,
                                duration='annual'):
    df = site_df.copy()
    df["year"] = df.DATE.apply(lambda x: int(x[:4]))
    df = df[(df.year >= 1946) & (df.year <= 2020)]

    if duration == 'annual':
        iterable = list(range(1946, 2021))
        index = pd.Index(iterable, name="year")
        outdf = pd.DataFrame(columns=index).T

        outdf.loc[:, "TMAX"] = df.groupby(["year"]).TMAX.mean().reset_index().set_index(["year"])
        outdf.loc[:, "TMIN"] = df.groupby(["year"]).TMIN.mean().reset_index().set_index(["year"])
        outdf.loc[:, "PRCP"] = df.groupby(["year"]).PRCP.mean().reset_index().set_index(["year"])

    elif duration == 'monthly':
        df["month"] = df.DATE.apply(lambda x: int(x[5:7]))

        iterables = [list(range(1946, 2021)), list(range(1, 13))]
        index = pd.MultiIndex.from_product(iterables, names=["year", "month"])
        outdf = pd.DataFrame(columns=index).T

        outdf.loc[:, "TMAX"] = df.groupby(["year", "month"]).TMAX.mean().reset_index().set_index(["year", "month"])
        outdf.loc[:, "TMIN"] = df.groupby(["year", "month"]).TMIN.mean().reset_index().set_index(["year", "month"])
        outdf.loc[:, "PRCP"] = df.groupby(["year", "month"]).PRCP.mean().reset_index().set_index(["year", "month"])

    else:
        raise ValueError('Impermissible computation interval:', duration)

    site_array = np.hstack((outdf.TMAX.to_numpy(),
                            outdf.TMAX.to_numpy(),
                            outdf.PRCP.to_numpy()))
    return site_array


def create_climate_data(qualifying_sites_path: str = None,
                        duration: str = 'annual'):
    datalist = list()
    with open(qualifying_sites_path) as file:
        for site_csv_path in file:
            if '.csv' in site_csv_path:
                try:
                    df = pd.read_csv(site_csv_path.strip(), low_memory=False)
                    site_array = transform_site_csv_to_array(df, duration)
                    datalist.append(site_array)
                except:
                    print("Invalid File: ", site_csv_path)

    dataset = np.array(datalist)
    dataset = stats.zscore(dataset, axis=0, nan_policy='raise')
    return dataset


def load_dataset_climate(qualifying_sites_path: str = None):
    annual_data = create_climate_data(qualifying_sites_path, 'annual')
    monthly_data = create_climate_data(qualifying_sites_path, 'monthly')

    climate_data_results = dict(
        monthly_data=monthly_data,
        annual_data=annual_data)
    return climate_data_results


def load_dataset_morph_environment(data_dir: str = 'data',
                                   **kwargs,
                                   ) -> Dict[str, np.ndarray]:
    ### Initialize helper variables -- TO-DO: CHECK IF NEED THESE
    # getf = lambda s : s*2.5 + (1-s)*3.5
    # gettheta = lambda s: (s*60. + (1-s)*10.)*np.pi/180
    # xfreq = lambda s: np.abs(getf(s)*1500/120*(np.cos(gettheta(s) + np.pi/4.)-np.sin(gettheta(s)+np.pi/4.)))
    # yfreq = lambda s: np.abs(getf(s)*1500/120*(np.cos(gettheta(s) + np.pi/4.)+np.sin(gettheta(s)+np.pi/4.)))
    # ang = lambda x,y: np.arctan(x/y)*180/np.pi

    # wallmorphx = lambda s: 1.2*(xfreq(s)-xfreq(-.1))/(xfreq(1.1)-xfreq(-.1))-.1
    # wallmorphy = lambda s: 1.2*(yfreq(s)-yfreq(-.1))/(yfreq(1.1)-yfreq(-.1))-.1

    ### Load data
    df = pp.load_session_db(dir=data_dir)
    # df = pp.load_session_db()
    df = df[df['RewardCount'] > 40]
    df = df[df['Imaging'] == 1]
    df = df[(df['ImagingRegion'] == 'CA1') | (df['ImagingRegion'] == '')]
    df = df.sort_values(['MouseName', 'DateTime', 'SessionNumber'])
    df = df[df["Track"] == "TwoTower_foraging"]

    mice = ['4139265.3', '4139265.4', '4139265.5', '4222168.1', '4343703.1', '4343706', '4222153.1', '4222153.2',
            '4222153.3', '4222174.1', '4222154.1', '4343702.1']
    first_sess = [5, 5, 5, 3, 5, 2, 4, 4, 4, 4, 4, 4]
    rare = [i < 6 for i in range(len(mice))]
    freq = [(1 - r) > 0 for r in rare]
    print(rare, freq)

    ### Morph binning
    morphbin = np.linspace(-.11, 1.11, num=11)
    SM = []
    DIST_REG = []
    COSDIST_REG = []
    mouselabel = []
    rf_label = []
    sess_ind = []
    SIG_CELLS = {}

    for m, mouse in enumerate(mice):
        print(mouse)
        df_mouse = df[df["MouseName"] == mouse]
        SIG_CELLS[mouse] = {}
        for ind in range(first_sess[m], df_mouse.shape[0]):
            sess = df_mouse.iloc[ind]

            dist, centroid_diff, S_trial_mat, trial_info = pp.single_sess_dist(sess, metric='cosine')
            pval = pp.centroid_diff_perm_test(centroid_diff, S_trial_mat, trial_info, nperms=1000)

            morphs = trial_info['morphs'] + trial_info['wallJitter']
            morphOrder = np.argsort(morphs)
            sig_cells = pval <= .001
            S_tm_sig = S_trial_mat[:, :, sig_cells]

            dist_sig = dist[sig_cells, :]
            cd_sig = centroid_diff[sig_cells]

            dist_reg = pp.regress_distance(dist_sig, morphs)
            DIST_REG.append(dist_reg)

            morphdig = np.digitize(morphs, morphbin)
            S_tm_bin = np.zeros([10, S_tm_sig.shape[1], S_tm_sig.shape[2]])
            for i in range(10):
                if (morphdig == i).sum() > 0:
                    S_tm_bin[i, :, :] = S_tm_sig[morphdig == i, :, :].mean(axis=0)

            SM.append(S_tm_bin.reshape(-1, S_tm_bin.shape[-1]).T)
            mouselabel.append(m * np.ones((S_tm_bin.shape[-1],)))
            rf_label.append(1 * rare[m] * np.ones((S_tm_bin.shape[-1],)))
            sess_ind.append(ind * np.ones((S_tm_bin.shape[-1],)))
            SIG_CELLS[mouse][ind] = sig_cells

    ### Concatenate and reshape matrices
    sm = np.concatenate(SM, axis=0)
    dr = np.concatenate(DIST_REG, axis=0)
    ml = np.concatenate(mouselabel)
    rfl = np.concatenate(rf_label)
    # sess_ind = np.concatenate(sess_ind)
    print(sm.shape, ml.shape, rfl.shape)
    print(rfl.sum(), rfl.shape[0] - rfl.sum())
    print(np.unique(ml))


    ### TODO: GENERATE DATASET FOR CLUSTERING HERE

    # _sm = np.reshape(sm,(10,45,-1))
    # print(_sm.shape)

    ### UMAP dimensionality reduction & DBSCAN clustering to obtain 2 manifolds, corresponding to S=0 and S=1
    mapper = umap.UMAP(metric="correlation", n_neighbors=100, min_dist=.01, n_components=3).fit(sm)
    # umap.plot.points(mapper) # visualize if desired

    # clust_labels = sk.cluster.DBSCAN(min_samples=200).fit_predict(mapper.embedding_)
    # print(np.unique(clust_labels))
    # print((clust_labels==-1).sum())
    # c=1
    #
    # morphpref = np.argmax(sm.reshape(sm.shape[0],10,-1).mean(axis=-1),axis=-1)/10.
    # morphpref_clustlabels = 0*morphpref
    # morphpref_clustlabels[clust_labels==0]=np.mean(morphpref[clust_labels==0])
    # morphpref_clustlabels[clust_labels==1]=np.mean(morphpref[clust_labels==1])
    # f = plt.figure()
    # ax = f.add_subplot(111,projection='3d')
    # ax.scatter(mapper.embedding_[clust_labels>-1,0],mapper.embedding_[clust_labels>-1,1],mapper.embedding_[clust_labels>-1,2],c=1-morphpref_clustlabels[clust_labels>-1],cmap='cool',s=100/rfl.shape[0]**.5)

    # ### Generate dataset dictionary
    # morph_environment_dataset_results = {}
    # for c in [0,1]:
    #     clustmask = clust_labels==c
    #     clust_rfl = rfl[clustmask]
    #     clust_sm = sm[clustmask]
    #
    #     clust_embedding = mapper.embedding_[clustmask,:]
    #     morph_environment_dataset_results['subclust_labels_unsorted_c'+str(c)] = clust_embedding
    return morph_environment_dataset_results


if __name__ == '__main__':
    load_dataset(dataset_name='ames_housing_2011')
