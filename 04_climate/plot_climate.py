import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import joblib

import geopandas as gpd
from geopandas import GeoDataFrame
from shapely.geometry import Point, Polygon

# common plotting functions
# import rncrp.plot.plot_general


def plot_analyze_all_inf_algs_results(all_inf_algs_results_df: pd.DataFrame,
                                      periodicity: str = 'monthly',
                                      with_or_without_subclasses: str = 'with',
                                      years_to_show: tuple = (1950, 1980, 2005, 2020),
                                      month_to_show: int = 12,
                                      plot_dir: str = None):
    os.makedirs(plot_dir, exist_ok=True)
    if with_or_without_subclasses == 'with':
        label_colname = 'climate_subtype'
    else:
        label_colname = 'climate_type'

    for i in range(1):
        sweep_dynamics_str_dir = '/om2/user/gkml/FieteLab-Recursive-Nonstationary-CRP/04_climate/results'
        # for dynamics_str, sweep_subset_results_df in all_inf_algs_results_df.groupby('dynamics_str'):
        #     sweep_dynamics_str_dir = os.path.join(plot_dir, dynamics_str)
        #     os.makedirs(sweep_dynamics_str_dir, exist_ok=True)
        #     print(f'Plotting dynamics {dynamics_str}')
        #
        #     if dynamics_str == 'step':
        #         title_str = r'$\Theta(\Delta)$'
        #     elif dynamics_str == 'exp':
        #         title_str = r'$\exp(-\Delta)$'
        #     elif dynamics_str == 'sinusoid':
        #         title_str = r'$\cos(\Delta)$'
        #     elif dynamics_str == 'hyperbolic':
        #         title_str = r'$\frac{1}{1 + \Delta}$'
        #     else:
        #         title_str = None
        #
        #     num_failed_loads = 0

        for i in range(1):
            # for inf_alg_results_joblib_path in sweep_subset_results_df['inf_alg_results_path']:
            #     try:
            #         joblib_file = joblib.load(inf_alg_results_joblib_path)
            #
            #     except TypeError:
            #         print(f'Error: could not load {inf_alg_results_joblib_path}')
            #         num_failed_loads += 1
            #         continue

            # # Load cluster assignments (inferred climate classifications)
            # cluster_assignment_posteriors = joblib_file['inference_alg_results']['cluster_assignment_posteriors']
            # inferred_cluster_assignments_per_obs = cluster_assignment_posteriors.argmax(axis=1)

            # Load observations and ground truth labels
            climate_df = pd.read_csv(
                f'/om2/user/gkml/FieteLab-Recursive-Nonstationary-CRP/exp2_climate/{periodicity}_climate_data_and_all_true_labels.csv')

            # Concatenate observations, ground truth labels, and cluster assignment and rename dataframe
            climate_df['inferred_label'] = climate_df[label_colname].copy().to_numpy()
            # climate_df['inferred_label'] = inferred_cluster_assignments_per_obs

            # Plot map visualizations
            plot_climate_clusters_over_time(obs_and_labels_df=climate_df,
                                            periodicity=periodicity,
                                            with_or_without_subclasses=with_or_without_subclasses,
                                            label_colname=label_colname,
                                            years_to_show=years_to_show,
                                            month_to_show=month_to_show,
                                            plot_dir=sweep_dynamics_str_dir,
                                            )
        # plot_fns = [
        #     rncrp.plot.plot_general.plot_num_clusters_by_alpha_colored_by_alg,
        #     rncrp.plot.plot_general.plot_runtime_by_alpha_colored_by_alg,
        #     rncrp.plot.plot_general.plot_runtime_by_dimension_colored_by_alg,
        #     rncrp.plot.plot_general.plot_scores_by_snr_colored_by_alg,
        #     rncrp.plot.plot_general.plot_scores_by_alpha_colored_by_alg,
        #     rncrp.plot.plot_general.plot_scores_by_dimension_colored_by_alg,
        # ]
        #
        # for plot_fn in plot_fns:
        #     # try:
        #     plot_fn(sweep_results_df=sweep_subset_results_df,
        #             plot_dir=sweep_dynamics_str_dir,
        #             title_str=title_str)
        #     # except Exception as e:
        #     #     print(f'Exception: {e}')
        #
        #     # Close all figure windows to not interfere with next plots
        #     plt.close('all')
        #     print(f'Plotted {str(plot_fn)}')


def plot_climate_clusters_over_time(obs_and_labels_df: pd.DataFrame,
                                    periodicity: str = 'monthly',
                                    with_or_without_subclasses: str = 'with',
                                    label_colname: str='climate_subtype',
                                    years_to_show: tuple = (1950, 1980, 2005, 2020),
                                    month_to_show: int = 12,
                                    plot_dir: str = None):
    """
    Generate maps displaying ground truth and inferred climate classifications for all sites at each of
        4 given years in the interval 1946-2015 (or 1946-2020).
    If using monthly data, defaults to showing climate classifications from the 12th month, December.
    """
    month_to_string = {1: 'January ', 2: 'February ', 3: 'March ', 4: 'April ', 5: 'May ', 6: 'June ', 7: 'July ',
                       8: 'August ', 9: 'September ', 10: 'October ', 11: 'November ', 12: 'December '}

    if periodicity == 'monthly':
        data_to_visualize = obs_and_labels_df.loc[obs_and_labels_df['MONTH'] == month_to_show]
        title_addendum = month_to_string[month_to_show]
    else:
        data_to_visualize = obs_and_labels_df
        title_addendum = ''

    # Generate and save plot for each year
    # world_map = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    # us_map = world_map[world_map.name == 'United States of America']
    us_map = gpd.read_file(
        '/om2/user/gkml/FieteLab-Recursive-Nonstationary-CRP/exp2_climate/Shapefiles/world_climates_completed_koppen_geiger.shp')

    for year in years_to_show:
        year_specific_data = data_to_visualize.loc[data_to_visualize['YEAR'] == year]
        geo_df = gpd.GeoDataFrame(year_specific_data,
                                  geometry=gpd.points_from_xy(
                                      year_specific_data['LONGITUDE'],
                                      year_specific_data['LATITUDE']))

        # Plot ground truth Koppen-Geiger climate classifications
        # fig_true, ax_true = plt.subplots(nrows=1, ncols=1, figsize=(5, 4))
        # ax_true = geo_df.plot(column=label_colname,
        #                       ax=us_map.plot(color='white', edgecolor='0.5', linewidth=0.2), cmap='Spectral',
        #                       marker='o', markersize=3);
        # ax_true.set_title('True Clusters, ' + title_addendum + str(year))
        # ax_true.set_aspect('auto')
        # ax_true.set_xlim(-128, -67)
        # ax_true.set_ylim(23, 53)
        # plt.xlabel('Longitude')
        # plt.ylabel('Latitude')
        # print("FIGURE SAVED TO:",
        #       plot_dir + f'/plot_climate_clusters_over_time_{periodicity}_{with_or_without_subclasses}_subclasses_{year}_true.png')
        # plt.savefig(os.path.join(plot_dir,
        #                          f'plot_climate_clusters_over_time_{periodicity}_{with_or_without_subclasses}_subclasses_{year}_true.png'),
        #             bbox_inches='tight',
        #             dpi=300)
        # plt.close()

        # Plot inferred climate classifications
        fig_inferred, ax_inferred = plt.subplots(nrows=1, ncols=1, figsize=(5, 4))
        ax_inferred = geo_df.copy().plot(column='inferred_label',
                                         ax=us_map.copy().plot(color='white', edgecolor='0.5', linewidth=0.2),
                                         cmap='Spectral', marker='o', markersize=3);
        ax_inferred.set_title('Inferred Clusters, ' + title_addendum + str(year))
        ax_inferred.set_aspect('auto')
        ax_inferred.set_xlim(-128, -67)
        ax_inferred.set_ylim(23, 53)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        print("FIGURE SAVED TO:",
              plot_dir + f'/plot_climate_clusters_over_time_{periodicity}_{with_or_without_subclasses}_subclasses_{year}_inferred.png')
        plt.savefig(os.path.join(plot_dir,
                                 f'plot_climate_clusters_over_time_{periodicity}_{with_or_without_subclasses}_subclasses_{year}_inferred.png'),
                    bbox_inches='tight',
                    dpi=300)
        plt.close()


plot_analyze_all_inf_algs_results(
    all_inf_algs_results_df=pd.DataFrame(),
    periodicity='annually',
    with_or_without_subclasses='without',
    plot_dir='placeholder',
)

# def plot_inference_results(sampled_climate_data: dict,
#                            inference_results: dict,
#                            inference_alg_str: str,
#                            inference_alg_param: float,
#                            plot_dir,
#                            num_tables_to_plot: int = 10):
#     assert isinstance(num_tables_to_plot, int)
#     assert num_tables_to_plot > 0
#     # num_obs = sampled_mog_results['gaussian_samples_seq'].shape[0]
#     # yticklabels = np.arange(num_obs)
#     # indices_to_keep = yticklabels % 10 == 0
#     # yticklabels += 1
#     # yticklabels = yticklabels.astype(np.str)
#     # yticklabels[~indices_to_keep] = ''
#
#     xmin = 1.1 * np.min(sampled_mog_data['gaussian_samples_seq'][:, 0])
#     xmax = 1.1 * np.max(sampled_mog_data['gaussian_samples_seq'][:, 0])
#     ymin = 1.1 * np.min(sampled_mog_data['gaussian_samples_seq'][:, 1])
#     ymax = 1.1 * np.max(sampled_mog_data['gaussian_samples_seq'][:, 1])
#
#     fig, axes = plt.subplots(nrows=1,
#                              ncols=3,
#                              figsize=(12, 4))
#
#     ax_idx = 0
#     # plot ground truth data
#     ax = axes[ax_idx]
#     sns.scatterplot(x=sampled_mog_data['gaussian_samples_seq'][:, 0],
#                     y=sampled_mog_data['gaussian_samples_seq'][:, 1],
#                     hue=sampled_mog_data['assigned_table_seq'],
#                     ax=ax,
#                     palette='Set1',
#                     legend=False)
#     ax.set_xlim(xmin=xmin, xmax=xmax)
#     ax.set_ylim(ymin=ymin, ymax=ymax)
#     ax.set_title('Ground Truth Data')
#
#     # plot cluster centroids
#     ax_idx += 1
#     ax = axes[ax_idx]
#     ax.scatter(inference_results['cluster_parameters']['means'][:, 0],
#                inference_results['cluster_parameters']['means'][:, 1],
#                s=2 * inference_results['table_assignment_posteriors_running_sum'][-1, :],
#                facecolors='none',
#                edgecolors='k')
#     ax.set_xlim(xmin=xmin, xmax=xmax)
#     ax.set_ylim(ymin=ymin, ymax=ymax)
#     ax.set_title(r'Cluster Centroids $\mu_z$')
#
#     # plot predicted cluster labels
#     ax_idx += 1
#     ax = axes[ax_idx]
#     pred_cluster_labels = np.argmax(inference_results['table_assignment_posteriors'],
#                                     axis=1)
#     sns.scatterplot(x=sampled_mog_data['gaussian_samples_seq'][:, 0],
#                     y=sampled_mog_data['gaussian_samples_seq'][:, 1],
#                     hue=pred_cluster_labels,
#                     palette='Set1',
#                     legend=False)
#     ax.set_xlim(xmin=xmin, xmax=xmax)
#     ax.set_ylim(ymin=ymin, ymax=ymax)
#     ax.set_title(r'Predicted Cluster Labels')
#
#     plt.savefig(os.path.join(plot_dir,
#                              'inference_alg_results.png'),
#                 bbox_inches='tight',
#                 dpi=300)
#     # plt.show()
#     plt.close()
#
#     fig, axes = plt.subplots(nrows=1,
#                              ncols=2,
#                              figsize=(8, 4))
#
#     ax_idx = 0
#     # plot prior table assignments
#     ax = axes[ax_idx]
#     if 'table_assignment_priors' in inference_results:
#         sns.heatmap(inference_results['table_assignment_priors'][:, :num_tables_to_plot],
#                     ax=ax,
#                     cmap='Blues',
#                     xticklabels=1 + np.arange(num_tables_to_plot),
#                     # yticklabels=yticklabels
#                     mask=np.isnan(inference_results['table_assignment_priors'][:, :num_tables_to_plot]),
#                     vmin=0.,
#                     vmax=1.,
#                     )
#         ax.set_title(r'$P(z_t|o_{<t})$')
#         ax.set_ylabel('Observation Index')
#         ax.set_xlabel('Cluster Index')
#
#     # plot posterior table assignments
#     ax_idx += 1
#     ax = axes[ax_idx]
#     sns.heatmap(inference_results['table_assignment_posteriors'][:, :num_tables_to_plot],
#                 ax=ax,
#                 cmap='Blues',
#                 xticklabels=1 + np.arange(num_tables_to_plot),
#                 # yticklabels=yticklabels
#                 vmin=0.,
#                 vmax=1.
#                 )
#     ax.set_title(r'$P(z_t|o_{\leq t})$')
#     ax.set_ylabel('Observation Index')
#     ax.set_xlabel('Cluster Index')
#
#     plt.savefig(os.path.join(plot_dir,
#                              'pred_assignments.png'),
#                 bbox_inches='tight',
#                 dpi=300)
#     # plt.show()
#     plt.close()
