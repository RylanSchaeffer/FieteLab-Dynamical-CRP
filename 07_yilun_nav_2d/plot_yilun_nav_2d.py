import matplotlib.pyplot as plt
import matplotlib.cm
import matplotlib.colors
import numpy as np
import pandas as pd
import os
import seaborn as sns
from typing import Dict

import rncrp.plot.plot_general
from rncrp.plot.plot_general import algorithm_color_map


def plot_analyze_all_inf_algs_results(all_inf_algs_results_df: pd.DataFrame,
                                      plot_dir: str):
    os.makedirs(plot_dir, exist_ok=True)

    plot_fns = [
        plot_scores_by_alpha_colored_by_alg_cross_hyperparameters,
        rncrp.plot.plot_general.plot_num_clusters_by_alpha_colored_by_alg,
        rncrp.plot.plot_general.plot_runtime_by_alpha_colored_by_alg,
        # rncrp.plot.plot_general.plot_scores_by_alpha_colored_by_alg,
    ]

    for plot_fn in plot_fns:
        # try:
        plot_fn(sweep_results_df=all_inf_algs_results_df,
                plot_dir=plot_dir)
        # except Exception as e:
        #     print(f'Exception: {e}')

        # Close all figure windows to not interfere with next plots
        plt.close('all')
        print(f'Plotted {str(plot_fn)}')


def expand_prev(edge, ix):
    (x1, y1), (x2, y2) = edge
    offset = 0.05
    xmean, ymean = (x1 + x2) / 2., (y1 + y2) / 2.

    edges = []
    if ix == -1:
        edges = [((x1, y1), (x2, y2))]
    elif ix == 0:
        edges = [((x1, y1), (x1, ymean + offset)), ((x1, ymean - offset), (x2, y2))]
    elif ix == 1:
        edges = [((x1, y1), (xmean + offset, y1)), ((xmean - offset, y2), (x2, y2))]
    elif ix == 2:
        edges = [((x1, y1), (x1, ymean - offset)), ((x1, ymean + offset), (x2, y2))]
    elif ix == 3:
        edges = [((x1, y1), (xmean - offset, y1)), ((xmean + offset, y2), (x2, y2))]
    else:
        assert False

    return edges


def expand_next(edge, ix, room):
    (x1, y1), (x2, y2) = edge
    center, width, height = room
    room_xmin, room_xmax = center[0] - width, center[0] + width
    room_ymin, room_ymax = center[1] - height, center[1] + height
    xmean, ymean = (x1 + x2) / 2., (y1 + y2) / 2.
    offset = 0.05

    if ix == 0:
        edges = [((x1, y1), (x1, ymean + offset)), ((x1, ymean + offset), (room_xmin, ymean + offset)),
                 ((x1, ymean - offset), (room_xmin, ymean - offset)), ((x1, ymean - offset), (x2, y2))]
    elif ix == 1:
        edges = [((x1, y1), (xmean + offset, y1)), ((xmean + offset, y1), (xmean + offset, room_ymax)),
                 ((xmean - offset, y1), (xmean - offset, room_ymax)), ((xmean - offset, y2), (x2, y2))]
    elif ix == 2:
        edges = [((x1, y1), (x1, ymean - offset)), ((x1, ymean - offset), (room_xmax, ymean - offset)),
                 ((x1, ymean + offset), (room_xmax, ymean + offset)), ((x1, ymean + offset), (x2, y2))]
    elif ix == 3:
        edges = [((x1, y1), (xmean - offset, y1)), ((xmean - offset, y1), (xmean - offset, room_ymin)),
                 ((xmean + offset, y1), (xmean + offset, room_ymin)), ((xmean + offset, y2), (x2, y2))]
    else:
        assert False

    return edges


def convert_segments(room_list, edges):
    ix_prev = -1
    edges_total = []

    for i, room in enumerate(room_list):
        center, width, height = room
        xmin = center[0] - width
        xmax = center[0] + width
        ymin = center[1] - height
        ymax = center[1] + height

        edge1 = [(xmax, ymax), (xmax, ymin)]
        edge2 = [(xmax, ymin), (xmin, ymin)]
        edge3 = [(xmin, ymin), (xmin, ymax)]
        edge4 = [(xmin, ymax), (xmax, ymax)]

        edges_list = [edge1, edge2, edge3, edge4]

        edge_prev = expand_prev(edges_list[ix_prev], ix_prev)
        ix = edges[i]

        if i != len(room_list) - 1:
            edge_next = expand_next(edges_list[ix], ix, room_list[i + 1])
        else:
            edge_next = [edges_list[ix]]

        edge_seg = []
        for i in range(4):
            if i == ix_prev:
                edge_seg.extend(edge_prev)
            elif i == ix:
                edge_seg.extend(edge_next)
            else:
                edge_seg.append(edges_list[i])

        edges_total.extend(edge_seg)

        # assert ix_prev != ix

        if ix == 0:
            ix_prev = 2
        elif ix == 1:
            ix_prev = 3
        elif ix == 2:
            ix_prev = 0
        elif ix == 3:
            ix_prev = 1

    return edges_total


def plot_room_clusters_on_one_run(yilun_nav_2d_dataset: Dict[str, np.ndarray],
                                  cluster_assignment_posteriors: np.ndarray,
                                  run_config: Dict[str, dict],
                                  plot_dir: str,
                                  env_idx: int):
    edge = yilun_nav_2d_dataset['edges'][env_idx]
    room_list = yilun_nav_2d_dataset['room_lists'][env_idx]
    landmark = yilun_nav_2d_dataset['landmarks'][env_idx]
    viewpoint = yilun_nav_2d_dataset['points'][env_idx]

    room_list = [((r[0], r[1]), r[2], r[3]) for r in room_list]

    plt.close()
    segs = convert_segments(room_list, edge)
    for seg in segs:
        plt.plot([seg[0][0], seg[1][0]], [seg[0][1], seg[1][1]], color='k')

    # Add landmarks
    plt.scatter(landmark[:, 0],
                landmark[:, 1],
                label='Landmarks',
                s=7,
                color='k',
                marker='d')

    map_cluster_assignments = np.argmax(cluster_assignment_posteriors, axis=1)
    # Convert classes to colors
    # https://stackoverflow.com/a/32740814/4570472
    norm = matplotlib.colors.Normalize(
        vmin=np.min(map_cluster_assignments),
        vmax=np.max(map_cluster_assignments),
        clip=True)
    mapper = matplotlib.cm.ScalarMappable(
        norm=norm,
        cmap=plt.cm.Spectral)
    colors = mapper.to_rgba(map_cluster_assignments)

    plt.quiver(viewpoint[:-1, 0],
               viewpoint[:-1, 1],
               viewpoint[1:, 0] - viewpoint[:-1, 0],
               viewpoint[1:, 1] - viewpoint[:-1, 1],
               scale_units='xy',
               angles='xy',
               scale=0.5,
               # color='tab:blue',
               color=colors)
    # plt.plot(viewpoint[:, 0], viewpoint[:, 1], label='Trajectory')
    plt.legend()
    plt.setp(plt.gca().get_xticklabels(), visible=False)
    plt.setp(plt.gca().get_yticklabels(), visible=False)
    plt.title(
        rf"D-CRP($\alpha$={run_config['alpha']}), Comp Prior: Beta({run_config['beta_arg1']}, {run_config['beta_arg2']})")
    # plt.show()
    file_name = f"dyn={run_config['dynamics_str']}_a={run_config['dynamics_a']}_b={run_config['dynamics_b']}" \
                f"_alpha={run_config['alpha']}_b1={run_config['beta_arg1']}_b2={run_config['beta_arg2']}" \
                f"_env={env_idx}"
    if run_config['narrow_hallways']:
        file_name += '_narrowhallways'
    if run_config['finite_vision']:
        file_name += '_finitevision'
    plt.savefig(os.path.join(plot_dir, f'{file_name}.png'),
                bbox_inches='tight',
                dpi=300)
    print(f'Plotted and saved {file_name}')
    # plt.show()
    plt.close()


def plot_scores_by_alpha_colored_by_alg_cross_hyperparameters(
        sweep_results_df: pd.DataFrame,
        plot_dir: str,
        title_str: str = None):
    scores_columns = [col for col in sweep_results_df.columns.values
                      if 'Score' in col]

    unique_dynamics_as = sweep_results_df['dynamics_a'].unique()

    for score_column in scores_columns:

        plt.close()

        # Manually make figure bigger to handle external legend
        fig, axes = plt.subplots(
            nrows=1,
            ncols=1 + len(unique_dynamics_as),
            sharex=True,
            sharey=True,
            figsize=(16, 4))

        # Plot R-CRP
        ax = axes[0]
        ax.set_title(r'R-CRP($\alpha=2$)')
        g = sns.lineplot(data=sweep_results_df[sweep_results_df['inference_alg_str'] == 'Recursive-CRP'],
                         x='beta_arg1',
                         y=score_column,
                         hue='beta_arg2',
                         legend='full',
                         ax=ax)
        ax.set_xlabel('Beta Prior Param 1')
        g.get_legend().set_title('Beta Prior Param 2')

        # Plot D-CRP for each dynamics a
        for ax_idx, unique_dynamics_a in enumerate(unique_dynamics_as, start=1):
            ax = axes[ax_idx]
            ax.set_title(rf'D-CRP($\exp(\tau={unique_dynamics_a}), \alpha=2$)')
            sweep_results_subset_df = sweep_results_df[
                (sweep_results_df['inference_alg_str'] == 'Dynamical-CRP') &
                (sweep_results_df['dynamics_a'] == unique_dynamics_a)]
            g = sns.lineplot(data=sweep_results_subset_df,
                             x='beta_arg1',
                             y=score_column,
                             hue='beta_arg2',
                             legend='full',
                             ax=ax)
            ax.set_xlabel('Beta Prior Param 1')
            g.get_legend().set_title('Beta Prior Param 2')

        if title_str is not None:
            plt.title(title_str)

        # plt.ylim(0., 1.05)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir,
                                 f'comparison_score={score_column}_by_alpha.png'),
                    bbox_inches='tight',
                    dpi=300)
        # plt.show()
        plt.close()
