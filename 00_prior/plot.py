import matplotlib as mpl
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.special
import scipy.stats
import seaborn as sns

alphas_color_map = {
    1.1: 'tab:blue',
    10.78: 'tab:orange',
    15.37: 'tab:purple',
    30.91: 'tab:green'
}

plt.rcParams.update({'font.size': 20})


def plot_customer_assignments_analytical_vs_monte_carlo(sampled_customer_assignments_by_customer,
                                                        analytical_customer_assignments_by_customer,
                                                        alpha: float,
                                                        beta: float,
                                                        plot_dir: str,
                                                        dynamics_latex_str: str):
    # plot customer assignments, comparing analytics versus monte carlo estimates
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

    fig.suptitle(dynamics_latex_str)

    avg_sampled_customer_assignments_by_customer = np.mean(
        sampled_customer_assignments_by_customer, axis=0)

    # replace 0s with nans to allow for log scaling
    cutoff = np.nanmin(avg_sampled_customer_assignments_by_customer[
                           avg_sampled_customer_assignments_by_customer > 0.])
    cutoff_idx = avg_sampled_customer_assignments_by_customer < cutoff
    avg_sampled_customer_assignments_by_customer[cutoff_idx] = np.nan

    ax = axes[0]
    sns.heatmap(avg_sampled_customer_assignments_by_customer,
                ax=ax,
                mask=np.isnan(avg_sampled_customer_assignments_by_customer),
                cmap='jet',
                vmin=cutoff,
                vmax=1.,
                norm=LogNorm())

    ax.set_title(rf'Monte Carlo ($\alpha=${alpha})')  # , $\beta=${beta}
    ax.set_ylabel(r'Customer Index')
    ax.set_xlabel(r'Table Index')

    ax = axes[1]
    cutoff_idx = analytical_customer_assignments_by_customer < cutoff
    analytical_customer_assignments_by_customer[cutoff_idx] = np.nan
    sns.heatmap(analytical_customer_assignments_by_customer,
                ax=ax,
                mask=np.isnan(analytical_customer_assignments_by_customer),
                cmap='jet',
                vmin=cutoff,
                vmax=1.,
                norm=LogNorm(),
                )
    ax.set_title(rf'Analytical ($\alpha=${alpha})')  # , $\beta=${beta}
    ax.set_xlabel(r'Table Index')

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # left, bottom right, top in normalized coordinates

    # for some reason, on OpenMind, colorbar ticks disappear without calling plt.show() first
    fig.savefig(os.path.join(plot_dir, f'customer_assignments_monte_carlo_vs_analytical.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()


def plot_num_tables_analytical_vs_monte_carlo(sampled_num_tables_by_customer: np.ndarray,
                                              analytical_num_tables_by_customer: np.ndarray,
                                              alpha: float,
                                              beta: float,
                                              plot_dir: str,
                                              dynamics_latex_str: str):

    # plot customer assignments, comparing analytics versus monte carlo estimates
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

    fig.suptitle(dynamics_latex_str)

    avg_sampled_num_tables_by_customer = np.mean(
        sampled_num_tables_by_customer, axis=0)

    # replace 0s with nans to allow for log scaling
    nan_idx = avg_sampled_num_tables_by_customer == 0.
    avg_sampled_num_tables_by_customer[nan_idx] = np.nan
    cutoff = np.nanmin(avg_sampled_num_tables_by_customer)
    cutoff_idx = avg_sampled_num_tables_by_customer < cutoff
    avg_sampled_num_tables_by_customer[cutoff_idx] = np.nan

    ax = axes[0]
    sns.heatmap(avg_sampled_num_tables_by_customer,
                ax=ax,
                mask=np.isnan(avg_sampled_num_tables_by_customer),
                cmap='jet',
                norm=LogNorm(vmin=cutoff, vmax=1., ),
                )

    ax.set_title(rf'Monte Carlo ($\alpha=${alpha})')  # , $\beta=${beta}
    ax.set_ylabel(r'Customer Index')
    ax.set_xlabel(r'Table Index')

    ax = axes[1]
    cutoff_idx = analytical_num_tables_by_customer < cutoff
    analytical_num_tables_by_customer[cutoff_idx] = np.nan
    sns.heatmap(analytical_num_tables_by_customer,
                ax=ax,
                mask=np.isnan(analytical_num_tables_by_customer),
                cmap='jet',
                norm=LogNorm(vmin=cutoff, vmax=1., ),
                )
    ax.set_title(rf'Analytical ($\alpha=${alpha})')  # , $\beta=${beta}
    ax.set_xlabel(r'Table Index')

    # for some reason, on OpenMind, colorbar ticks disappear without calling plt.show() first
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # left, bottom right, top in normalized coordinates
    fig.savefig(os.path.join(plot_dir, f'num_tables_monte_carlo_vs_analytical.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()


def plot_analytics_vs_monte_carlo_table_occupancies(sampled_table_occupancies_by_alpha,
                                                    analytical_table_occupancies_by_alpha,
                                                    plot_dir):
    alphas = list(sampled_table_occupancies_by_alpha.keys())
    num_samples, T = sampled_table_occupancies_by_alpha[alphas[0]].shape
    table_nums = 1 + np.arange(T)
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
    for ax_idx, (alpha, crp_samples) in enumerate(sampled_table_occupancies_by_alpha.items()):
        row = ax_idx % 2
        col = int(ax_idx / 2)
        table_cutoff = alpha * np.log(1 + T / alpha)
        empiric_table_occupancies_mean_by_repeat = np.mean(crp_samples, axis=0)
        empiric_table_occupancies_sem = scipy.stats.sem(crp_samples, axis=0)
        axes[row, col].set_title(rf'CRP($\alpha$={alpha})')
        for num_samples_idx in range(200):
            axes[row, col].plot(table_nums, crp_samples[num_samples_idx, :], alpha=0.01, color='k')
        axes[row, col].errorbar(x=table_nums,
                                y=empiric_table_occupancies_mean_by_repeat,
                                yerr=empiric_table_occupancies_sem,
                                # linewidth=2,
                                fmt='--',
                                color='k',
                                label=f'Empiric (N={num_samples})')
        axes[row, col].scatter(table_nums[:len(analytical_table_occupancies_by_alpha[alpha])],
                               analytical_table_occupancies_by_alpha[alpha],
                               # '--',
                               marker='d',
                               color=alphas_color_map[alpha],
                               # linewidth=2,
                               label=f'Analytic')
        print(f'Plotted alpha={alpha}')
        axes[row, col].legend()
        if col == 0:
            axes[row, col].set_ylabel('Num. Table Occupants')
        if row == 1:
            axes[row, col].set_xlabel('Table Number')
        axes[row, col].set_xlim(1, table_cutoff)

    fig.savefig(os.path.join(plot_dir, f'analytics_vs_monte_carlo_table_occupancies.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()


def plot_analytical_vs_monte_carlo_mse(error_means_per_num_samples_per_alpha,
                                       error_sems_per_num_samples_per_alpha,
                                       num_reps,
                                       plot_dir):
    alphas = list(error_sems_per_num_samples_per_alpha.keys())

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 4))
    for alpha in alphas:
        ax.errorbar(x=list(error_means_per_num_samples_per_alpha[alpha].keys()),
                    y=list(error_means_per_num_samples_per_alpha[alpha].values()),
                    yerr=list(error_sems_per_num_samples_per_alpha[alpha].values()),
                    label=rf'$\alpha$={alpha}',
                    c=alphas_color_map[alpha])
    ax.legend(title=f'Num Repeats: {num_reps}')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel(r'$(Analytic - Monte Carlo Estimate)^2$')
    # ax.set_ylabel(r'$\mathbb{E}_D[\sum_k (\mathbb{E}[N_{T, k}] - \frac{1}{S} \sum_{s=1}^S N_{T, k}^{(s)})^2]$')
    ax.set_xlabel('Number of Monte Carlo Samples')
    fig.savefig(os.path.join(plot_dir, f'crp_expected_mse.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()


def plot_chinese_restaurant_table_dist_by_customer_num(analytical_table_distributions_by_alpha_by_T,
                                                       plot_dir):
    # plot how the CRT table distribution changes for T customers
    alphas = list(analytical_table_distributions_by_alpha_by_T.keys())
    T = len(analytical_table_distributions_by_alpha_by_T[alphas[0]])
    table_nums = 1 + np.arange(T)
    cmap = plt.get_cmap('jet_r')
    for alpha in alphas:
        for t in table_nums:
            plt.plot(table_nums,
                     analytical_table_distributions_by_alpha_by_T[alpha][t],
                     # label=f'T={t}',
                     color=cmap(float(t) / T))

        # https://stackoverflow.com/questions/43805821/matplotlib-add-colorbar-to-non-mappable-object
        norm = mpl.colors.Normalize(vmin=1, vmax=T)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        # sm.set_array([])
        colorbar = plt.colorbar(sm,
                                ticks=np.arange(1, T + 1, 5),
                                # boundaries=np.arange(-0.05, T + 0.1, .1)
                                )
        colorbar.set_label('Number of Customers')
        plt.title(fr'Chinese Restaurant Table Distribution ($\alpha$={alpha})')
        plt.xlabel(r'Number of Tables after T Customers')
        plt.ylabel(r'P(Number of Tables after T Customers)')
        plt.savefig(os.path.join(plot_dir, f'crt_table_distribution_alpha={alpha}.png'),
                    bbox_inches='tight',
                    dpi=300)
        # plt.show()
        plt.close()


def plot_recursion_visualization(analytical_customer_tables_by_alpha,
                                 analytical_table_distributions_by_alpha_by_T,
                                 plot_dir):
    alphas = list(analytical_customer_tables_by_alpha.keys())
    cutoff = 1e-8

    for alpha in alphas:
        fig, axes = plt.subplots(nrows=1,
                                 ncols=5,
                                 figsize=(13, 4),
                                 gridspec_kw={"width_ratios": [1, 0.1, 1, 0.1, 1]},
                                 sharex=True)

        ax = axes[0]
        cum_customer_seating_probs = np.cumsum(analytical_customer_tables_by_alpha[alpha], axis=0)
        cum_customer_seating_probs[cum_customer_seating_probs < cutoff] = np.nan

        max_table_idx = np.argmax(np.nansum(cum_customer_seating_probs, axis=0) < cutoff)
        sns.heatmap(
            data=cum_customer_seating_probs[:, :max_table_idx],
            ax=ax,
            cbar_kws=dict(label=r'$\sum_{t^{\prime} = 1}^{t-1} p(z_{t\prime} = k)$'),
            cmap='jet',
            mask=np.isnan(cum_customer_seating_probs[:, :max_table_idx]),
            norm=LogNorm(vmin=cutoff),
        )
        ax.set_xlabel('Table Index')
        ax.set_ylabel('Customer Index')
        ax.set_title('Running Sum of\nPrev. Customers\' Distributions')

        # necessary to make space for colorbar text
        axes[1].axis('off')

        ax = axes[2]
        table_distributions_by_T_array = np.stack([
            analytical_table_distributions_by_alpha_by_T[alpha][key]
            for key in sorted(analytical_table_distributions_by_alpha_by_T[alpha].keys())])
        table_distributions_by_T_array[table_distributions_by_T_array < cutoff] = np.nan
        sns.heatmap(
            data=table_distributions_by_T_array[:, :max_table_idx],
            ax=ax,
            cbar_kws=dict(label='$p(K_t = k)$'),
            cmap='jet',
            mask=np.isnan(table_distributions_by_T_array[:, :max_table_idx]),
            norm=LogNorm(vmin=cutoff, ),
        )
        ax.set_title('Distribution over\nNumber of Non-Empty Tables')
        ax.set_xlabel('Table Index')

        axes[3].axis('off')

        ax = axes[4]
        analytical_customer_tables = np.copy(analytical_customer_tables_by_alpha[alpha])
        analytical_customer_tables[analytical_customer_tables < cutoff] = np.nan
        sns.heatmap(
            data=analytical_customer_tables[:, :max_table_idx],
            ax=ax,
            cbar_kws=dict(label='$p(z_t)$'),
            cmap='jet',
            mask=np.isnan(analytical_customer_tables[:, :max_table_idx]),
            norm=LogNorm(vmin=cutoff, ),
        )
        ax.set_title('New Customer\'s Distribution')
        ax.set_xlabel('Table Index')
        fig.savefig(os.path.join(plot_dir, f'crp_recursion_alpha={alpha}.png'),
                    bbox_inches='tight',
                    dpi=300)
        # plt.show()
        plt.close()
