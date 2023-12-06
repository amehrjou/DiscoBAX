"""
Copyright 2022 Pascal Notin, University of Oxford
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
import pandas as pd
import ast

result_location = (
    "/users/pastin/projects/discobax/output/20231115_DiscoBAX_Final_plot.csv"
)
result_location_processed = "/users/pastin/projects/discobax/output/20231115_DiscoBAX_Final_plot_processed_main.csv"
fig_suffix = "_Nov15_final_plots"
main_plot = False

if main_plot:
    plot_all_on_1_fig = True
    methods = [
        "discobax",
        "topk_bax",
        "levelset_bax",
        "ucb",
        "jepig",
        "thompson_sampling",
        "coreset",
        "random",
    ]
    dash_cutoff = 4
    fig_suffix += "_main"
else:
    plot_all_on_1_fig = False
    methods = [
        "discobax",
        "topk_bax",
        "levelset_bax",
        "ucb",
        "jepig",
        "thompson_sampling",
        "coreset",
        "random",
        "topuncertain",
        "marginsample",
        "badge",
        "kmeans_embedding",
        "kmeans_data",
        "adversarialBIM",
    ]
    target_datasets = [
        "schmidt_2021_ifng",
        "schmidt_2021_il2",
        "zhuang_2019_nk",
        "sanchez_2021_tau",
        "zhu_2021_sarscov2",
    ]
    legend_index = 2
    fig_x, fig_y = 22, 32
    dash_cutoff = 10
    fig_suffix += "_all_detailed"

plt.rcParams.update({"font.size": 18})


def main():
    try:
        performance_data = pd.read_csv(result_location, low_memory=False)
        print("Loaded without issue")
    except:
        print("Need to escape the lists in file")
        with open(result_location, "r") as f:
            lines = f.readlines()
        for line_index, line in enumerate(lines):
            lines[line_index] = line.replace("[", '"[').replace("]", ']"')
        with open(result_location_processed, "w") as f:
            for line in lines:
                f.write(line)
        performance_data = pd.read_csv(result_location_processed, low_memory=False)

    performance_data = performance_data.loc[performance_data.acquisition_cycle == 25, :]
    print("Check completion rate")
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(
            performance_data.groupby(
                ["feature_set_name", "dataset_name", "acquisition_function_name"]
            ).size()
        )

    feature_sets = ["achilles"]
    seeds = [
        1000,
        2000,
        3000,
        4000,
        5000,
        6000,
        7000,
        8000,
        9000,
        10000,
        11000,
        12000,
        13000,
        14000,
        15000,
        16000,
        17000,
        18000,
        19000,
        20000,
    ]

    pretty_method_names = {
        "random": "Random",
        "topk_bax": "Top-K BAX",
        "levelset_bax": "Levelset BAX",
        "discobax": "DiscoBAX",
        "jepig": "JEPIG",
        "ucb": "UCB",
        "thompson_sampling": "Thompson sampling",
        "topuncertain": "Top uncertainty",
        "softuncertain": "Soft uncertainty",
        "marginsample": "Margin sample",
        "badge": "BADGE",
        "coreset": "Coreset",
        "kmeans_embedding": "Kmeans embedding",
        "kmeans_data": "Kmeans data",
        "adversarialBIM": "Adversarial BIM",
    }
    pretty_dataset_names = {
        "schmidt_2021_ifng": "Interferon Gamma",
        "schmidt_2021_il2": "Interleukin 2",
        "zhuang_2019_nk": "Leukemia / NK cells",
        "sanchez_2021_tau": "Tau protein",
        "zhu_2021_sarscov2": "SARS-CoV-2",
    }

    sns.set_style("whitegrid")
    if not plot_all_on_1_fig:
        for feature_set in feature_sets:
            fig, axs = plt.subplots(len(target_datasets), 2, figsize=(fig_x, fig_y))

            for dataset_index, target_dataset in enumerate(target_datasets):
                for m_index, m in enumerate(methods):
                    agg_topk_precision_all_seeds = []
                    agg_topk_recall_all_seeds = []
                    agg_topk_cluster_recall_all_seeds = []
                    for seed in seeds:
                        try:
                            data_point = performance_data.loc[
                                (performance_data.feature_set_name == feature_set)
                                & (performance_data.dataset_name == target_dataset)
                                & (performance_data.acquisition_function_name == m)
                                & (performance_data.seed == seed)
                            ]
                            assert (
                                len(data_point) <= 1
                            ), "Duplicate runs found in the performance data for params: {} {} {} {}".format(
                                feature_set, target_dataset, m, seed
                            )
                            assert (
                                len(data_point) >= 1
                            ), "Seed run {} {} {} {} could not be found".format(
                                feature_set, target_dataset, m, seed
                            )
                            num_results = len(data_point)
                            if num_results > 1:
                                print(
                                    "Duplicate runs found in the performance data for params: {} {} {} {}".format(
                                        feature_set, target_dataset, m, seed
                                    )
                                )
                                print(data_point)
                            while num_results > 0:
                                agg_topk_precision_all_seeds.append(
                                    ast.literal_eval(
                                        data_point["cumulative_precision_topk"].values[
                                            num_results - 1
                                        ]
                                    )
                                )
                                agg_topk_recall_all_seeds.append(
                                    ast.literal_eval(
                                        data_point["cumulative_recall_topk"].values[
                                            num_results - 1
                                        ]
                                    )
                                )
                                agg_topk_cluster_recall_all_seeds.append(
                                    ast.literal_eval(
                                        data_point[
                                            "cumulative_proportion_top_clusters_recovered"
                                        ].values[num_results - 1]
                                    )
                                )
                                num_results -= 1
                        except:
                            pass
                    num_seeds_completed = len(agg_topk_precision_all_seeds)
                    mean_agg_topk_recall_all_seeds = np.mean(
                        np.array(agg_topk_recall_all_seeds), axis=0
                    )
                    mean_agg_topk_cluster_recall_all_seeds = np.mean(
                        np.array(agg_topk_cluster_recall_all_seeds), axis=0
                    )
                    stde_agg_topk_recall_all_seeds = (
                        np.std(np.array(agg_topk_recall_all_seeds), axis=0)
                        / (num_seeds_completed) ** 0.5
                    )
                    stde_agg_topk_cluster_recall_all_seeds = (
                        np.std(np.array(agg_topk_cluster_recall_all_seeds), axis=0)
                        / (num_seeds_completed) ** 0.5
                    )

                    if m == "discobax":
                        print(mean_agg_topk_cluster_recall_all_seeds)
                    linestyle = "dashed" if m_index >= dash_cutoff else "solid"
                    axs[dataset_index][0].plot(
                        mean_agg_topk_recall_all_seeds,
                        label=pretty_method_names[m],
                        linestyle=linestyle,
                    )
                    axs[dataset_index][1].plot(
                        mean_agg_topk_cluster_recall_all_seeds,
                        label=pretty_method_names[m],
                        linestyle=linestyle,
                    )

                axs[dataset_index][0].set_yticklabels(
                    ["{:,.0%}".format(x) for x in axs[dataset_index][0].get_yticks()]
                )
                axs[dataset_index][1].set_yticklabels(
                    ["{:,.0%}".format(x) for x in axs[dataset_index][1].get_yticks()]
                )
                axs[dataset_index][0].title.set_text(
                    pretty_dataset_names[target_dataset]
                )
                axs[dataset_index][1].title.set_text(
                    pretty_dataset_names[target_dataset]
                )
                axs[dataset_index][0].set_xlabel("Acquisition cycle")
                axs[dataset_index][1].set_xlabel("Acquisition cycle")
                axs[dataset_index][0].set_ylabel("Top-K recall")
                axs[dataset_index][1].set_ylabel("Diversity score")

            if main_plot:
                axs[1][1].legend(
                    loc="center right",  # Position of legend
                    bbox_to_anchor=(1.55, 1.15),
                    borderaxespad=0.1,  # Small spacing around legend box
                    title="Acquisition methods",  # Title for the legend
                    fontsize=14,
                )
            else:
                axs[2][1].legend(
                    loc="center right",  # Position of legend
                    bbox_to_anchor=(1.55, 0.5),
                    borderaxespad=0.1,  # Small spacing around legend box
                    title="Acquisition methods",  # Title for the legend
                    fontsize=14,
                )
            fig.tight_layout()
            plt.subplots_adjust(wspace=0.25, hspace=0.3)
            plt.savefig(
                f"Performance_results_feature_set_{feature_set}{fig_suffix}.png"
            )
            plt.clf()
    else:
        for feature_set in feature_sets:
            fig, axs = plt.subplots(1, 3, figsize=(15, 7), constrained_layout=True)
            for m_index, m in enumerate(methods):
                agg_topk_precision_all_seeds = []
                agg_topk_recall_all_seeds = []
                agg_topk_cluster_recall_all_seeds = []
                for seed in seeds:
                    try:
                        data_point = performance_data.loc[
                            (performance_data.feature_set_name == feature_set)
                            & (performance_data.acquisition_function_name == m)
                            & (performance_data.seed == seed)
                        ]
                        num_datasets = len(data_point)
                        for dataset_index in range(num_datasets):
                            agg_topk_precision_all_seeds.append(
                                ast.literal_eval(
                                    data_point["cumulative_precision_topk"].values[
                                        dataset_index
                                    ]
                                )
                            )
                            agg_topk_recall_all_seeds.append(
                                ast.literal_eval(
                                    data_point["cumulative_recall_topk"].values[
                                        dataset_index
                                    ]
                                )
                            )
                            agg_topk_cluster_recall_all_seeds.append(
                                ast.literal_eval(
                                    data_point[
                                        "cumulative_proportion_top_clusters_recovered"
                                    ].values[dataset_index]
                                )
                            )
                    except:
                        pass
                num_seeds_completed = len(agg_topk_precision_all_seeds)
                print("Num seeds: {}".format(num_seeds_completed))
                mean_agg_topk_recall_all_seeds = np.mean(
                    np.array(agg_topk_recall_all_seeds), axis=0
                )
                mean_agg_topk_cluster_recall_all_seeds = np.mean(
                    np.array(agg_topk_cluster_recall_all_seeds), axis=0
                )
                mean_agg_overall_score = np.sqrt(
                    mean_agg_topk_recall_all_seeds
                    * mean_agg_topk_cluster_recall_all_seeds
                )
                stde_agg_topk_recall_all_seeds = (
                    np.std(np.array(agg_topk_recall_all_seeds), axis=0)
                    / (num_seeds_completed) ** 0.5
                )
                stde_agg_topk_cluster_recall_all_seeds = (
                    np.std(np.array(agg_topk_cluster_recall_all_seeds), axis=0)
                    / (num_seeds_completed) ** 0.5
                )
                stde_agg_overall_score = np.sqrt(
                    stde_agg_topk_recall_all_seeds
                    * stde_agg_topk_cluster_recall_all_seeds
                )

                linestyle = "dashed" if m_index >= dash_cutoff else "solid"
                linewidth = 2.5
                axs[0].plot(
                    mean_agg_topk_recall_all_seeds,
                    label=pretty_method_names[m],
                    linestyle=linestyle,
                    linewidth=linewidth,
                )
                axs[1].plot(
                    mean_agg_topk_cluster_recall_all_seeds,
                    label=pretty_method_names[m],
                    linestyle=linestyle,
                    linewidth=linewidth,
                )
                axs[2].plot(
                    mean_agg_overall_score,
                    label=pretty_method_names[m],
                    linestyle=linestyle,
                    linewidth=linewidth,
                )

            axs[0].set_yticklabels(["{:,.0%}".format(x) for x in axs[0].get_yticks()])
            axs[1].set_yticklabels(["{:,.0%}".format(x) for x in axs[1].get_yticks()])
            axs[2].set_yticklabels(["{:,.0%}".format(x) for x in axs[2].get_yticks()])
            axs[0].set_xlabel("Acquisition cycle")
            axs[1].set_xlabel("Acquisition cycle")
            axs[2].set_xlabel("Acquisition cycle")
            axs[0].set_ylabel("Top-K recall")
            axs[1].set_ylabel("Diversity score")
            axs[2].set_ylabel("Overall score")

            handles, labels = axs[1].get_legend_handles_labels()
            fig.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.52, 1.0),
                ncol=len(methods),
                fontsize=13,
                title="Acquisition methods",
            )
            plt.subplots_adjust(top=0.85)
            fig.tight_layout(rect=[0, 0, 1, 0.9])
            plt.savefig(
                f"Avg_performance_results_feature_set_{feature_set}_{fig_suffix}.png"
            )
            plt.clf()


if __name__ == "__main__":
    main()
