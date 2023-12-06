"""
Copyright 2022 Clare Lyle, University of Oxford
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
import pickle as pkl 
import seaborn as sns
import os 
import numpy as np
import pdb
# list directories

def main():
    
    sns.set_style("whitegrid")
    methods = ["random",
        "ucb",
        "topk_bax",
        "levelset_bax",
        "subsetmax_bax",
        "thompson_sampling"]
    datas = ['mog']
    fig, axs = plt.subplots(2, 1, figsize=(6, 8))
    for ds in datas:
        for m in methods:
            d = os.path.join(f'output_{ds}/{m}')
            if m == "random" or m=='thompson_sampling':
                ress = []
                ress_max = []
                for s in range(5):
                    d2 = os.path.join(d, str(s))
                    f = os.path.join(d2, "results.pickle")
                    with open(f, "rb") as fo:
                        x =pkl.load(fo)
                        res = [y['MyRecall'] for y in x]
                        resm = [y['ExpectedMax'] for y in x]
                        ress.append(res)
                        ress_max.append(resm)
                res = np.mean(ress, axis=0)
                axs[0].plot(res, label=m)
                axs[1].plot(np.mean(ress_max, axis=0), label=m)
                axs[0].fill_between(range(len(res)), np.min(ress, axis=0), np.max(ress, axis=0), alpha=0.2)
                axs[1].fill_between(range(len(res)), np.min(ress_max, axis=0), np.max(ress_max, axis=0), alpha=0.2)

                # pdb.set_trace()
                    
            else: 
                f = os.path.join(d, "0/results.pickle")
                with open(f, "rb") as fo:
                    x =pkl.load(fo)
                    res = [y['MyRecall'] for y in x]
                    res_max = [y['ExpectedMax'] for y in x]
                axs[0].plot(res, label=m)
                axs[1].plot(res_max, label=m)
        fig.suptitle("Performance of Active Learning Methods on Mixture of Gaussians")
        # axs[1].set_ylim(0.5, 0.62)
        axs[0].set_xlabel("Active learning cycles")
        axs[1].set_xlabel("Active learning cycles")
        axs[0].set_ylabel("Top k recall")
        axs[1].set_ylabel("$\mathbb{E}_{\eta} \max_{x \in S} h(x; \eta)$")
        plt.legend()
        fig.tight_layout()
        plt.savefig(f"topk_results_{ds}.pdf")
        plt.clf()


# write down
if __name__ == "__main__":
    main()
