# Spotter: An Unlearning Framework Against Over-Unlearning and Prototypical Relearning Attacks
Official Implementation of "**Unlearning’s Blind Spots: Over‑Unlearning and Prototypical Relearning Attack**"



<p align="center">
  <!--img src="https://github.com/Seung-B/FL-OpenPSG/assets/14955366/cdc892e9-9c9c-451c-a86f-53af9a8f81af" align="center" width="95%"-->

  <p align="center">
  <a href="https://arxiv.org/pdf/2506.01318" target='_blank'>
    <img src="https://img.shields.io/badge/Paper-arXiv%2025-b31b1b?style=flat-square">
  </a>

</p>


  <p align="center">
  <font size=5><strong>Benchmarking Federated Learning for Semantic Datasets: Federated Scene Graph Generation</strong></font>
    <br>
      <a href="https://github.com/Seung-B" target='_blank'>SeungBum Ha</a>,&nbsp;
      <a href="https://srompark.github.io/" target='_blank'>Saerom Park</a>,&nbsp;
      <a href="https://sites.google.com/view/swyoon89" target='_blank'>Sung Whan Yoon</a>
    <br>
  Graduate School of Artificial Intelligence, Ulsan National Institute of Science & Technology
  </p>
</p>

## News



## Abstract
Machine unlearning (MU) aims to expunge a designated forget set from a trained model without costly retraining, yet the existing techniques overlook two critical blind spots: “over‑unlearning’’ that deteriorates retained data near the forget set, and post‑hoc “relearning” attacks that aim to resurrect the forgotten knowledge.
We first derive the over-unlearning metric $\text{OU}@\varepsilon$, which represents the collateral damage to the nearby region of the forget set, where the over-unlearning mainly appears.
Next, we expose an unforeseen relearning threat on MU, i.e., the Prototypical Relearning Attack, which exploits the per-class prototype of the forget class with just a few samples, and easily restores the pre-unlearning performance.
To counter both blind spots, we introduce $\texttt{Spotter}$, a plug‑and‑play objective that combines (i) a masked knowledge‑distillation penalty on the nearby region of forget set to suppress $\text{OU}@\varepsilon$, and (ii) an intra‑class dispersion loss that scatters forget-class embeddings, neutralizing prototypical relearning attacks.
On CIFAR-10, as one of validations, $\texttt{Spotter}$ reduces $\text{OU}@\varepsilon$ by below the $0.05\times$ of the baseline, drives forget accuracy to 0\%, preserves accuracy of the retain set within 1\% of difference with the original, and denies the prototype‑attack by keeping the forget set accuracy within <1\%, without accessing retained data.
It confirms that $\texttt{Spotter}$ is a practical remedy of the unlearning’s blind spots.

## Settings



## Citation
```
@article{ha2025_2506.01318,
  title={ Unlearning's Blind Spots: Over-Unlearning and Prototypical Relearning Attack },
  author={ SeungBum Ha and Saerom Park and Sung Whan Yoon },
  journal={arXiv preprint arXiv:2506.01318},
  year={ 2025 }
}
```
