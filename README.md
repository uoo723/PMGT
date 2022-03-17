# PMGT

Implementation of "[Pre-training Graph Transformer with Multimodal Side Information for Recommendation](https://arxiv.org/abs/2010.12284)" (MM'21)

## Experiments

<table>
  <tr>
    <td rowspan="2">Datasets</td>
    <td rowspan="2">Metrics</td>
    <td colspan="4">Top-N Recommendation</td>
    <td colspan="2">CTR Prediction</td>
  </tr>
  <tr>
    <td>GMF</td>
    <td>MLP</td>
    <td>NeuMF</td>
    <td>NeuMF-PMGT</td>
    <td>DCN</td>
    <td>DCN-PMGT</td>
  </tr>
  <tr>
    <td rowspan="7">VG</td>
  </tr>
  <tr>
    <td>N@10</td>
    <td>0.1426</td>
    <td>0.0972</td>
    <td>0.1621</td>
    <td>0.1810</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>N@20</td>
    <td>0.1602</td>
    <td>0.1209</td>
    <td>0.1815</td>
    <td>0.2067</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>R@10</td>
    <td>0.2057</td>
    <td>0.1724</td>
    <td>0.2365</td>
    <td>0.2748</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>R@20</td>
    <td>0.2687</td>
    <td>0.2592</td>
    <td>0.3060</td>
    <td>0.3661</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>AUC</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>0.8178</td>
    <td>0.8667</td>
  </tr>
  <tr>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td rowspan="6">TG</td>
  </tr>
  <tr>
    <td>N@10</td>
    <td>0.1730</td>
    <td>0.1163</td>
    <td>0.1995</td>
    <td>0.2192</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>N@20</td>
    <td>0.1837</td>
    <td>0.1369</td>
    <td>0.2189</td>
    <td>0.2384</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>R@10</td>
    <td>0.2104</td>
    <td>0.1828</td>
    <td>0.2733</td>
    <td>0.2889</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>R@20</td>
    <td>0.2497</td>
    <td>0.2589</td>
    <td>0.3445</td>
    <td>0.3590</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>AUC</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>0.8387</td>
    <td>0.8486</td>
  </tr>
</table>

## Stats

<table>
  <tr>
    <td rowspan="2" style="text-align:center">Datasets</td>
    <td colspan="3" style="text-align:center">Data for Downstream tasks</td>
    <td colspan="2" style="text-align:center">Item Graph</td>
    <td colspan="2" style="text-align:center">Multimodal Feat.</td>
  </tr>
  <tr>
    <td style="text-align:center"># Users</td>
    <td style="text-align:center" ># Items</td>
    <td style="text-align:center"># Interact.</td>
    <td style="text-align:center"># Nodes</td>
    <td style="text-align:center"># Edges</td>
    <td style="text-align:center"># Visual Feat.</td>
    <td style="text-align:center"># Textual Feat.</td>
  </tr>
  <tr>
    <td style="text-align:center">VG</td>
    <td style="text-align:right">27,988</td>
    <td style="text-align:right">6,551</td>
    <td style="text-align:right">98,278</td>
    <td style="text-align:right">7,252</td>
    <td style="text-align:right">88,606</td>
    <td style="text-align:right">502</td>
    <td style="text-align:right">7,252</td>
  </tr>
  <tr>
    <td style="text-align:center">TG</td>
    <td style="text-align:right">134,697</td>
    <td style="text-align:right">10,337</td>
    <td style="text-align:right">378,138</td>
    <td style="text-align:right">10,834</td>
    <td style="text-align:right">38,252</td>
    <td style="text-align:right">1,279</td>
    <td style="text-align:right">10,834</td>
  </tr>
</table>

## Log

[2022.01.16]  
  - Add TG dataset & experiment

[2022.01.15]  
  - Fixed bug for Validation & test set [diff](https://github.com/uoo723/PMGT/commit/3f55ba1715d9ba74790ed5d2b7bffcf45b50ddb1)  
  - Re-experiment for NeuMF-end & NeuMF-PMGT
  - Implement DCN ([pmgt/dcn/models.py](pmgt/dcn/models.py))
  - Implement DCN training ([pmgt/dcn/trainer.py](pmgt/dcn/trainer.py))

[2022.01.14]  
  - Experiment for NeuMF-PMGT  

[2022.01.13]  
  - Implement PMGT pre-training  

[2022.01.12]  
  - Implement PMGT training ([pmgt/pmgt/trainer.py](pmgt/pmgt/trainer.py))  

[2022.01.11]  
 - Implement Graph structure reconstruction loss ([pmgt/pmgt/models.py](pmgt/pmgt/models.py))  
 - Implement Node feature reconstruction loss  

[2022.01.10]  
 - Implement MCNSampling ([pmgt/pmgt/datasets.py](pmgt/pmgt/datasets.py))  

[2022.01.09]  
 - Implement PMGT ([pmgt/pmgt/modeling_pmgt.py](pmgt/pmgt/modeling_pmgt.py))  

[2022.01.08]  
 - HPO for NCF (VG dataset)  

[2022.01.07]  
 - Implment NCF Training  
 - Experiment on VG dataset over NCF Baseline (except NeuMF-pre)

[2022.01.06]  
 - Report stats for VG dataset  
 - Implement NCF Dataset  
 - Implement NDCG@k  
 - Implement Recall@k 

[2022.01.05]  
 - Implement Item Graph  
 - Implement NCF

[2022.01.04]  
 - Download Images ([notebooks/PMGT.ipynb](notebooks/PMGT.ipynb))  
 - Extract Visual (Inception-v4) & Textual (BERT) features

[2022.01.03]  
 - Download Amazon Review ([notebooks/PMGT.ipynb](notebooks/PMGT.ipynb))
