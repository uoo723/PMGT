# PMGT

Implementation of "[Pre-training Graph Transformer with Multimodal Side Information for Recommendation](https://arxiv.org/abs/2010.12284)" (MM'21)

## Experiments

    n10: 0.0992
    n20: 0.1254
    r10: 0.1531
    r20: 0.2340

<table>
  <tr>
    <td>Datasets</td>
    <td>Metrics</td>
    <td>GMF</td>
    <td>MLP</td>
    <td>NeuMF-end</td>
    <td>NeuMF-pre</td>
    <td>NeuMF-PMGT</td>
  </tr>
  <tr>
    <td rowspan="5">VG</td>
  </tr>
  <tr>
    <td>N@10</td>
    <td>0.0532</td>
    <td>0.0989</td>
    <td>0.0987</td>
    <td>0.0992</td>
    <td></td>
  </tr>
  <tr>
    <td>N@20</td>
    <td>0.0684</td>
    <td>0.1256</td>
    <td>0.1251</td>
    <td>0.1254</td>
    <td></td>
  </tr>
  <tr>
    <td>R@10</td>
    <td>0.0820</td>
    <td>0.1523</td>
    <td>0.1524</td>
    <td>0.1531</td>
    <td></td>
  </tr>
  <tr>
    <td>R@20</td>
    <td>0.1291</td>
    <td>0.2277</td>
    <td>0.2332</td>
    <td>0.2340</td>
    <td></td>
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
</table>

## Log

[2022.01.08]  
 - 하이퍼 파라미터 search for NCF (VG 데이터셋)  

[2022.01.07]  
 - NCF Training 구현  
 - VG 데이터셋 NCF Baseline (except NeuMF-pre) 실험

[2022.01.06]  
 - VG 데이터셋 통계  
 - NCF Dataset 구현  
 - NDCG@k 구현  
 - Recall@k 구현

[2022.01.05]  
 - Item Graph 생성  
 - NCF 구현

[2022.01.04]  
 - 이미지 다운로드 ([notebooks/PMGT.ipynb](notebooks/PMGT.ipynb))  
 - Visual (Inception-v4) & Textual (BERT) feature 추출

[2022.01.03]  
 - Amazon Review 데이터셋 다운로드 ([notebooks/PMGT.ipynb](notebooks/PMGT.ipynb))
