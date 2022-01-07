# PMGT

Implementation of "[Pre-training Graph Transformer with Multimodal Side Information for Recommendation](https://arxiv.org/abs/2010.12284)" (MM'21)

## Experiments

n10: 0.0893
    n20: 0.1138
    r10: 0.1451
    r20: 0.2195

<table>
  <tr>
    <td>Datasets</td>
    <td>Metrics</td>
    <td>NCF (GMF)</td>
    <td>NCF (MLP)</td>
    <td>NCF (NeuMF-end)</td>
    <td>NCF (NeuMF-pre)</td>
    <td>NCF (PMGT)</td>
  </tr>
  <tr>
    <td rowspan="5">VG</td>
  </tr>
  <tr>
    <td>REC-R@10</td>
    <td>0.0720</td>
    <td>0.1390</td>
    <td>0.1451</td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>REC-N@10</td>
    <td>0.0460</td>
    <td>0.0869</td>
    <td>0.0893</td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>REC-R@20</td>
    <td>0.1107</td>
    <td>0.2196</td>
    <td>0.2195</td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>REC-N@20</td>
    <td>0.0583</td>
    <td>0.1130</td>
    <td>0.1138</td>
    <td></td>
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
