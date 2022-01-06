# PMGT

Implementation of "[Pre-training Graph Transformer with Multimodal Side Information for Recommendation](https://arxiv.org/abs/2010.12284)" (MM'21)

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

[2022.01.06]  
    - NCF Dataset     
    - NDCG

[2022.01.05]  
    - Item Graph 생성   
    - NCF 구현

[2022.01.04]  
    - 이미지 다운로드 ([notebooks/PMGT.ipynb](notebooks/PMGT.ipynb))  
    - Visual (Inception-v4) & Textual (BERT) feature 추출

[2022.01.03]  
    - Amazon Review 데이터셋 다운로드 ([notebooks/PMGT.ipynb](notebooks/PMGT.ipynb))
