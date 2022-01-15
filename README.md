# PMGT

Implementation of "[Pre-training Graph Transformer with Multimodal Side Information for Recommendation](https://arxiv.org/abs/2010.12284)" (MM'21)

## Experiments

- 버그 수정 후 재실험 결과

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
    <td rowspan="8">VG</td>
  </tr>
  <tr>
    <td>N@10</td>
    <td>0.1426</td>
    <td>0.0972</td>
    <td>0.1621</td>
    <td>-</td>
    <td>0.1810</td>
  </tr>
  <tr>
    <td>N@20</td>
    <td>0.1602</td>
    <td>0.1209</td>
    <td>0.1815</td>
    <td>-</td>
    <td>0.2067</td>
  </tr>
  <tr>
    <td>R@10</td>
    <td>0.2057</td>
    <td>0.1724</td>
    <td>0.2365</td>
    <td>-</td>
    <td>0.2748</td>
  </tr>
  <tr>
    <td>R@20</td>
    <td>0.2687</td>
    <td>0.2592</td>
    <td>0.3060</td>
    <td>-</td>
    <td>0.3661</td>
  </tr>
  <tr>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>Metrics</td>
    <td>DCN</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>DCN-PMGT</td>
  </tr>
  <tr>
    <td>AUC</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
</table>

- Deprecated (Validation & test set 버그)

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
    <td>0.0991</td>
  </tr>
  <tr>
    <td>N@20</td>
    <td>0.0684</td>
    <td>0.1256</td>
    <td>0.1251</td>
    <td>0.1254</td>
    <td>0.1258</td>
  </tr>
  <tr>
    <td>R@10</td>
    <td>0.0820</td>
    <td>0.1523</td>
    <td>0.1524</td>
    <td>0.1531</td>
    <td>0.1529</td>
  </tr>
  <tr>
    <td>R@20</td>
    <td>0.1291</td>
    <td>0.2277</td>
    <td>0.2332</td>
    <td>0.2340</td>
    <td>0.2354</td>
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

[2022.01.15]  
  - Validation & test set 버그 수정 [diff](https://github.com/uoo723/PMGT/commit/3f55ba1715d9ba74790ed5d2b7bffcf45b50ddb1)  
  - NeuMF-end & NeuMF-PMGT 재실험
  - DCN 모델 구현 ([pmgt/dcn/models.py](pmgt/dcn/models.py))
  - DCN training 구현 ([pmgt/dcn/trainer.py](pmgt/dcn/trainer.py))

[2022.01.14]  
  - NeuMF-PMGT 실험 진행  

[2022.01.13]  
  - PMGT pre-training 진행  

[2022.01.12]  
  - PMGT training 구현 ([pmgt/pmgt/trainer.py](pmgt/pmgt/trainer.py))  

[2022.01.11]  
 - Graph structure reconstruction loss 구현 ([pmgt/pmgt/models.py](pmgt/pmgt/models.py))  
 - Node feature reconstruction loss 구현  

[2022.01.10]  
 - MCNSampling 구현 ([pmgt/pmgt/datasets.py](pmgt/pmgt/datasets.py))  

[2022.01.09]  
 - PMGT 모델 구현 ([pmgt/pmgt/modeling_pmgt.py](pmgt/pmgt/modeling_pmgt.py))  

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
