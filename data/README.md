# Data sources

Raw datasets are downloaded locally into `data/raw/` and are not committed.

| Dataset | Paper shape | Source used by this repository | Notes |
|---|---:|---|---|
| MNIST | 70,000 x 784 | https://yann.lecun.com/exdb/mnist/ (download mirror in script) | Automatically downloaded by `scripts/download_data.py`. |
| Extended Yale B | 2,432 x 1,024 in the paper | http://www.cad.zju.edu.cn/home/dengcai/Data/FaceData.html | The paper says 32x32 crops; public packages vary between 2,414 and 2,432 images, so preprocessing must be recorded. |
| RSSCN7 | 2,800 images; 2,048-D ResNet features | https://github.com/palewithout/RSSCN7 | Redistribution terms are not explicit; download manually and retain source attribution. |
| Cora/Citeseer/Pubmed | Planetoid citation networks | https://github.com/kimiyoung/planetoid/tree/master/data | Use the standard Planetoid splits referenced by the paper. |

The independent pilot in this repository downloads MNIST. The other links are
provided for the full extension because the source paper does not release its
exact processed arrays, random indices, or ResNet checkpoint/features.
