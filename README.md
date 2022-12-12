[![GitHub license](https://img.shields.io/github/license/CascadingRadium/CUDA-Hungarian-Clustering)](https://github.com/CascadingRadium/CUDA-Hungarian-Clustering/blob/main/LICENCE)
[![GitHub forks](https://img.shields.io/github/forks/CascadingRadium/CUDA-Hungarian-Clustering)](https://github.com/CascadingRadium/CUDA-Hungarian-Clustering/network)
[![GitHub stars](https://img.shields.io/github/stars/CascadingRadium/CUDA-Hungarian-Clustering)](https://github.com/CascadingRadium/CUDA-Hungarian-Clustering/stargazers)
[![GitHub issues](https://img.shields.io/github/issues/CascadingRadium/CUDA-Hungarian-Clustering)](https://github.com/CascadingRadium/CUDA-Hungarian-Clustering/issues)
![GitHub repo size](https://img.shields.io/github/repo-size/CascadingRadium/CUDA-Hungarian-Clustering)
![GitHub last commit](https://img.shields.io/github/last-commit/CascadingRadium/CUDA-Hungarian-Clustering)
<img src="https://developer.nvidia.com/favicon.ico" align ='right' width ='50'>
# CUDA-Hungarian-Clustering
A GPU-Accelerated Clustering Algorithm that uses the Hungarian method

Written in CUDA and C++

Introduction:
  - Parameterless (Almost) Clustering Algorithm
  - Input is a single CSV file and the output will be a file named 'output.csv' which has the full original data + an extra 'label' column that specifies what cluster/group it belongs to.
  - Does not need any prior knowledge of the number of clusters/groups present in the dataset
  - Results similar to ones obtained from Spectral Clustering (but without the requirement of the number of clusters parameter)
  - Combined the work of two research papers:
      - A hierarchical clustering algorithm based on the Hungarian method, <i>Journal of Pattern Recognition Letters</i> (2008) (https://doi.org/10.1016/j.patrec.2008.04.003)
      - GPU-accelerated Hungarian algorithms for the Linear Assignment Problem, <i>Journal of Parallel Computing</i> (2016)  (https://doi.org/10.1016/j.parco.2016.05.012)
  - Mainly used to find the number of groups in the dataset with each group being a set of 'similar' rows similar to DBSCAN

Execution instructions:

```
nvcc Clustering.cu

./a.out [INPUT_FILE] [PARAMETER] [Number of Columns from right to skip/ignore] [Number of Rows from top to skip/ignore]

python3 plot_output.py 

```

INPUT FILE - Any file(.xlxs .csv) that can be opened in spreadsheet software like LibreOffice calc/MS Excel.

PARAMETER - Integral value in the range [0,8] for most inputs (must be manually tuned) - 7 works for most datasets (Independent of the real number of clusters in the dataset)

The other two command-line arguments are meant to filter out the label column and the column header row respectively before passing on the raw data to the model

Constraints:
- The input file should only have numeric columns (float/ integer)
- The input file should not have any NaN or null values - Dataset cleaning must be done prior
- Parameter tuning can only be possible if a rough estimate of the number of values the label can take is known, otherwise, a pure unsupervised clustering without any tuning can be done by just assuming Parameter as 7
- Sensitive to noise
- Parameter, being fully independent of the dataset, cannot be estimated and is mostly tuned based on trial-and-error, but almost always takes a value in the range [0,10] 

Working Example:

```
nvcc Clustering.cu
./a.out data_banknote_authentication.csv 10 1 1
```
This will now use parameter 10 and cluster the input .csv file into some number of groups and output a file named 'output.csv' which has an additional column called label which represents the groupID or the group to which it belongs.

Sample output images - using datasets in the TestedDataset directory:

<p float="left">
  <img src="TestedDatasets/data0.png" width="32%" title="data0.csv" alt="data0">
  <img src="TestedDatasets/data1.png" width="32%" title="data1.csv" alt="data1">
  <img src="TestedDatasets/data2.png" width="32%" title="data2.csv" alt="data2">
</p>
<p float="left">
  <img src="TestedDatasets/data3.png" width="32%" title="data3.csv" alt="data3">
  <img src="TestedDatasets/data4.png" width="32%" title="data4.csv" alt="data4">
  <img src="TestedDatasets/data5.png" width="32%" title="data5.csv" alt="data5">
</p>
<p float="left">
  <img src="TestedDatasets/data6.png" width="32%" title="data6.csv" alt="data6">
  <img src="TestedDatasets/data7.png" width="32%" title="data7.csv" alt="data7">
  <img src="TestedDatasets/data8.png" width="32%" title="data8.csv" alt="data8">
</p>
<p float="left">
  <img src="TestedDatasets/data9.png" width="32%" title="data9.csv" alt="data9">
  <img src="TestedDatasets/data10.png" width="32%" title="data10.csv" alt="data10">
  <img src="TestedDatasets/data11.png" width="32%" title="data11.csv" alt="data11">
</p>
<p float="left">
  <img src="TestedDatasets/data12.png" width="32%" title="data12.csv" alt="data12">
  <img src="TestedDatasets/data13.png" width="32%" title="data13.csv" alt="data13">
  <img src="TestedDatasets/data14.png" width="32%" title="data14.csv" alt="data14">
</p>
<p float="left">
  <img src="TestedDatasets/data15.png" width="32%" title="data15.csv" alt="data15">
  <img src="TestedDatasets/data16.png" width="32%" title="data16.csv" alt="data16">
  <img src="TestedDatasets/data17.png" width="32%" title="data17.csv" alt="data17">
</p>
<p float="left">
  <img src="TestedDatasets/data18.png" width="32%" title="data18.csv" alt="data18">
  <img src="TestedDatasets/data19.png" width="32%" title="data19.csv" alt="data19">
  <img src="TestedDatasets/data20.png" width="32%" title="data20.csv" alt="data20">
</p>
