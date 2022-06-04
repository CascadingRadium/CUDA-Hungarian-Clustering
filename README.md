# CUDA-Hungarian-Clustering
A GPU-Accelerated Clustering Algorithm that uses the Hungarian method

Written in CUDA and C++

Introduction:
  - Parameterless (Almost) Clustering Algorithm
  - Input is a single csv file and the output will be a file named 'output.csv' which has the full original data + an extra 'label' column that specifies what cluster/group it belongs to.
  - Does not need any prior knowledge of the number of clusters/groups present in the dataset
  - Results similar to ones obtained from Spectral Clustering (but without the requirement of the number of clusters parameter)
  - Combined the work of two research papers:
      - A hierarchical clustering algorithm based on the Hungarian method (https://doi.org/10.1016/j.patrec.2008.04.003)
      - GPU-accelerated Hungarian algorithms for the Linear Assignment Problem (https://doi.org/10.1016/j.parco.2016.05.012)
  - Mainly used to find the number of groups in the dataset with each group being a set of 'similar' rows similar to DBSCAN

Execution instructions:

nvcc Clustering.cu

./a.out [INPUT_FILE] [PARAMETER] [Number of Columns from right to skip/ignore] [Number of Rows from top to skip/ignore]

python3 plot_output.py 

INPUT FILE - Any file(.xlxs .csv) that can be opened in a spreadsheet software like libreoffice calc/MS Excel.

PARAMETER - Discreet value in range [0,8] for most inputs (must be manually tuned) - 7 works for most datasets (Independent of real number of clusters in dataset)

The other two command line arguments are meant to filter out the label column and the column header row respectively before passing on the raw data to the model

Constraints:
- The input file should only have numeric columns (float/ integer)
- The input file should not have any NaN or null values - Dataset cleaning must be done prior
- Parameter tuning can only be possible if a rough estimate of the number of values the label can take is known, otherwise a pure unsupervised clustering without any tuning can be done by just assuming Parameter as 7

Working Example:
nvcc Clustering.cu
./a.out data_banknote_authentication.csv 10 1 1

Output is a file named "output.csv"

