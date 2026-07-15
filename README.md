# ProTwins: Protein Structural Hierarchy and Functional Annotation Pipeline

**ProTwins** is a command-line tool designed for high-throughput structural comparison, hierarchical clustering, and functional annotation of protein structures (`.pdb` or `.cif`). It integrates the precision of **USalign** for pairwise structural alignments with the speed of **Foldseek** for cloud-based database homology searches.

## Key Features

- **Automated Pairwise Alignment:** Calculates comprehensive structural similarity matrices (TM-scores) using local `USalign` execution.
- **Hierarchical Clustering:** Groups proteins based on structural distance ($1 - \text{TM-score}$) using the average linkage method.
- **Advanced Visualizations:** Automatically generates publication-ready Heatmaps, Clustermaps, and Dendrograms partitioned by user-defined cutting thresholds.
- **Representative Medoid Selection:** Identifies the exact medoid for each structural cluster to capture representative topologies.
- **PyMOL Integration:** Generates customized `.pml` visualization scripts with distance-based color gradients for multi-structure alignment.
- **Functional Annotation:** Connects to the Foldseek API to screen query structures or cluster medoids against global structural databases (PDB, AlphaFold DB, Swissprot, etc.), generating consolidated Excel/CSV reports.

## External Dependency Installation (USalign)

ProTwins relies on a local binary execution of USalign to perform high-throughput structural alignments. Because the script invokes this program via a relative command ("./USalign"), the binary must always be located in the current working directory from which you open the terminal and execute the command.

## Step-by-Step Compilation in the Linux Terminal

Open your Linux terminal and run the following commands to clone and compile the official USalign source code from its repository:

```bash
### 1. Clone the official USalign repository
git clone [https://github.com/pylelab/USalign.git](https://github.com/pylelab/USalign.git)
cd USalign

### 2. Compile the C++ source code using g++
g++ -O3 -ffast-math -lm -o USalign USalign.cpp
```

## Requirements & Installation
```bash
### 1. Clone the Repository
git clone https://github.com/matiaspoblete0701-ui/ProTwins
cd ProTwins

### 2. Install the requiriments 
pip install -r requirements.txt
```

## Usage

ProTwins is executed from the command line. You can display the help message and all available options by running:

```bash
python ProTwins.py --help

usage: ProTwins.py [-h] -r PATH [PATH ...] -o OUTPUT -d DIRECTORY [-u THRESHOLD [THRESHOLD ...]] [-md MAKEDENDROGRAM]
                   [-fs {1,2,3,4}]

options:
  -h, --help            show this help message and exit
  -r PATH [PATH ...], --path PATH [PATH ...]
                        Path(s) to folders containing .pdb or .cif files
  -o OUTPUT, --output OUTPUT
                        Prefix for generated files (Required)
  -d DIRECTORY, --directory DIRECTORY
                        Output directory (Required)
  -u THRESHOLD [THRESHOLD ...], --threshold THRESHOLD [THRESHOLD ...]
                        One or more distance thresholds for clustering. (Defaults to 0.2 and 0.5 in complete mode if
                        none specified).
  -md MAKEDENDROGRAM, --makedendrogram MAKEDENDROGRAM
                        Fast mode: Receives the path to a precomputed distance matrix (*_distance.csv) to skip
                        USalign.
  -fs {1,2,3,4}, --foldseek {1,2,3,4}
                        4 options:
                        1: TM-score calculation and medoid search in Foldseek.
                        2: TM-score calculation and search of all proteins in Foldseek.
                        3: No TM-score calculation, medoid search (requires a distance matrix and must be used in conjunction with -md).
                        4: Search of all proteins in Foldseek without calculating TM-score.
```

### Output Directory Structure

markdown
## Output Directory Structure

The specified output directory (`-d/--directory`) will be structured as follows:
```
example of use: 

python ProTwins.py -r ./my_proteins/ -o [prefix] -d ./output_directory/ -u [threshold -> (0.5 0.4)] -fs 1

output_directory/
│
├── [prefix]_similarity.csv          # Pairwise structural similarity matrix (TM-scores)
├── [prefix]_distance.csv            # Structural distance matrix (1 - TM-score)
├── [prefix]_tree.nwk                 # Phylogenetic/structural tree in Newick format
├── [prefix]_similarity.pdf          # Pairwise similarity heatmap
├── [prefix]_distance.pdf            # Pairwise distance heatmap
├── [prefix]_clustermap_final.pdf    # Dendrogram-coupled heatmap (Seaborn clustermap)
├── [prefix]_[threshold]_dendrogram.pdf # Cutoff dendrogram showing colored clusters
│
├── foldseek_results/                 # Created if Foldseek analysis (-fs) is activated
│   └── [prefix]_Foldseek_Report.xlsx # Consolidated Excel workbook with hits, taxonomy, E-values, and source database links
│
└── pymol_scripts/                    # Automated 3D visualization scripts for PyMOL
    ├── [threshold]/                  # Individual .pml scripts to visualize and align each separate cluster
    └── global_medoids/               # Core .pml script to align all global medoids together
```

## Authors and Citation

Developed as a structural bioinformatics utility for protein analysis. 

If you use ProTwins in your research, please cite this repository and the underlying tools:
- **USalign**: Zhang, J., et al. (2022). US-align: universal structure alignment of proteins, nucleic acids and macromolecular complexes. *Nature Methods*.
- **Foldseek**: van Kempen, M., et al. (2023). Fast and accurate protein structure search with Foldseek. *Nature Biotechnology*.


