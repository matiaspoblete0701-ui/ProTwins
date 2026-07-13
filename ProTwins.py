import glob
import subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, to_tree
import os
import time
import argparse
import pandas as pd
import gc
import requests
import json
import tarfile
import shutil
import re
from tqdm import tqdm

def get_tm_score(pdb1, pdb2):
    start_time = time.perf_counter()
    process = subprocess.run(["./USalign", pdb1, pdb2, "-mm", "1", "-ter", "0"], capture_output=True, text=True)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    score1, score2 = 0.0, 0.0
    for line in process.stdout.split('\n'):
        if line.startswith("TM-score=") and "Structure_1" in line:
            score1 = float(line.split()[1])
        if line.startswith("TM-score=") and "Structure_2" in line:
            score2 = float(line.split()[1])
            return score1, score2, elapsed_time
    return score1, score2, elapsed_time

def define_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--path", nargs='+', required=True,
                        help="Path(s) to folders containing .pdb or .cif files")
    parser.add_argument("-o", "--output", type=str, required=True, 
                        help="Prefix for generated files (Required)")
    parser.add_argument("-d", "--directory", type=str, required=True, 
                        help="Output directory (Required)")
    parser.add_argument("-u", "--threshold", type=float, nargs='+', default=[],
                        help="One or more distance thresholds for clustering. (Defaults to 0.2 and 0.5 in complete mode if none specified).")
    parser.add_argument("-md", "--makedendrogram", type=str, default=None,
                        help="Fast mode: Receives the path to a precomputed distance matrix (*_distance.csv) to skip USalign.")
    parser.add_argument("-fs", "--foldseek", type=int, default=None, choices=[1, 2, 3, 4],
                        help="4 options, 1: TM-score calculation and medoid search in Foldseek, 2: TM-score calculation and search of all proteins in Foldseek, " \
                             "3: No TM-score calculation, medoid search (requires a distance matrix and must be used in conjunction with -md), 4: Search of all proteins without calculating TM-score")
    return parser.parse_args()

def execute_threshold_analysis(linkage_matrix, dist_matrix, labels, threshold, mode_name, args, protein_files):
    cluster_labels = fcluster(linkage_matrix, threshold, criterion='distance')
    
    df_res = pd.DataFrame({"Protein": labels, "Cluster": cluster_labels})
    df_res['Is_Medoid'] = False
    graph_name_mapping = {prot: prot for prot in labels}
    medoids_by_cluster = {}

    for cluster_id in np.unique(cluster_labels):
        cluster_prots = df_res[df_res['Cluster'] == cluster_id]['Protein'].tolist()
        medoid = find_medoid(cluster_prots, dist_matrix, labels)
        df_res.loc[df_res['Protein'] == medoid, 'Is_Medoid'] = True
        
        if len(cluster_prots) >= 2:
            graph_name_mapping[medoid] = f"*** {medoid}"
        else:
            graph_name_mapping[medoid] = medoid
            
        medoids_by_cluster[cluster_id] = medoid

    n_prot = len(labels)
    final_names = [graph_name_mapping[e] for e in labels]
    max_char = max(len(n) for n in final_names)
    
    base_width = 14 + (max_char * 0.15)
    fig_height = max(10, n_prot * 0.3) 
    leaf_font_size = max(4, min(10, 600 / n_prot)) 
    
    fig, ax = plt.subplots(figsize=(base_width, fig_height))
    plt.subplots_adjust(left=0.4) 

    ddata = dendrogram(
        linkage_matrix, 
        labels=final_names, 
        orientation='right', 
        color_threshold=threshold, 
        above_threshold_color='grey',
        leaf_font_size=leaf_font_size,
        ax=ax
    )

    transform = ax.get_yaxis_transform() 
    y_coords = {leaf: i * 10 + 5 for i, leaf in enumerate(ddata['ivl'])}
    
    x_cluster_id = -0.35    
    x_bracket_back = -0.22  
    x_bracket_front = -0.18 

    cluster_ordering = []
    for cluster_id in np.unique(cluster_labels):
        cluster_prots = df_res[df_res['Cluster'] == cluster_id]['Protein'].tolist()
        y_vals = [y_coords[graph_name_mapping[p]] for p in cluster_prots if graph_name_mapping[p] in y_coords]
        
        if len(y_vals) >= 2:
            cluster_ordering.append((cluster_id, max(y_vals), y_vals))

    cluster_ordering.sort(key=lambda x: x[1], reverse=True)
    real_clusters_count = len(cluster_ordering)

    for new_id, (old_id, _, y_vals) in enumerate(cluster_ordering, 1):
        y_min, y_max = min(y_vals), max(y_vals)
        y_mid = (y_min + y_max) / 2
        
        ax.text(x_cluster_id, y_mid, f"C{new_id}", transform=transform, 
                va='center', ha='left', fontsize=max(8, leaf_font_size), fontweight='bold', clip_on=False)
        
        ax.plot([x_bracket_front, x_bracket_back, x_bracket_back, x_bracket_front], 
                [y_min, y_min, y_max, y_max], 
                color='black', transform=transform, lw=2.0, clip_on=False)

    plt.axvline(x=threshold, color='r', linestyle='--', label=f'Cutoff Threshold ({threshold:.2f}) | Total Clusters (K>=2): {real_clusters_count}')
    plt.legend(loc='upper right', frameon=True, shadow=True)
    ax.set_title(f"Similarity Dendrogram - {args.output}", fontsize=16, pad=30)
    ax.set_xlabel("Structural Distance (1 - TM-score)", fontsize=12)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=leaf_font_size)

    pdf_path = os.path.join(args.directory, f"{args.output}_{mode_name}_dendrogram.pdf")
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=150)
    plt.close('all')
    gc.collect()

    print(f"Dendrogram '{mode_name}' generated.")
    
    upload_dir = os.path.join(args.directory, "pymol_scripts", mode_name)
    os.makedirs(upload_dir, exist_ok=True)
    
    paths_dict = {}
    for f in protein_files:
        name = os.path.basename(f).split('.')[0]
        paths_dict[name] = os.path.relpath(f, start=upload_dir)

    valid_medoids = []

    for new_id, (old_id, _, _) in enumerate(cluster_ordering, 1):
        medoid = medoids_by_cluster[old_id]
        cluster_prots = df_res[df_res['Cluster'] == old_id]['Protein'].tolist()
        
        valid_medoids.append(medoid)
        pml_path = os.path.join(upload_dir, f"cluster_{new_id}.pml")
        
        try:
            with open(pml_path, "w") as f:
                f.write(f"reinitialize\n\n")
                f.write("python\n")
                f.write("import os\n")
                f.write("_self_dir = os.path.dirname(cmd.get_script_path())\n")
                f.write("if _self_dir: os.chdir(_self_dir)\n")
                f.write("python end\n\n")
                
                ref_idx = labels.index(medoid)
                
                members_with_distance = []
                for prot in cluster_prots:
                    if prot != medoid and prot in paths_dict:
                        idx_p = labels.index(prot)
                        dist = dist_matrix[idx_p][ref_idx]
                        members_with_distance.append((prot, dist))
                
                members_with_distance.sort(key=lambda x: x[1])
                
                local_distances = [d for _, d in members_with_distance]
                min_local_dist = min(local_distances) if local_distances else 0.0
                max_local_dist = max(local_distances) if local_distances else 1.0
                
                f.write(f"load {paths_dict[medoid]}, {medoid}\n")
                f.write(f"color purple, {medoid}\n\n")
                
                for prot, dist in members_with_distance:
                    if max_local_dist != min_local_dist:
                        norm = (dist - min_local_dist) / (max_local_dist - min_local_dist)
                    else:
                        norm = 0.0
                        
                    r = 1.0
                    g = norm
                    b = 0.0
                    
                    color_name = f"color_dist_{prot}"
                    f.write(f"set_color {color_name}, [{r:.3f}, {g:.3f}, {b:.3f}]\n")
                    f.write(f"load {paths_dict[prot]}, {prot}\n")
                    f.write(f"color {color_name}, {prot}\n")
                    f.write(f"align {prot}, {medoid}\n")
                
                f.write("\nshow cartoon\n")
                f.write("orient\n")
        except Exception:
            pass

    if valid_medoids:
        medoids_dir = os.path.join(args.directory, "pymol_scripts", "global_medoids")
        os.makedirs(medoids_dir, exist_ok=True)
        master_path = os.path.join(medoids_dir, f"{threshold}_medoids.pml")
        
        master_paths_dict = {}
        for f in protein_files:
            name = os.path.basename(f).split('.')[0]
            master_paths_dict[name] = os.path.relpath(f, start=medoids_dir)

        try:
            with open(master_path, "w") as f:
                f.write("reinitialize\n\n")
                f.write("python\n")
                f.write("import os\n")
                f.write("_self_dir = os.path.dirname(cmd.get_script_path())\n")
                f.write("if _self_dir: os.chdir(_self_dir)\n")
                f.write("python end\n\n")
                
                ref_medoid = valid_medoids[0]
                ref_idx = labels.index(ref_medoid)
                
                distances = []
                for m in valid_medoids[1:]:
                    idx_m = labels.index(m)
                    distances.append(dist_matrix[idx_m][ref_idx])
                    
                min_dist = min(distances) if distances else 0.0
                max_dist = max(distances) if distances else 1.0

                f.write("set_color color_fixed_medoid, [0.6, 0.0, 0.8]\n")
                f.write(f"load {master_paths_dict[ref_medoid]}, {ref_medoid}\n")
                f.write(f"color color_fixed_medoid, {ref_medoid}\n\n")
                
                for m in valid_medoids[1:]:
                    idx_m = labels.index(m)
                    dist = dist_matrix[idx_m][ref_idx]
                    
                    if max_dist != min_dist:
                        norm = (dist - min_dist) / (max_dist - min_dist)
                    else:
                        norm = 0.0
                        
                    r = 1.0
                    g = norm
                    b = 0.0
                    
                    color_name = f"color_dist_{m}"
                    f.write(f"set_color {color_name}, [{r:.3f}, {g:.3f}, {b:.3f}]\n")
                    f.write(f"load {master_paths_dict[m]}, {m}\n")
                    f.write(f"color {color_name}, {m}\n")
                    f.write(f"align {m}, {ref_medoid}\n")
                
                f.write("\nshow cartoon\n")
                f.write("orient\n")
                
        except Exception:
            pass

    gc.collect()
    return valid_medoids

def build_newick(node, newick, parentdist, leaf_names):
    if node.is_leaf():
        return f"{leaf_names[node.id]}:{(parentdist - node.dist):.6f}{newick}"
    else:
        if len(newick) > 0:
            newick = f":{(parentdist - nodo.dist):.6f}{newick}"
        newick = f"({build_newick(node.left, '', node.dist, leaf_names)},{build_newick(node.right, '', node.dist, leaf_names)}){newick}"
        return newick

def save_newick(linkage_matrix, labels, args):
    tree = to_tree(linkage_matrix, rd=False)
    newick_string = build_newick(tree, "", tree.dist, labels) + ";"
    newick_path = os.path.join(args.directory, f"{args.output}_tree.nwk")
    with open(newick_path, "w") as f:
        f.write(newick_string)

def generate_heat_maps(sim_matrix, dist_matrix, labels, args):
    tasks = [
        (sim_matrix, "similarity", "coolwarm", 0, 1),
        (dist_matrix, "distance", "viridis", 0, 1)
    ]
    
    n_prot = len(labels)
    fig_side = min(50, max(12, n_prot * 0.35)) 
    hm_font_size = max(3, min(10, 400 / n_prot))
    show_labels = n_prot <= 150
    
    for matrix, base_name, color_map, v_min, v_max in tasks:
        plt.figure(figsize=(fig_side, fig_side)) 
        ax = sns.heatmap(matrix, xticklabels=labels if show_labels else False, 
                         yticklabels=labels if show_labels else False, cmap=color_map, 
                         vmin=v_min, vmax=v_max, annot=False, rasterized=True, cbar_kws={"shrink": 0.75})      
        if show_labels:
            ax.tick_params(axis='x', labelsize=hm_font_size)
            ax.tick_params(axis='y', labelsize=hm_font_size)
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        plt.tight_layout()
        final_name = os.path.join(args.directory, f"{args.output}_{base_name}.pdf")
        plt.savefig(final_name, format='pdf', bbox_inches='tight', dpi=150)
        plt.close() 
        gc.collect()

def generate_clustermap(sim_matrix, linkage_matrix, labels, args):
    n_prot = len(labels)
    fig_side = min(50, max(14, n_prot * 0.3))
    cm_font_size = max(4, min(10, 400 / n_prot))
    show_labels = n_prot <= 150
    
    g = sns.clustermap(sim_matrix, row_linkage=linkage_matrix, col_linkage=linkage_matrix, xticklabels=labels if show_labels else False,
                       yticklabels=labels if show_labels else False, cmap="YlGnBu", linewidths=0,
                       rasterized=True, figsize=(fig_side, fig_side), cbar_pos=(0.02, 0.8, 0.05, 0.18))

    if show_labels:
        g.ax_heatmap.tick_params(axis='x', labelsize=cm_font_size)
        g.ax_heatmap.tick_params(axis='y', labelsize=cm_font_size)
        
    output_path = os.path.join(args.directory, f"{args.output}_clustermap_final.pdf")
    g.savefig(output_path, format='pdf', bbox_inches='tight', dpi=150)
    plt.close()
    gc.collect()

def find_medoid(cluster_proteins, dist_matrix, labels):
    if len(cluster_proteins) == 1:
        return cluster_proteins[0]
    indices = [labels.index(p) for p in cluster_proteins]
    submatrix = dist_matrix[np.ix_(indices, indices)]
    local_medoid_idx = np.argmin(submatrix.sum(axis=1))
    return cluster_proteins[local_medoid_idx]

def save_matrices_csv(sim_matrix, dist_matrix, labels, args):
    df_sim = pd.DataFrame(sim_matrix, index=labels, columns=labels)
    df_dist = pd.DataFrame(dist_matrix, index=labels, columns=labels)
    df_sim.to_csv(os.path.join(args.directory, f"{args.output}_similarity.csv"))
    df_dist.to_csv(os.path.join(args.directory, f"{args.output}_distance.csv"))

def generate_db_link(target):
    target_str = str(target).strip()
    if target_str.startswith("AF-"):
        match = re.search(r"AF-([A-Z0-9]{6,10})", target_str)
        if match:
            return f"https://alphafold.ebi.ac.uk/entry/{match.group(1)}"
    match_pdb = re.search(r"\b([0-9][A-Z0-9]{3})\b", target_str.upper())
    if match_pdb:
        return f"https://www.rcsb.org/structure/{match_pdb.group(1)}"
    return "N/A"

def query_and_download_foldseek(pdb_path, download_dir):
    ticket_url = "https://search.foldseek.com/api/ticket"
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    try:
        with open(pdb_path, "rb") as f:
            files = {"q": f}
            data = [
                ("mode", "3diaa"), 
                ("database[]", "bfmd"),
                ("database[]", "gmgcl_id"),
                ("database[]", "mgnify_esm30"),
                ("database[]", "BFVD"),
                ("database[]", "afdb-proteome"),
                ("database[]", "afdb-swissprot"),
                ("database[]", "afdb50"),
                ("database[]", "pdb100"),
                ("database[]", "cath50"),
            ]
            resp = requests.post(ticket_url, files=files, data=data, headers=headers)
            
        if resp.status_code != 200:
            return None
            
        ticket_id = resp.json().get("id")
        if not ticket_id: return None
        
        time.sleep(3)
        status = "PENDING"
        
        while status in ["PENDING", "RUNNING"]:
            time.sleep(4) 
            status_resp = requests.get(f"https://search.foldseek.com/api/ticket/{ticket_id}", headers=headers)
            
            if status_resp.status_code == 200:
                status = status_resp.json().get("status", "ERROR")
            else:
                break
                
        if status == "COMPLETE":
            download_url = f"https://search.foldseek.com/api/result/download/{ticket_id}"
            res_download = requests.get(download_url, headers=headers, stream=True)
            if res_download.status_code == 200:
                base_name = os.path.basename(pdb_path).split('.')[0]
                tar_path = os.path.join(download_dir, f"{base_name}_foldseek_results.tar.gz")
                with open(tar_path, 'wb') as file:
                    for chunk in res_download.iter_content(chunk_size=8192):
                        file.write(chunk)
                return tar_path
    except Exception:
        pass
    return None

def process_foldseek_results(paths_to_search, args):
    foldseek_dir = os.path.join(args.directory, "foldseek_results")
    downloads_dir = os.path.join(foldseek_dir, "downloads")
    os.makedirs(downloads_dir, exist_ok=True)
    
    excel_data = []
    
    for pdb_path in tqdm(paths_to_search, desc="Processing Foldseek queries"):
        query_name = os.path.basename(pdb_path).split('.')[0]
        tar_path = query_and_download_foldseek(pdb_path, downloads_dir)
        
        if tar_path:
            temp_dir = os.path.join(downloads_dir, f"temp_{query_name}")
            os.makedirs(temp_dir, exist_ok=True)
            
            try:
                with tarfile.open(tar_path, "r:gz") as tar:
                    tar.extractall(path=temp_dir)
                    
                m8_files = glob.glob(os.path.join(temp_dir, "*.m8"))
                
                for m8_file in m8_files:
                    if "_report" in m8_file:
                        continue
                    
                    if os.path.getsize(m8_file) == 0:
                        continue
                        
                    db_name = os.path.basename(m8_file).replace("alis_", "").replace(".m8", "")
                    
                    try:
                        df = pd.read_csv(m8_file, sep='\t', header=None, on_bad_lines='skip')
                        
                        for _, row in df.iterrows():
                            if len(row) < 20:
                                continue
                                
                            try:
                                tm_score = float(row[10])
                                e_val = float(row[11])
                            except ValueError:
                                continue
                                
                            target_full = str(row[1])
                            target_clean = target_full.split(' ')[0]
                            
                            organism = str(row[20]).strip() if pd.notnull(row[20]) else "N/A"
                            if organism == "" or str(row[20]).lower() == "nan":
                                organism = "N/A"
                            
                            if tm_score >= 0.50:
                                excel_data.append({
                                    "Query Protein (Lab)": query_name,
                                    "Hit Protein": target_clean,
                                    "Organism Species": organism,
                                    "Database": db_name,
                                    "TM-score": round(tm_score, 4),
                                    "E-value": f"{e_val:.2e}",
                                    "Structural Link (Click)": generate_db_link(target_clean)
                                })
                    except Exception:
                        continue
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)
        else:
            excel_data.append({
                "Query Protein (Lab)": query_name,
                "Hit Protein": "Error",
                "Organism Species": "Failed",
                "Database": "N/A",
                "TM-score": 0.0,
                "E-value": "N/A",
                "Structural Link (Click)": "N/A"
            })
            
    if excel_data:
        df_results = pd.DataFrame(excel_data)
        
        df_valid = df_results[df_results["Database"] != "N/A"]
        df_errors = df_results[df_results["Database"] == "N/A"]
        
        if not df_valid.empty:
            df_valid = df_valid.sort_values(
                by=["Query Protein (Lab)", "Database", "TM-score"], 
                ascending=[True, True, False]
            )
            df_valid = df_valid.drop_duplicates(
                subset=["Query Protein (Lab)", "Database"], 
                keep="first"
            )
            
        df_results = pd.concat([df_valid, df_errors], ignore_index=True)
        df_results = df_results.sort_values(by=["Query Protein (Lab)", "Database"])
        
        excel_path = os.path.join(foldseek_dir, f"{args.output}_Foldseek_Report.xlsx")
        try:
            df_results.to_excel(excel_path, index=False)
            print(f"Report generated: {excel_path}")
        except ImportError:
            csv_path = excel_path.replace(".xlsx", ".csv")
            df_results.to_csv(csv_path, index=False)
            print(f"Report generated: {csv_path}")

def main():
    args = define_arguments() 
    os.makedirs(args.directory, exist_ok=True)

    available_files = {}
    for folder in args.path:
        if os.path.isdir(folder):
            extensions = ["*.pdb", "*.cif", "*.cif.gz", "*.pdb.gz"]
            for ext in extensions:
                for f in glob.glob(os.path.join(folder, ext)):
                    prot_id = os.path.basename(f).split('.')[0]
                    available_files[prot_id] = f

    if args.foldseek == 4:
        protein_files = list(available_files.values())
        if not protein_files:
            return
        process_foldseek_results(protein_files, args)
        return

    if args.makedendrogram:
        if not os.path.exists(args.makedendrogram):
            return
        
        df_dist_loaded = pd.read_csv(args.makedendrogram, index_col=0)
        labels = df_dist_loaded.index.tolist()
        dist_matrix = df_dist_loaded.to_numpy()
        sim_matrix_s = 1 - dist_matrix
        np.fill_diagonal(dist_matrix, 0)
        
        protein_files = []
        for e in labels:
            if e in available_files:
                protein_files.append(available_files[e])
            else:
                return
    else:
        protein_files = sorted(list(set(available_files.values())))
        n = len(protein_files)
        
        if n < 2:
            return 
        
        sim_matrix = np.ones((n, n)) 
        total_comparisons = (n * (n - 1)) // 2
        
        with tqdm(total=total_comparisons, desc="Calculating TM-scores (USalign)") as pbar:
            for i in range(n):
                for j in range(i+1, n):
                    s1, s2, elapsed = get_tm_score(protein_files[i], protein_files[j])
                    sim_matrix[i][j], sim_matrix[j][i] = s1, s2
                    pbar.update(1)
        
        sim_matrix_s = (sim_matrix + sim_matrix.T) / 2 
        dist_matrix = 1 - sim_matrix_s
        np.fill_diagonal(dist_matrix, 0) 
        labels = [os.path.basename(p).split('.')[0] for p in protein_files]

        save_matrices_csv(sim_matrix_s, dist_matrix, labels, args)

    cond_dist = scipy.spatial.distance.squareform(dist_matrix)
    linkage_matrix = linkage(cond_dist, method="average")

    entered_thresholds = args.threshold if args.threshold else []
    
    if args.makedendrogram:
        if not entered_thresholds:
            return
        unique_thresholds = sorted(list(set(entered_thresholds)))
    else:
        unique_thresholds = sorted(list(set(entered_thresholds if entered_thresholds else [0.2, 0.5])))
    
    total_unique_medoids = set()

    for idx, u in enumerate(unique_thresholds, 1):
        medoids_from_threshold = execute_threshold_analysis(linkage_matrix, dist_matrix, labels, u, str(u), args, protein_files)
        total_unique_medoids.update(medoids_from_threshold)

    if not args.makedendrogram:
        generate_heat_maps(sim_matrix_s, dist_matrix, labels, args)
        save_newick(linkage_matrix, labels, args)
        generate_clustermap(sim_matrix_s, linkage_matrix, labels, args)

    if args.foldseek:
        if args.foldseek in [1, 3]:
            paths_to_search = [available_files[m] for m in total_unique_medoids if m in available_files]
            process_foldseek_results(paths_to_search, args)
        elif args.foldseek == 2:
            process_foldseek_results(protein_files, args)

if __name__ == "__main__":
    main()
