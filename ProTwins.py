import glob
import subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import scipy
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, to_tree
import os
import time
import argparse
import pandas as pd
import gc

def obtener_tm_score(pdb1, pdb2):
    inicio = time.perf_counter()
    proceso = subprocess.run(["./USalign", pdb1, pdb2, "-mm", "1", "-ter", "0"], capture_output=True, text=True)
    fin = time.perf_counter()
    tiempo = fin - inicio
    score1, score2 = 0.0, 0.0
    for linea in proceso.stdout.split('\n'):
        if linea.startswith("TM-score=") and "Structure_1" in linea:
            score1 = float(linea.split()[1])
        if linea.startswith("TM-score=") and "Structure_2" in linea:
            score2 = float(linea.split()[1])
            return score1, score2, tiempo
    return score1, score2, tiempo

def definir_argumentos():
    parser = argparse.ArgumentParser(description="ProTwins: Análisis de Similitud Estructural y Funcional de Proteínas")
    parser.add_argument("-r", "--ruta", nargs='+', required=True,
                        help="Ruta(s) a las carpetas que contienen archivos .pdb o .cif")
    parser.add_argument("-o", "--output", type=str, required=True, 
                        help="Prefijo para los archivos generados (Obligatorio)")
    parser.add_argument("-d", "--outdir", type=str, required=True, 
                        help="Carpeta de salida (Obligatorio)")
    parser.add_argument("-u", "--umbral", type=float, nargs='+', default=[],
                        help="Uno o más umbrales de distancia para clustering. 0.2 y 0.5 siempre se incluyen.")
    return parser.parse_args()

def ejecutar_analisis_por_umbral(agrup, m_dist, etiquetas, umbral, nombre_modo, args, protein_files):
    # 1. Clustering y Medoides
    labels = fcluster(agrup, umbral, criterion='distance')
    k_encontrado = len(np.unique(labels))
    
    df_res = pd.DataFrame({"Proteina": etiquetas, "Cluster": labels})
    df_res['Es_Medoide'] = False
    mapeo_nombres_grafico = {prot: prot for prot in etiquetas}
    medoides_por_cluster = {}

    for cluster_id in np.unique(labels):
        prot_cluster = df_res[df_res['Cluster'] == cluster_id]['Proteina'].tolist()
        medoide = encontrar_medoide(prot_cluster, m_dist, etiquetas)
        df_res.loc[df_res['Proteina'] == medoide, 'Es_Medoide'] = True
        mapeo_nombres_grafico[medoide] = f"*** {medoide}"
        medoides_por_cluster[cluster_id] = medoide

    # 2. Configuración Visual Dinámica (Evitar Overlapping)
    n_prot = len(etiquetas)
    nombres_finales = [mapeo_nombres_grafico[e] for e in etiquetas]
    max_char = max(len(n) for n in nombres_finales)
    
    ancho_base = 14 + (max_char * 0.15)
    alto_figura = max(10, n_prot * 0.3) 
    tamanio_fuente_hojas = max(4, min(10, 600 / n_prot)) 
    
    fig, ax = plt.subplots(figsize=(ancho_base, alto_figura))
    plt.subplots_adjust(left=0.4) 

    ddata = dendrogram(
        agrup, 
        labels=nombres_finales, 
        orientation='right', 
        color_threshold=umbral, 
        above_threshold_color='grey',
        leaf_font_size=tamanio_fuente_hojas,
        ax=ax
    )

    # --- LÓGICA DE ALINEACIÓN ESTILO REPORTE (Sin Singletons) ---
    transform = ax.get_yaxis_transform() 
    y_coords = {leaf: i * 10 + 5 for i, leaf in enumerate(ddata['ivl'])}
    
    x_id_cluster = -0.35    
    x_bracket_back = -0.22  
    x_bracket_front = -0.18 

    for cluster_id in np.unique(labels):
        prot_cluster = df_res[df_res['Cluster'] == cluster_id]['Proteina'].tolist()
        y_vals = [y_coords[mapeo_nombres_grafico[p]] for p in prot_cluster if mapeo_nombres_grafico[p] in y_coords]
        
        if not y_vals: continue
        
        if len(y_vals) > 1:
            y_min, y_max = min(y_vals), max(y_vals)
            y_mid = (y_min + y_max) / 2
            
            ax.text(x_id_cluster, y_mid, f"C{cluster_id}", transform=transform, 
                    va='center', ha='left', fontsize=max(8, tamanio_fuente_hojas), fontweight='bold', clip_on=False)
            
            ax.plot([x_bracket_front, x_bracket_back, x_bracket_back, x_bracket_front], 
                    [y_min, y_min, y_max, y_max], 
                    color='black', transform=transform, lw=2.0, clip_on=False)

    # Estética Final
    plt.axvline(x=umbral, color='r', linestyle='--', label=f'Cutoff Threshold ({umbral:.2f})')
    plt.legend(loc='upper right', frameon=True, shadow=True)
    ax.set_title(f"Similarity Dendrogram - {args.output}", fontsize=16, pad=30)
    ax.set_xlabel("Structural Distance (1 - TM-score)", fontsize=12)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=tamanio_fuente_hojas)

    # Guardado (Nombre en Inglés: _dendrogram.pdf)
    ruta_pdf = os.path.join(args.outdir, f"{args.output}_{nombre_modo}_dendrogram.pdf")
    plt.savefig(ruta_pdf, format='pdf', bbox_inches='tight', dpi=150)
    plt.close('all')
    gc.collect()

    print(f"    [+] Dendrogram '{nombre_modo}' generado con éxito. K={k_encontrado}")
    
    # --- REINTEGRACIÓN DE SCRIPTS PYMOL ---
    subir_dir = os.path.join(args.outdir, "pymol_scripts", nombre_modo)
    os.makedirs(subir_dir, exist_ok=True)
    
    rutas_dict = {}
    for f in protein_files:
        nombre = os.path.basename(f).split('.')[0]
        rutas_dict[nombre] = os.path.relpath(f, start=subir_dir)

    medoides_validos = []

    for cluster_id, medoide in medoides_por_cluster.items():
        prot_cluster = df_res[df_res['Cluster'] == cluster_id]['Proteina'].tolist()
        
        if len(prot_cluster) < 2: 
            continue 
            
        medoides_validos.append(medoide)
        ruta_pml = os.path.join(subir_dir, f"cluster_{cluster_id}.pml")
        
        try:
            with open(ruta_pml, "w") as f:
                f.write(f"# Script PyMOL - Proyecto: {args.output}\n")
                f.write(f"# Modo: {nombre_modo} - Cluster {cluster_id}\n")
                f.write("reinitialize\n\n")
                
                # Inyección de código autónomo de ruta para la Terminal
                f.write("python\n")
                f.write("import os\n")
                f.write("_self_dir = os.path.dirname(cmd.get_script_path())\n")
                f.write("if _self_dir: os.chdir(_self_dir)\n")
                f.write("python end\n\n")
                
                f.write(f"load {rutas_dict[medoide]}, {medoide}\n")
                f.write(f"color magenta, {medoide}\n")
                
                for prot in prot_cluster:
                    if prot == medoide: 
                        continue
                    if prot in rutas_dict:
                        f.write(f"load {rutas_dict[prot]}, {prot}\n")
                        f.write(f"align {prot}, {medoide}\n")
                
                f.write("\nshow cartoon\n")
                f.write("util.cbc\n") 
                f.write("orient\n")
        except Exception as e:
            print(f"    [!] Error generando script PyMOL para C{cluster_id}: {e}")

    # Generar Sesión Maestra de Medoides (Nombres en Inglés: global_medoids / _medoids.pml)
    if medoides_validos:
        medoids_dir = os.path.join(args.outdir, "pymol_scripts", "global_medoids")
        os.makedirs(medoids_dir, exist_ok=True)
        ruta_master = os.path.join(medoids_dir, f"{umbral}_medoids.pml")
        
        rutas_dict_master = {}
        for f in protein_files:
            nombre = os.path.basename(f).split('.')[0]
            rutas_dict_master[nombre] = os.path.relpath(f, start=medoids_dir)

        try:
            with open(ruta_master, "w") as f:
                f.write(f"# Script PyMOL - Global Medoids (Clusters K >= 2) - Cutoff {umbral}\n")
                f.write("reinitialize\n\n")
                
                # Inyección de código autónomo de ruta para la Terminal
                f.write("python\n")
                f.write("import os\n")
                f.write("_self_dir = os.path.dirname(cmd.get_script_path())\n")
                f.write("if _self_dir: os.chdir(_self_dir)\n")
                f.write("python end\n\n")
                
                medoide_ref = medoides_validos[0]
                f.write(f"load {rutas_dict_master[medoide_ref]}, {medoide_ref}\n")
                f.write(f"color cyan, {medoide_ref}\n")
                
                for m in medoides_validos[1:]:
                    f.write(f"load {rutas_dict_master[m]}, {m}\n")
                    f.write(f"align {m}, {medoide_ref}\n")
                
                f.write("\nshow cartoon\nutil.cbc\norient\n")
            print(f"    [+] Sesión maestra de medoides generada: {ruta_master}")
        except Exception as e:
            print(f"    [!] Error generando script maestro de medoides: {e}")

    print(f"    [+] Análisis '{nombre_modo}' finalizado. Modelos guardados en: {subir_dir}")
    gc.collect()
    
def construir_newick(nodo, newick, parentdist, nombres_hojas):
    if nodo.is_leaf():
        return f"{nombres_hojas[nodo.id]}:{(parentdist - nodo.dist):.6f}{newick}"
    else:
        if len(newick) > 0:
            newick = f":{(parentdist - nodo.dist):.6f}{newick}"
        newick = f"({construir_newick(nodo.left, '', nodo.dist, nombres_hojas)},{construir_newick(nodo.right, '', nodo.dist, nombres_hojas)}){newick}"
        return newick

def guardar_newick(agrup, etiquetas, args):
    arbol = to_tree(agrup, rd=False)
    cadena_newick = construir_newick(arbol, "", arbol.dist, etiquetas) + ";"
    # Nombre en Inglés: _tree.nwk
    ruta_newick = os.path.join(args.outdir, f"{args.output}_tree.nwk")
    with open(ruta_newick, "w") as f:
        f.write(cadena_newick)
    print(f"Formato Newick guardado en: {ruta_newick}")

def generar_heat_maps(m_sim, m_dist_sim, etiquetas, args):
    # Nombres en Inglés: similarity / distance
    tareas = [
        (m_sim, "similarity", "coolwarm", 0, 1),
        (m_dist_sim, "distance", "viridis", 0, 1)
    ]
    
    n_prot = len(etiquetas)
    lado_figura = min(50, max(12, n_prot * 0.35)) 
    tamanio_fuente_hm = max(3, min(10, 400 / n_prot))
    
    mostrar_etiquetas = n_prot <= 150
    
    for matriz, nombre_base, mapa_color, v_min, v_max in tareas:
        plt.figure(figsize=(lado_figura, lado_figura)) 
        ax = sns.heatmap(
            matriz, 
            xticklabels=etiquetas if mostrar_etiquetas else False, 
            yticklabels=etiquetas if mostrar_etiquetas else False, 
            cmap=mapa_color, 
            vmin=v_min, 
            vmax=v_max,
            annot=False, 
            rasterized=True,
            cbar_kws={"shrink": 0.75}
        )    
        
        if mostrar_etiquetas:
            ax.tick_params(axis='x', labelsize=tamanio_fuente_hm)
            ax.tick_params(axis='y', labelsize=tamanio_fuente_hm)
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            
        plt.tight_layout()
        
        nombre_final = os.path.join(args.outdir, f"{args.output}_{nombre_base}.pdf")
        plt.savefig(nombre_final, format='pdf', bbox_inches='tight', dpi=150)
        print(f"    [+] Heatmap guardado en: {nombre_final}")
        
        plt.close() 
        gc.collect()

def generar_clustermap(m_sim, agrup, etiquetas, args):
    n_prot = len(etiquetas)
    lado_figura = min(50, max(14, n_prot * 0.3))
    tamanio_fuente_cm = max(4, min(10, 400 / n_prot))
    
    mostrar_etiquetas = n_prot <= 150
    
    g = sns.clustermap(
        m_sim,
        row_linkage=agrup,
        col_linkage=agrup,
        xticklabels=etiquetas if mostrar_etiquetas else False,
        yticklabels=etiquetas if mostrar_etiquetas else False,
        cmap="YlGnBu",
        linewidths=0,
        rasterized=True,  
        figsize=(lado_figura, lado_figura),
        cbar_pos=(0.02, 0.8, 0.05, 0.18),
        cbar_kws={'label': 'TM-score'}
    )

    if mostrar_etiquetas:
        g.ax_heatmap.tick_params(axis='x', labelsize=tamanio_fuente_cm)
        g.ax_heatmap.tick_params(axis='y', labelsize=tamanio_fuente_cm)
        
    plt.title(f"Global Clustermap - {args.output}")
    
    # Nombre en Inglés: _clustermap_final.pdf
    output_path = os.path.join(args.outdir, f"{args.output}_clustermap_final.pdf")
    g.savefig(output_path, format='pdf', bbox_inches='tight', dpi=150)
    plt.close()
    gc.collect()
    print(f"    [+] Clustermap guardado en: {output_path}")

def encontrar_medoide(cluster_proteinas, m_dist, etiquetas):
    if len(cluster_proteinas) == 1:
        return cluster_proteinas[0]
    
    indices = [etiquetas.index(p) for p in cluster_proteinas]
    submatriz = m_dist[np.ix_(indices, indices)]
    indice_medoide_local = np.argmin(submatriz.sum(axis=1))

    medoide_elegido = cluster_proteinas[indice_medoide_local]
    print(f"   > Medoide del cluster ({len(cluster_proteinas)} prot): {medoide_elegido}")
    
    return medoide_elegido

def guardar_matrices_csv(m_sim, m_dist, etiquetas, args):
    # Nombres en Inglés: _similarity.csv / _distance.csv
    ruta_sim = os.path.join(args.outdir, f"{args.output}_similarity.csv")
    ruta_dist = os.path.join(args.outdir, f"{args.output}_distance.csv")
    
    df_sim = pd.DataFrame(m_sim, index=etiquetas, columns=etiquetas)
    df_dist = pd.DataFrame(m_dist, index=etiquetas, columns=etiquetas)
    
    df_sim.to_csv(ruta_sim)
    df_dist.to_csv(ruta_dist)
    print(f"Matrices CSV guardadas en: {args.outdir}")

def main():
    args = definir_argumentos() 
    os.makedirs(args.outdir, exist_ok=True)

    print("\n" + "="*38)
    print("    PROTWINS: Análisis Proteico ")
    print("      De estructura y función   ")
    print("   Basado en USalign de ZhangLab")
    print("="*38 + "\n")

    protein_files = []
    for carpeta in args.ruta:
        if os.path.isdir(carpeta):
            protein_files.extend(glob.glob(os.path.join(carpeta, "*.pdb")) + 
                                glob.glob(os.path.join(carpeta, "*.cif")) +
                                glob.glob(os.path.join(carpeta, "*.cif.gz")) +
                                glob.glob(os.path.join(carpeta, "*.pdb.gz")))

    protein_files = sorted(list(set(protein_files)))
    n = len(protein_files)
    
    if n < 2:
        print("\n[!] ERROR: Se requieren al menos 2 archivos.")
        return 
    
    m_sim = np.ones((n, n)) 
    tiempo_total = 0 
    total_comparaciones = (n * (n - 1)) // 2
    print(f"Procesando {n} estructuras ({total_comparaciones} comparaciones totales)...")

    with tqdm(total=total_comparaciones, desc="Calculando TM-scores", unit="calc", colour="#228B22") as pbar:
        for i in range(n):
           for j in range(i+1, n):
                s1, s2, tiempo = obtener_tm_score(protein_files[i], protein_files[j])
                m_sim[i][j], m_sim[j][i] = s1, s2
                tiempo_total += tiempo  
                pbar.update(1)
    
    m_sim_s = (m_sim + m_sim.T) / 2 
    m_dist = 1 - m_sim_s
    np.fill_diagonal(m_dist, 0) 
    etiquetas = [os.path.basename(p).split('.')[0] for p in protein_files]

    guardar_matrices_csv(m_sim_s, m_dist, etiquetas, args)

    cond_dist = scipy.spatial.distance.squareform(m_dist)
    agrup = linkage(cond_dist, method="average")

    umbrales_ingresados = args.umbral if args.umbral else []
    umbrales_unicos = sorted(list(set(umbrales_ingresados + [0.2, 0.5])))
    
    print(f"\nEjecutando análisis jerárquico para los umbrales: {umbrales_unicos}")

    for idx, u in enumerate(umbrales_unicos, 1):
        print(f"\n[{idx}/{len(umbrales_unicos)}] Procesando Umbral de Corte: {u} ...")
        ejecutar_analisis_por_umbral(agrup, m_dist, etiquetas, u, str(u), args, protein_files)

    print("\nGenerando mapas de calor y dendrogramas globales...")
    generar_heat_maps(m_sim_s, m_dist, etiquetas, args)
    guardar_newick(agrup, etiquetas, args)
    generar_clustermap(m_sim_s, agrup, etiquetas, args)
    
    print(f"\n[!] ProTwins ha finalizado con éxito. Los resultados están en: {args.outdir}\n")

if __name__ == "__main__":
    main()