import glob
import subprocess
import numpy as np
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
    return parser.parse_args()

def ejecutar_analisis_por_umbral(agrup, m_dist, etiquetas, umbral, nombre_modo, args, protein_files):
    """
    Dendrograma con layout dinámico:
    Los IDs de cluster (C1, C2...) se mueven automáticamente a la izquierda
    dependiendo del largo de los nombres de las proteínas.
    """
    import gc
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import fcluster, dendrogram

    # 1. Clustering
    labels = fcluster(agrup, umbral, criterion='distance')
    k_encontrado = len(np.unique(labels))
    
    # 2. Medoides y Etiquetas con Asteriscos
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

    # 3. Guardar CSV
    df_res.to_csv(os.path.join(args.outdir, f"{args.output}_{nombre_modo}_resultados.csv"), index=False)

    # 4. Dendrograma Dinámico
    # Calculamos el nombre más largo para ajustar el margen
    nombres_finales = [mapeo_nombres_grafico[e] for e in etiquetas]
    max_char = max(len(n) for n in nombres_finales)
    
    # Ajustamos el tamaño de la figura (más ancho si hay nombres largos)
    ancho_base = 12 + (max_char * 0.1)
    plt.figure(figsize=(ancho_base, max(10, len(etiquetas) * 0.3)))
    
    ddata = dendrogram(
        agrup, 
        labels=nombres_finales, 
        orientation='right', 
        color_threshold=umbral, 
        above_threshold_color='grey'
    )
    
    plt.axvline(x=umbral, color='r', linestyle='--', label=f'Umbral {nombre_modo} ({umbral:.2f})')
    plt.plot([], [], ' ', label="*** = Medoide del cluster") 
    plt.legend(loc='upper left')

    # --- LÓGICA DE POSICIONAMIENTO DINÁMICO ---
    ax = plt.gca()
    transform = ax.get_yaxis_transform() 
    y_coords = {leaf: 5 + i * 10 for i, leaf in enumerate(ddata['ivl'])}
    
    # Factor de desplazamiento: 
    # Los nombres están a la izquierda de X=0. 
    # Desplazamos los corchetes proporcionalmente al largo del nombre máximo.
    offset_nombres = - (max_char * 0.009) - 0.02 # Ajuste fino basado en caracteres
    x_bracket = offset_nombres 
    x_text = x_bracket - 0.04 # El ID del cluster un poco más a la izquierda

    for cluster_id in np.unique(labels):
        prot_cluster = df_res[df_res['Cluster'] == cluster_id]['Proteina'].tolist()
        y_vals = [y_coords[mapeo_nombres_grafico[p]] for p in prot_cluster if mapeo_nombres_grafico[p] in y_coords]
        
        if not y_vals: continue
        y_min, y_max = min(y_vals), max(y_vals)
        y_mid = (y_min + y_max) / 2
        
        # Dibujar corchete "["
        if len(y_vals) > 1:
            ax.plot([x_bracket, x_bracket - 0.01, x_bracket - 0.01, x_bracket], 
                    [y_min, y_min, y_max, y_max], 
                    color='black', transform=transform, clip_on=False, lw=1.5)
            
            # Texto del cluster (C#)
            ax.text(x_text, y_mid, f"C{cluster_id}", va='center', ha='right', 
                    transform=transform, clip_on=False, fontsize=10, fontweight='bold')
        else:
            # Para solitarios, solo el nombre del cluster
            ax.text(x_bracket - 0.01, y_mid, f"C{cluster_id}", va='center', ha='right', 
                    transform=transform, clip_on=False, fontsize=10, fontweight='bold')

    # Aumentamos el margen izquierdo dinámicamente para que quepa todo
    margin_left = min(0.4, 0.15 + (max_char * 0.01))
    plt.subplots_adjust(left=margin_left) 
    
    plt.savefig(os.path.join(args.outdir, f"{args.output}_{nombre_modo}_dendrograma.pdf"), format='pdf', bbox_inches='tight')
    plt.close()
    gc.collect()

    # 5. Generar Scripts de PyMOL (Lógica intacta)
    subir_dir = os.path.join(args.outdir, "scripts_pymol", nombre_modo)
    os.makedirs(subir_dir, exist_ok=True)
    rutas_dict = {os.path.basename(f).split('.')[0]: os.path.abspath(f) for f in protein_files}

    for cluster_id in np.unique(labels):
        prot_cluster = df_res[df_res['Cluster'] == cluster_id]['Proteina'].tolist()
        if len(prot_cluster) < 2: continue 
        
        medoide = medoides_por_cluster[cluster_id]
        ruta_pml = os.path.join(subir_dir, f"cluster_{cluster_id}.pml")
        
        with open(ruta_pml, "w") as f:
            f.write(f"# Script PyMOL - {nombre_modo.capitalize()} - Cluster {cluster_id}\nreinitialize\n\n")
            f.write(f"load {rutas_dict[medoide]}, {medoide}\n")
            f.write(f"color magenta, {medoide}\n")
            for prot in prot_cluster:
                if prot == medoide: continue
                f.write(f"load {rutas_dict[prot]}, {prot}\n")
                f.write(f"align {prot}, {medoide}\n")
            f.write("\nshow cartoon\nutil.cbc\norient\n")
    
    print(f"[+] Vista '{nombre_modo}' completada. K={k_encontrado}")
    gc.collect()

# --- FUNCIONES PARA NEWICK ---
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
    ruta_newick = os.path.join(args.outdir, f"{args.output}_arbol.nwk")
    with open(ruta_newick, "w") as f:
        f.write(cadena_newick)
    print(f"Formato Newick guardado en: {ruta_newick}")

def guardar_matrices_csv(m_sim, m_dist, etiquetas, args):
    ruta_sim = os.path.join(args.outdir, f"{args.output}_similitud.csv")
    ruta_dist = os.path.join(args.outdir, f"{args.output}_distancia.csv")
    
    df_sim = pd.DataFrame(m_sim, index=etiquetas, columns=etiquetas)
    df_dist = pd.DataFrame(m_dist, index=etiquetas, columns=etiquetas)
    
    df_sim.to_csv(ruta_sim)
    df_dist.to_csv(ruta_dist)
    print(f"Matrices CSV guardadas en: {args.outdir}")

def generar_heat_maps(m_sim, m_dist_sim, etiquetas, args):
    tareas = [
        (m_sim, "similitud", "coolwarm", 0, 1),
        (m_dist_sim, "distancia", "viridis", 0, 1)
    ]
    
    n_prot = len(etiquetas)
    lado_figura = max(10, n_prot * 0.5) 
    
    for matriz, nombre_base, mapa_color, v_min, v_max in tareas:
        plt.figure(figsize=(lado_figura, lado_figura)) 
        ax = sns.heatmap(
            matriz, 
            xticklabels=etiquetas, 
            yticklabels=etiquetas, 
            cmap=mapa_color, 
            vmin=v_min, 
            vmax=v_max,
            annot=False, 
            fmt=".3f",
            rasterized=True # Evita problemas de RAM en PDFs enormes
        )    
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        plt.tight_layout()
        
        nombre_final = os.path.join(args.outdir, f"{args.output}_{nombre_base}.pdf")
        plt.savefig(nombre_final, format='pdf', bbox_inches='tight')
        print(f"Heatmap guardado en: {nombre_final}")
        
        plt.close() 
        gc.collect()

def generar_clustermap(m_sim, agrup, etiquetas, args):
    """
    Genera un mapa de calor jerárquico (Clustermap) de las similitudes.
    """
    import seaborn as sns
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 12))
    
    g = sns.clustermap(
        m_sim,
        row_linkage=agrup,
        col_linkage=agrup,
        xticklabels=etiquetas,
        yticklabels=etiquetas,
        cmap="YlGnBu",
        linewidths=0,
        rasterized=True,  # Esto es lo que hace que el PDF no pese 100MB
        cbar_kws={'label': 'TM-score'}
    )

    plt.title(f"Clustermap Global - {args.output}")
    
    output_path = os.path.join(args.outdir, f"{args.output}_clustermap_final.pdf")
    g.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close()
    gc.collect()
    print(f"[+] Clustermap guardado en: {output_path}")

def encontrar_medoide(cluster_proteinas, m_dist, etiquetas):
    if len(cluster_proteinas) == 1:
        return cluster_proteinas[0]
    
    indices = [etiquetas.index(p) for p in cluster_proteinas]
    submatriz = m_dist[np.ix_(indices, indices)]
    indice_medoide_local = np.argmin(submatriz.sum(axis=1))

    medoide_elegido = cluster_proteinas[indice_medoide_local]
    print(f"   > Medoide del cluster ({len(cluster_proteinas)} prot): {medoide_elegido}")
    
    return medoide_elegido

def generar_scripts_pymol(df_resultados, protein_files, m_dist, etiquetas, args):
    scripts_dir = os.path.join(args.outdir, "scripts_pymol")
    os.makedirs(scripts_dir, exist_ok=True)
    
    rutas_dict = {os.path.basename(f).split('.')[0]: os.path.abspath(f) for f in protein_files}

    for cluster_id in df_resultados['Cluster'].unique():
        prot_cluster = df_resultados[df_resultados['Cluster'] == cluster_id]['Proteina'].tolist()
        
        if len(prot_cluster) < 2: continue 
        
        medoide = encontrar_medoide(prot_cluster, m_dist, etiquetas)
        ruta_pml = os.path.join(scripts_dir, f"cluster_{cluster_id}.pml")
        
        with open(ruta_pml, "w") as f:
            f.write(f"# Script PyMOL - Cluster {cluster_id}\nreinitialize\n\n")
            f.write(f"load {rutas_dict[medoide]}, {medoide}\n")
            f.write(f"color magenta, {medoide}\n")
            
            for prot in prot_cluster:
                if prot == medoide: continue
                f.write(f"load {rutas_dict[prot]}, {prot}\n")
                f.write(f"align {prot}, {medoide}\n")
            
            f.write("\nshow cartoon\nutil.cbc\norient\n")

def main():
    args = definir_argumentos() 
    os.makedirs(args.outdir, exist_ok=True)

    print("\n" + "="*40)
    print("   PROTWINS: Análisis de Gemelos Proteicos")
    print("   Basado en algoritmos de Zhang Lab")
    print("="*40 + "\n")

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

    with tqdm(total=total_comparaciones, desc="Calculando TM-scores", unit="calc", colour="green") as pbar:
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

    print("\n[1/2] Ejecutando Vista Estructural (TM >= 0.5)...")
    ejecutar_analisis_por_umbral(agrup, m_dist, etiquetas, 0.5, "estructural", args, protein_files)

    print("\n[2/2] Ejecutando Vista Funcional (TM >= 0.8)...")
    ejecutar_analisis_por_umbral(agrup, m_dist, etiquetas, 0.2, "funcional", args, protein_files)

    print("\nGenerando mapas de calor y dendrogramas globales...")
    generar_heat_maps(m_sim_s, m_dist, etiquetas, args)
    guardar_newick(agrup, etiquetas, args)
    generar_clustermap(m_sim_s, agrup, etiquetas, args)
    
    print(f"\n[!] ProTwins ha finalizado con éxito. Los resultados están en: {args.outdir}\n")

if __name__ == "__main__":
    main()
