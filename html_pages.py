
import html

#reutilisaveis
P1_WARNING = """
<tr>
    <td colspan="2" style="background:#fff8e1;border-left:5px solid #f39c12;
                           color:#7a5200;padding:12px 14px;">
        ⚠️ <b>Warning:</b> the serological classification (seroclass and serogroup) was
        trained only on types from <b>clade P1</b>. For genomes from other clades,
        the result may not be reliable.
    </td>
</tr>"""


def confidence_color(value):
    """Verde (>=70), laranja (40-70), vermelho (<40)."""
    if value >= 70:
        return "#27ae60"
    if value >= 40:
        return "#e67e22"
    return "#c0392b"


def build_debug_log_html(log):
    """Bloco recolhível com o log da análise (só aparece se houve aviso/erro)."""
    if not log.has_problems:
        return ""
    text = html.escape("\n".join(log.lines))
    return f"""
    <details style="margin-top:25px;" open>
        <summary style="cursor:pointer;font-weight:bold;color:#7a5200;">
            🛠 Debug log (job {log.job_id[:8]})
        </summary>
        <pre style="background:#1e1e1e;color:#ddd;padding:14px;border-radius:8px;
                    overflow-x:auto;font-size:0.8em;white-space:pre-wrap;">{text}</pre>
    </details>"""


def build_error_page(title, message, hint, log):
    hint_html = f"<p><b>Hint:</b> {html.escape(hint)}</p>" if hint else ""
    return f"""
    <html lang="en">
    <head><title>Error</title></head>
    <body style="font-family:Arial,sans-serif;margin:40px;background:#f4f6f8;">
        <div style="background:white;padding:30px;border-radius:12px;
                    box-shadow:0 0 10px rgba(0,0,0,0.1);max-width:900px;margin:auto;">
            <h2 style="color:#c0392b;">{html.escape(title)}</h2>
            <p>{html.escape(message)}</p>
            {hint_html}
            {build_debug_log_html(log)}
            <a href="/" style="display:inline-block;margin-top:20px;padding:10px 20px;
               background:#3498db;color:white;text-decoration:none;border-radius:8px;">Back</a>
        </div>
    </body>
    </html>"""


#resultados

def build_ani_top3_html(top3):
    rows = ""
    for i, r in enumerate(top3, start=1):
        destaque = "background:#eafaf1;font-weight:bold;" if i == 1 else ""
        rows += f"""
        <tr style="{destaque}">
            <td style="text-align:center;">{i}º</td>
            <td><code>{html.escape(r["gcf"])}</code></td>
            <td><i>{html.escape(r["species"])}</i></td>
            <td>{r["ani"]:.3f}%</td>
            <td>{r["mapped_fragments"]} / {r["total_fragments"]}</td>
        </tr>"""
    return f"""
    <tr>
        <th colspan="2" style="background:#5d6d7e;color:white;">
            🏆 Top 3 closest reference genomes (FastANI)
        </th>
    </tr>
    <tr>
        <td colspan="2" style="padding:0;">
            <table style="margin:0;width:100%;border-collapse:collapse;">
                <tr>
                    <th style="width:45px;">#</th>
                    <th>Genome code (GCF/GCA)</th>
                    <th>Species</th>
                    <th>ANI</th>
                    <th>Fragments</th>
                </tr>
                {rows}
            </table>
        </td>
    </tr>"""


def build_serogroup_html(sg_result):
    """Bloco do serogrupo. Se houve erro, mostra o erro sem derrubar o ANI."""
    if "error" in sg_result:
        hint_row = ""
        if sg_result.get("hint"):
            hint_row = (f"<tr><th>Hint</th>"
                        f"<td>{html.escape(sg_result['hint'])}</td></tr>")
        return f"""
        <tr>
            <th colspan="2" style="background:#e67e22;color:white;">
                Serogroup Prediction
            </th>
        </tr>
        {P1_WARNING}
        <tr>
            <th>Status</th>
            <td style="color:#e74c3c;">{html.escape(sg_result["error"])}</td>
        </tr>
        {hint_row}"""

    top3_rows = "".join(
        f"<tr><td style='padding-left:20px'>{html.escape(sg)}</td><td>{conf}%</td></tr>"
        for sg, conf in sg_result["top3_serogroups"]
    )
    class_color = confidence_color(sg_result["class_confidence"])
    sg_color    = confidence_color(sg_result["serogroup_confidence"])
    return f"""
    <tr>
        <th colspan="2" style="background:#8e44ad;color:white;font-size:1.05em;">
            🧬 Serogroup Prediction (ML — rfb locus)
        </th>
    </tr>
    {P1_WARNING}
    <tr>
        <th>Seroclass</th>
        <td>
            <span style="font-size:1.15em;font-weight:bold;">
                Class {sg_result["predicted_class"]}
            </span>
            &nbsp;
            <span style="background:{class_color};color:white;padding:4px 12px;
                         border-radius:5px;font-weight:bold;font-size:1.05em;">
                {sg_result["class_confidence"]}% confidence
            </span>
        </td>
    </tr>
    <tr>
        <th>Predicted serogroup</th>
        <td>
            <span style="font-size:1.15em;font-weight:bold;">
                {html.escape(sg_result["predicted_serogroup"])}
            </span>
            &nbsp;
            <span style="background:{sg_color};color:white;padding:4px 12px;
                         border-radius:5px;font-weight:bold;font-size:1.05em;">
                {sg_result["serogroup_confidence"]}% confidence
            </span>
        </td>
    </tr>
    <tr>
        <th>Top-3 candidates</th>
        <td>
            <table style="width:100%;border:none;">
                {top3_rows}
            </table>
        </td>
    </tr>
    <tr>
        <th>Time (ML)</th>
        <td>{sg_result["elapsed_seconds"]} s</td>
    </tr>"""


def build_pca_html(sg_result, pcas, log):
    """
    Bloco com o PCA geral e o PCA da classe predita (amostra = estrela preta).
    `pcas` é o dicionário PCAS do main.py. Qualquer falha aqui NÃO derruba
    o resto do resultado.
    """
    if "error" in sg_result:
        return ""
    try:
        ps  = sg_result["protein_scores"]
        cl  = int(sg_result["predicted_class"])
        img_geral  = pcas["geral"].grafico(ps)
        img_classe = pcas[cl].grafico(ps)
        log.debug("PCA", "PCA plots generated")
        return f"""
        <tr>
            <th colspan="2" style="background:#16a085;color:white;font-size:1.05em;">
                📊 Principal Component Analysis (PCA)
            </th>
        </tr>
        <tr>
            <td colspan="2" style="text-align:center;">
                <p style="margin:5px;color:#666;">General PCA (all reference samples)</p>
                <img style="max-width:100%;" src="data:image/png;base64,{img_geral}">
                <p style="margin:15px 5px 5px;color:#666;">Class {cl} PCA (predicted class)</p>
                <img style="max-width:100%;" src="data:image/png;base64,{img_classe}">
            </td>
        </tr>"""
    except Exception:
        log.exception("PCA", "could not generate the PCA plots (results are still shown)")
        return ""


#pagina completona

def build_results_page(ani_result, sg_html, pca_html, log):
    """Página de resultados. Chame DEPOIS de build_pca_html, para que o
    debug log inclua eventuais avisos do PCA."""
    best  = ani_result["best_match"]
    color = "#2ecc71" if best["ani"] >= 95 else "#e74c3c"
    ani_top3_html = build_ani_top3_html(ani_result["top3"])
    debug_html    = build_debug_log_html(log)

    return f"""
    <html lang="en">
    <head>
        <title>Results — Leptospira ANI + Serogroup</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 40px;
                background: #f4f6f8;
            }}
            .container {{
                background: white;
                padding: 30px;
                border-radius: 12px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
                max-width: 900px;
                margin: auto;
            }}
            h1 {{ color: #2c3e50; }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 20px;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 12px;
                text-align: left;
            }}
            th {{ background: #34495e; color: white; }}
            .section-header {{
                background: #2c3e50;
                color: white;
                font-size: 1.05em;
            }}
            .classification {{
                background: {color};
                color: white;
                padding: 10px;
                border-radius: 8px;
                display: inline-block;
                font-weight: bold;
                margin-top: 20px;
            }}
            .ref-code {{
                display: inline-block;
                background: #2c3e50;
                color: white;
                padding: 6px 14px;
                border-radius: 6px;
                font-family: monospace;
                font-size: 1.2em;
                font-weight: bold;
                letter-spacing: 0.5px;
            }}
            .btn {{
                display: inline-block;
                margin-top: 30px;
                padding: 10px 20px;
                background: #3498db;
                color: white;
                text-decoration: none;
                border-radius: 8px;
            }}
            .btn:hover {{ background: #2980b9; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Identification Results</h1>
            <p><b>Analyzed file:</b> {html.escape(ani_result["query"])}</p>

            <div class="classification">
                {ani_result["classification"]}
            </div>

            <table>
                <tr>
                    <th colspan="2" class="section-header">
                        🔬 Species Identification (FastANI)
                    </th>
                </tr>
                <tr>
                    <th>Identified reference genome</th>
                    <td>
                        <span class="ref-code">{html.escape(best["gcf"])}</span>
                        <div style="color:#666;font-size:0.85em;margin-top:6px;">
                            File: {html.escape(best["reference"])}
                        </div>
                    </td>
                </tr>
                <tr><th>Species</th><td><i>{html.escape(best["species"])}</i></td></tr>
                <tr><th>ANI</th><td><b>{best["ani"]:.3f}%</b></td></tr>
                <tr>
                    <th>Mapped fragments</th>
                    <td>{best["mapped_fragments"]} / {best["total_fragments"]}</td>
                </tr>
                <tr>
                    <th>Time (FastANI)</th>
                    <td>{ani_result["tempo_total_segundos"]} s</td>
                </tr>

                {ani_top3_html}

                {sg_html}

                {pca_html}
            </table>

            {debug_html}

            <a class="btn" href="/">New analysis</a>
        </div>
    </body>
    </html>
    """


def build_pca_reference_page(pcas):
    """Página /pca com os 5 PCAs de referência."""
    blocos = ""
    for _, p in pcas.items():
        r = p.resumo()
        blocos += f"""
        <h3>{r["titulo"]}</h3>
        <p style="color:#666;">{r["n_amostras"]} samples · {r["n_features"]} features ·
           PC1 {r["variancia_explicada"][0]}% · PC2 {r["variancia_explicada"][1]}%</p>
        <img style="max-width:100%;" src="data:image/png;base64,{p.grafico()}">
        """
    return f"""
    <html>
    <head><title>PCA — Leptospira</title></head>
    <body style="font-family:Arial;margin:40px;background:#f4f6f8;">
        <div style="background:white;padding:30px;border-radius:12px;
                    box-shadow:0 0 10px rgba(0,0,0,0.1);max-width:900px;margin:auto;">
            <h1 style="color:#2c3e50;">PCA of serogroups</h1>
            {blocos}
            <a href="/" style="display:inline-block;margin-top:20px;padding:10px 20px;
               background:#3498db;color:white;text-decoration:none;border-radius:8px;">Back</a>
        </div>
    </body>
    </html>
    """


HOME_PAGE = """
<html lang="en">
<head>
    <title>Leptospira ANI Service</title>
    <style>
        body { font-family: Arial; margin: 60px; background: #f4f6f8; }
        .card {
            background: white; padding: 40px; border-radius: 12px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1); max-width: 500px; margin: auto;
        }
        h2 { color: #2c3e50; }
        input[type=file] { margin: 15px 0; }
        button {
            padding: 10px 24px; background: #3498db; color: white;
            border: none; border-radius: 8px; cursor: pointer; font-size: 1em;
        }
        button:hover { background: #2980b9; }
        p { color: #666; font-size: 0.9em; }
        .aviso {
            background: #fff8e1; border-left: 5px solid #f39c12;
            color: #7a5200; padding: 10px 14px; border-radius: 4px;
        }
    </style>
</head>
<body>
    <div class="card">
        <h2>🦠 Leptospira — ANI + Serogroup</h2>
        <p>
            Upload a genome (<code>.fna</code>, <code>.fasta</code> or <code>.fa</code>)
            to identify the species with <b>FastANI</b> and predict the
            <b>serogroup</b> from the rfb locus using machine learning.
        </p>
        <p class="aviso">
            ⚠️ The serological classification was trained only on types from <b>clade P1</b>.
        </p>
        <form action="/analyze" method="post" enctype="multipart/form-data">
            <input type="file" name="file" accept=".fna,.fasta,.fa">
            <br><br>
            <button type="submit">Analyze</button>
        </form>
        <p style="margin-top:20px;"><a href="/pca">View reference serogroup PCA</a></p>
    </div>
</body>
</html>
"""