from fastapi import FastAPI, UploadFile, File
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse
import os
import glob
import subprocess
import re
import shutil
import time
import uuid
import logging
import warnings
import tempfile

import pandas as pd
import numpy as np
import joblib

import geradorPCA   # módulo de PCA (geradorPCA.py na mesma pasta)
import html_pages   # todo o HTML fica em html_pages.py

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

app = FastAPI()

# ─────────────────────────────────────────────
# LOGGING / DEPURAÇÃO
# Cada requisição tem um "job id" curto.
#
# REGRA DE PRIVACIDADE:
#   - `message` é o texto SEGURO: vai para o terminal E para a página.
#     Nunca coloque nele caminhos, nomes de arquivo, stderr, comandos
#     ou mensagens de exceção.
#   - `detail` é o texto SENSÍVEL: vai SÓ para o terminal do servidor.
# ─────────────────────────────────────────────

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-7s %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("lepto")
logger.setLevel(logging.DEBUG)
for noisy_library in ("matplotlib", "PIL"):
    logging.getLogger(noisy_library).setLevel(logging.WARNING)


class JobLog:
    """Guarda e imprime as mensagens de depuração de UMA análise."""

    def __init__(self, job_id):
        self.job_id = job_id
        self.lines = []             # só mensagens seguras (mostradas na página)
        self.has_problems = False   # True se houve WARNING ou ERROR

    def _add(self, level_name, level_number, step, message, detail=None):
        self.lines.append(f"{level_name:<7} [{step}] {message}")
        full_message = f"{message} | {detail}" if detail else message
        logger.log(level_number, "[job %s] [%s] %s", self.job_id[:8], step, full_message)

    def debug(self, step, message, detail=None):
        self._add("DEBUG", logging.DEBUG, step, message, detail)

    def terminal_only(self, step, message):
        """Texto longo (ex.: stderr completo): só no terminal, não na página."""
        logger.debug("[job %s] [%s] %s", self.job_id[:8], step, message)

    def info(self, step, message, detail=None):
        self._add("INFO", logging.INFO, step, message, detail)

    def warning(self, step, message, detail=None):
        self.has_problems = True
        self._add("WARNING", logging.WARNING, step, message, detail)

    def error(self, step, message, detail=None):
        self.has_problems = True
        self._add("ERROR", logging.ERROR, step, message, detail)

    def exception(self, step, message, detail=None):
        """Chamar DENTRO de um 'except'. A página recebe só `message`;
        o traceback completo (que pode ter caminhos) fica no terminal."""
        self.has_problems = True
        self.lines.append(f"ERROR   [{step}] {message}")
        full_message = f"{message} | {detail}" if detail else message
        logger.error("[job %s] [%s] %s", self.job_id[:8], step, full_message, exc_info=True)


# ─────────────────────────────────────────────
# CONFIGURAÇÕES
# ─────────────────────────────────────────────

FASTANI   = "fastANI"
THREADS   = 8

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("LEPTO_DATA_DIR", os.path.join(BASE_DIR, "data"))

REF_DIR         = os.path.join(DATA_DIR, "reference_genomes")
MODEL_DIR       = os.path.join(DATA_DIR, "models")
FEATURES_DIR    = os.path.join(DATA_DIR, "features")
MULTIFASTAS_DIR = os.path.join(DATA_DIR, "multi_fasta")

TEMP_DIR = os.path.join(tempfile.gettempdir(), "lepto_jobs")
os.makedirs(TEMP_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = (".fa", ".fasta", ".fna")

# Frase padrão para problemas que só o administrador pode resolver
CONTACT_ADMIN = "This is a server-side problem. Please contact the administrator and mention the job code."

# ─────────────────────────────────────────────
# PRÉ-CARREGAMENTO DAS FEATURES
# Lê os TSVs uma única vez na inicialização do servidor.
# Assim não relê do disco a cada requisição.
# ─────────────────────────────────────────────

def _load_features(path):
    """Lê o cabeçalho de um TSV e retorna só as colunas de proteína."""
    drop = {"Unnamed: 0", "True_serogroup", "class", "Serogroup"}
    df   = pd.read_csv(path, sep="\t", nrows=0)
    return [c for c in df.columns if c not in drop]

# Estágio 1: todos os modelos de classe usam as features do TSV geral
GLOBAL_FEATURES = _load_features(
    os.path.join(FEATURES_DIR, "general_table_identity_rep.tsv")
)

# Estágio 2: cada classe tem seu próprio TSV com subconjunto de features
CLASS_FEATURES = {
    cl: _load_features(os.path.join(FEATURES_DIR, f"table_identity_cl_{cl}.tsv"))
    for cl in range(1, 5)
}

# PCAs de referência (geral + classes 1-4), calculados uma vez na inicialização
PCAS = geradorPCA.carregar_todos(FEATURES_DIR)

logger.info("Started: %d general features, %d PCA models, REF_DIR=%s",
            len(GLOBAL_FEATURES), len(PCAS), REF_DIR)


# ─────────────────────────────────────────────
# FUNÇÕES AUXILIARES — arquivos e nomes
# ─────────────────────────────────────────────

# Aceita GCF_/GCA_ + 9 dígitos + versão com "." OU "_" (ex.: GCF_000007685.1
# e GCA_945887855_1). O nome do arquivo pode ter qualquer prefixo/sufixo.
ACCESSION_PATTERN = re.compile(r"(GC[FA])_(\d{9})(?:[._](\d+))?")

# Gêneros da ordem Leptospirales: se aparecerem no cabeçalho, são a fonte
# mais confiável do nome da espécie.
SPECIES_PATTERN = re.compile(r"\b(Leptospira|Leptonema|Turneriella)\s+([a-z][A-Za-z0-9.\-]*)")

# Palavras comuns de cabeçalho que NÃO são gênero (evita "Complete genome" etc.)
NON_GENUS_WORDS = {
    "Complete", "Whole", "Chromosome", "Plasmid", "Contig", "Scaffold", "Genome",
    "Draft", "Strain", "Serovar", "Isolate", "MAG", "Shotgun", "Genomic",
    "Assembly", "Sequence", "Uncultured", "Unplaced",
}

NUCLEOTIDE_CHARACTERS = "ACGTNRYKMSWBDHVU-*"
_REMOVE_NUCLEOTIDES = str.maketrans("", "", NUCLEOTIDE_CHARACTERS)


def make_safe_file_name(original_name):
    """Tira diretórios e troca qualquer caractere fora de [A-Za-z0-9._-] por '_'."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", os.path.basename(original_name))


def short_text(text, limit=300):
    """Junta as linhas em uma só e mantém o FINAL do texto (onde costuma estar o erro).
    Use só em `detail` (terminal): o texto pode conter caminhos do servidor."""
    one_line = " | ".join(part.strip() for part in text.strip().splitlines() if part.strip())
    return one_line if len(one_line) <= limit else "..." + one_line[-limit:]


def run_command(arguments):
    """Executa um programa SEM shell (nomes de arquivo com espaço, aspas, ';'
    ou acento não quebram o comando). Retorna o CompletedProcess."""
    return subprocess.run(arguments, capture_output=True, text=True)


def validate_genome_fasta(file_path, log):
    """
    Confere se o arquivo enviado parece um FASTA de nucleotídeos.
    Retorna (mensagem_de_erro, dica) ou (None, None) se estiver OK.
    """
    step = "VALIDATION"
    file_size = os.path.getsize(file_path)
    log.info(step, f"file size = {file_size:,} bytes")

    if file_size == 0:
        return "The uploaded file is empty.", "Check that the file was not truncated during upload."

    sequence_count = 0
    total_bases = 0
    non_nucleotide_bases = 0
    gc_bases = 0
    first_line_checked = False

    with open(file_path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if not first_line_checked:
                first_line_checked = True
                if not line.startswith(">"):
                    log.debug(step, "first line does not start with '>'",
                              detail=f"first line: {line[:60]!r}")
                    return ("The file does not start with '>' (not a FASTA file).",
                            "FASTA headers must start with '>'.")
            if line.startswith(">"):
                sequence_count += 1
                continue
            upper_line = line.upper()
            total_bases += len(upper_line)
            non_nucleotide_bases += len(upper_line.translate(_REMOVE_NUCLEOTIDES))
            gc_bases += upper_line.count("G") + upper_line.count("C")

    log.info(step, f"{sequence_count} sequence(s), {total_bases:,} bases, "
                   f"GC = {100 * gc_bases / max(total_bases, 1):.1f}%")

    if sequence_count == 0 or total_bases == 0:
        return "No sequences found in the file.", "The file has no FASTA records (header + sequence)."

    if non_nucleotide_bases / total_bases > 0.10:
        return ("The file does not look like a nucleotide FASTA "
                f"({100 * non_nucleotide_bases / total_bases:.0f}% of the characters are not nucleotides).",
                "It may be a protein FASTA. Upload the genome assembly (DNA).")

    if total_bases < 1_000_000:
        log.warning(step, f"only {total_bases:,} bases: very small for a Leptospira genome "
                          f"(expected about 3.5-4.7 Mb). FastANI may report no results.")
    return None, None


def extract_species(fasta_file, log):
    """
    Lê os primeiros cabeçalhos do FASTA e extrai o nome da espécie.
    1) procura 'Leptospira xxx' (ou Leptonema/Turneriella);
    2) senão, procura um par 'Gênero espécie' ignorando palavras comuns
       como 'Complete genome';
    3) senão, devolve 'Unknown' e registra o motivo.
    """
    first_header = None
    headers_checked = 0
    with open(fasta_file, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith(">"):
                continue
            header = line[1:].strip()
            first_header = first_header or header
            headers_checked += 1

            known_match = SPECIES_PATTERN.search(header)
            if known_match:
                return f"{known_match.group(1)} {known_match.group(2)}"

            words = header.split()
            for current_word, next_word in zip(words, words[1:]):
                if (current_word[0].isupper() and current_word.rstrip(":,;") not in NON_GENUS_WORDS
                        and current_word.isalpha()
                        and next_word.isalpha() and next_word.islower() and len(next_word) >= 3):
                    return f"{current_word} {next_word}"

            if headers_checked >= 20:     # não varrer arquivos gigantes inteiros
                break

    log.warning("SPECIES", "could not read a species name from the header of a reference genome.",
                detail=f"file={os.path.basename(fasta_file)!r} first header={first_header!r}")
    return "Unknown"


def extract_gcf(filename, log):
    """
    Extrai o código do genoma (GCF_ ou GCA_) do nome do arquivo.
    Se não houver código, devolve o próprio nome do arquivo (sem extensão)
    em vez de 'Unknown', e registra o aviso.
    """
    base = os.path.basename(filename)
    match = ACCESSION_PATTERN.search(base)
    if match:
        prefix, digits, version = match.groups()
        return f"{prefix}_{digits}" + (f".{version}" if version else "")

    stem = os.path.splitext(base)[0]
    log.warning("NAMING", "a reference file has no GCF_/GCA_ accession in its name; "
                          "showing the file name instead.",
                detail=f"file={base!r}")
    return stem


# ─────────────────────────────────────────────
# FUNÇÕES AUXILIARES — FastANI
# ─────────────────────────────────────────────

_IMPORTANT_STDERR_PATTERN = re.compile(
    r"\b(warn\w*|error\w*|skip\w*|cannot|could not|fail\w*|invalid|empty)\b", re.IGNORECASE)


def report_fastani_stderr(stderr_text, log):
    """O stderr do FastANI é quase todo mensagem de progresso e contém caminhos de
    arquivos. A página mostra só a CONTAGEM de linhas suspeitas; o texto vai ao terminal."""
    step = "FASTANI"
    lines = [line.strip() for line in stderr_text.splitlines() if line.strip()]
    # Ignora as linhas "Reference = [...]" / "Query = [...]": só listam caminhos de arquivos
    # (um nome de arquivo com a palavra "empty", por exemplo, não é um aviso).
    important = [line for line in lines
                 if not line.startswith(("Reference =", "Query ="))
                 and _IMPORTANT_STDERR_PATTERN.search(line)]
    log.terminal_only(step, "full stderr:\n" + stderr_text)
    log.debug(step, f"stderr has {len(lines)} line(s), {len(important)} look like warnings/errors")
    if important:
        log.warning(step, f"FastANI reported {len(important)} warning/error line(s) "
                          f"(details are in the server log)",
                    detail=" || ".join(line[:200] for line in important[:10]))


def inspect_reference_files(reference_paths, log):
    """Abre o início de cada referência e procura arquivos que o FastANI não
    consegue usar (vazios, compactados .gz, sem '>' ou que são proteína).
    Retorna a lista de (nome, problema). Nomes e problemas ficam só no terminal."""
    step = "DIAGNOSIS"
    problems = []
    for reference_path in reference_paths:
        name = os.path.basename(reference_path)
        try:
            size = os.path.getsize(reference_path)
            with open(reference_path, "rb") as handle:
                head = handle.read(4096)
        except OSError as os_error:
            problems.append((name, f"cannot be read ({os_error})"))
            continue
        if size == 0:
            problems.append((name, "empty file"))
        elif head[:2] == b"\x1f\x8b":
            problems.append((name, "gzip-compressed (FastANI needs plain-text FASTA; decompress it)"))
        elif not head.startswith(b">"):
            problems.append((name, f"does not start with '>' (starts with {head[:20]!r})"))
        else:
            sequence_text = "".join(
                line for line in head.decode("utf-8", errors="replace").splitlines()[1:]
                if not line.startswith(">")).upper()
            non_nucleotide = len(sequence_text.translate(_REMOVE_NUCLEOTIDES))
            if sequence_text and non_nucleotide / len(sequence_text) > 0.10:
                problems.append((name, "looks like a protein FASTA, not a genome (DNA)"))

    log.info(step, f"{len(reference_paths)} reference file(s) inspected: "
                   f"{len(problems)} with problems")
    if problems:
        shown = "; ".join(f"{name}: {problem}" for name, problem in problems[:10])
        if len(problems) > 10:
            shown += f"; ... and {len(problems) - 10} more"
        log.error(step, f"{len(problems)} reference file(s) have problems "
                        f"(details are in the server log)",
                  detail=shown)
    return problems


def check_query_against_itself(query_file, job_dir, log):
    """Compara o genoma com ELE MESMO. Se o FastANI não gerar 100%, o problema é o
    genoma enviado (muito pequeno/fragmentado ou não é DNA), não as referências."""
    step = "DIAGNOSIS"
    output_path = os.path.join(job_dir, "self_check.txt")
    process = run_command([FASTANI, "-q", query_file, "-r", query_file,
                           "-t", str(THREADS), "-o", output_path])
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        with open(output_path) as handle:
            columns = handle.readline().strip().split("\t")
        log.info(step, f"self-comparison OK: ANI {columns[2]}%, {columns[3]}/{columns[4]} "
                       f"fragments -> FastANI can read the uploaded genome")
        return True
    log.error(step, f"self-comparison produced NO result (exit code {process.returncode}): "
                    f"FastANI cannot use the uploaded genome")
    return False


def diagnose_no_fastani_results(query_file, reference_paths, log):
    """Roda quando o FastANI termina sem nenhuma linha de resultado.
    Registra o que encontrou no log e devolve a dica para a página."""
    job_dir = os.path.dirname(query_file)
    reference_problems = inspect_reference_files(reference_paths, log)
    query_is_usable = check_query_against_itself(query_file, job_dir, log)

    if len(reference_problems) == len(reference_paths):
        return ("None of the server's reference genomes could be used. " + CONTACT_ADMIN)
    extra = (" Some reference genomes on the server are also invalid. " + CONTACT_ADMIN
             if reference_problems else "")
    if not query_is_usable:
        return ("FastANI could not use the uploaded genome, not even against itself "
                "(too small/fragmented, or not DNA)." + extra)
    return ("FastANI can read the uploaded genome, but none of the reference genomes "
            "reached the minimum ANI that FastANI reports (about 80%). The sample may be "
            "unrelated to Leptospira." + extra)


def run_fastani(query_file, log):
    """
    Compara o genoma query com os genomas de referência.
    Retorna o melhor ANI (best_match) e os 3 melhores resultados (top3),
    ou {"error": ..., "hint": ...}.
    """
    step = "FASTANI"
    start_total = time.time()

    if shutil.which(FASTANI) is None:
        log.error(step, "program 'fastANI' was not found on the server.",
                  detail=f"'{FASTANI}' is not on the PATH")
        return {"error": "The FastANI program is not available on the server.",
                "hint": CONTACT_ADMIN}

    if not os.path.isdir(REF_DIR):
        log.error(step, "reference directory does not exist.", detail=f"REF_DIR={REF_DIR}")
        return {"error": "Reference genome directory not found on the server.",
                "hint": CONTACT_ADMIN}

    refs = []
    for ext in ALLOWED_EXTENSIONS:
        refs.extend(glob.glob(os.path.join(REF_DIR, f"*{ext}")))
    refs.sort()
    log.info(step, f"{len(refs)} reference genome(s) found", detail=f"REF_DIR={REF_DIR}")

    if not refs:
        log.error(step, "no reference genome files found.", detail=f"REF_DIR={REF_DIR}")
        return {"error": "No reference genome available on the server.",
                "hint": CONTACT_ADMIN}

    job_dir      = os.path.dirname(query_file)
    reflist_path = os.path.join(job_dir, "refs.txt")
    with open(reflist_path, "w") as f:
        for r in refs:
            f.write(r + "\n")

    output_file = os.path.join(job_dir, "fastani.txt")
    command = [FASTANI, "-q", query_file, "--rl", reflist_path,
               "-t", str(THREADS), "-o", output_file]
    log.debug(step, "running FastANI", detail="command: " + " ".join(command))

    process = run_command(command)
    log.debug(step, f"exit code = {process.returncode}")
    report_fastani_stderr(process.stderr, log)
    if process.returncode != 0:
        log.error(step, f"fastANI failed with exit code {process.returncode}",
                  detail="stderr: " + (short_text(process.stderr) or "(empty)"))
        return {"error": f"FastANI failed (exit code {process.returncode}).",
                "hint": "Check that the uploaded file is a valid genome assembly. "
                        "If it is, " + CONTACT_ADMIN[0].lower() + CONTACT_ADMIN[1:]}

    # Lê TODOS os resultados (o FastANI só reporta pares com ANI mínimo)
    results = []
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        with open(output_file) as f:
            for line_number, line in enumerate(f, start=1):
                cols = line.strip().split("\t")
                if len(cols) < 5:
                    log.warning(step, f"output line {line_number} has {len(cols)} columns "
                                      f"(expected 5)",
                                detail=f"line: {line.strip()[:80]!r}")
                    continue
                _, ref_path, ani, mapped, total = cols[:5]
                try:
                    ani_value = float(ani)
                except ValueError:
                    log.warning(step, f"output line {line_number}: ANI is not a number",
                                detail=f"value: {ani!r}")
                    continue
                results.append({
                    "ref_path":         ref_path,
                    "reference":        os.path.basename(ref_path),
                    "gcf":              extract_gcf(ref_path, log),
                    "ani":              ani_value,
                    "mapped_fragments": mapped,
                    "total_fragments":  total,
                })

    log.info(step, f"{len(results)} of {len(refs)} reference genome(s) produced an ANI value")

    if not results:
        log.error(step, "FastANI finished without errors but its output file is empty: "
                        "no reference reached the minimum ANI it reports. Running diagnosis...")
        return {"error": "No results found.",
                "hint": diagnose_no_fastani_results(query_file, refs, log)}

    # Top 3 por ANI (decrescente). A espécie é lida só dos 3 finalistas.
    results.sort(key=lambda r: r["ani"], reverse=True)
    top3 = results[:3]
    for r in top3:
        r["species"] = extract_species(r["ref_path"], log)
        log.info(step, f"top match: {r['gcf']} | {r['species']} | ANI {r['ani']:.3f}%")

    best = top3[0]
    classification = "Species identified" if best["ani"] >= 95 else "Species not identified"
    if best["ani"] < 95:
        log.warning(step, f"best ANI is {best['ani']:.3f}% (< 95%): species not identified.")

    return {
        "query":                os.path.basename(query_file),
        "best_match":           best,
        "top3":                 top3,
        "classification":       classification,
        "tempo_total_segundos": round(time.time() - start_total, 2),
    }


# ─────────────────────────────────────────────
# FUNÇÕES AUXILIARES — Modelo de Serogrupo
# ─────────────────────────────────────────────

def create_blast_db(genome_fasta, work_dir, log):
    """
    Cria um banco BLAST nucleotídico do genoma query.
    Necessário para o tblastn conseguir buscar nele.
    """
    step = "BLAST"
    if shutil.which("makeblastdb") is None:
        log.error(step, "program 'makeblastdb' was not found on the server.")
        return None, "A required program (BLAST+) is not available on the server. " + CONTACT_ADMIN

    db_out  = os.path.join(work_dir, "blast_db")
    command = ["makeblastdb", "-in", genome_fasta, "-dbtype", "nucl", "-out", db_out]
    log.debug(step, "creating BLAST database", detail="command: " + " ".join(command))
    process = run_command(command)
    if process.returncode != 0:
        log.error(step, f"makeblastdb failed (exit code {process.returncode})",
                  detail="stderr: " + short_text(process.stderr))
        return None, ("Could not build a BLAST database from the uploaded genome. "
                      "Check that the file is a valid genome assembly.")
    log.debug(step, "BLAST database created")
    return db_out, None


def run_tblastn(genome_fasta, db_name, work_dir, log):
    """
    Busca as proteínas do locus rfb (em MULTIFASTAS_DIR) contra o genoma query.

    tblastn = busca proteína (aminoácidos) contra banco de nucleotídeos (DNA),
    traduzindo o DNA em tempo real.

    Retorna dicionário {id_proteína: melhor_score}, onde:
        score = pident * (length / qlen)
        pident = % de identidade (0-100)
        length = nº de aminoácidos que alinharam
        qlen   = tamanho total da proteína de referência
    Score alto = proteína presente com alta identidade e cobertura.
    Score zero = proteína ausente no genoma.
    """
    step = "BLAST"
    temp_out       = os.path.join(work_dir, "temp_blast.out")
    protein_scores = {}

    if shutil.which("tblastn") is None:
        log.error(step, "program 'tblastn' was not found on the server.")
        return protein_scores

    fasta_files = sorted(
        glob.glob(os.path.join(MULTIFASTAS_DIR, "*.fasta")) +
        glob.glob(os.path.join(MULTIFASTAS_DIR, "*.fas"))
    )

    if not fasta_files:
        log.error(step, "no protein FASTA files found for the rfb locus.",
                  detail=f"MULTIFASTAS_DIR={MULTIFASTAS_DIR}")
        return protein_scores

    files_with_hits = 0
    files_failed    = 0
    for fasta_file in fasta_files:
        fasta_name = os.path.basename(fasta_file)
        try:
            if os.path.exists(temp_out):
                os.remove(temp_out)   # evita ler resultado velho do arquivo anterior
            command = [
                "tblastn",
                "-query",  fasta_file,
                "-db",     db_name,
                "-outfmt", "6 qseqid sseqid pident length qlen",
                "-out",    temp_out,
            ]
            process = run_command(command)
            if process.returncode != 0:
                files_failed += 1
                log.warning(step, f"tblastn failed for one protein FASTA "
                                  f"(exit code {process.returncode})",
                            detail=f"file={fasta_name!r} stderr: {short_text(process.stderr, 200)}")
                continue

            if os.path.exists(temp_out) and os.path.getsize(temp_out) > 0:
                files_with_hits += 1
                cols     = ["qseqid", "sseqid", "pident", "length", "qlen"]
                df_blast = pd.read_csv(temp_out, sep="\t", header=None, names=cols)

                df_blast["score"] = df_blast.apply(
                    lambda r: r["pident"] * (r["length"] / r["qlen"]) if r["qlen"] > 0 else 0.0,
                    axis=1,
                )
                best_hits = df_blast.groupby("qseqid")["score"].max().to_dict()
                protein_scores.update(best_hits)
            else:
                log.terminal_only(step, f"no hits for {fasta_name!r}")

        except Exception:
            files_failed += 1
            log.exception(step, "error while processing one protein FASTA",
                          detail=f"file={fasta_name!r}")
            continue

    log.info(step, f"{len(fasta_files)} protein FASTA file(s): {files_with_hits} with hits, "
                   f"{files_failed} failed; {len(protein_scores)} protein(s) with a score")
    return protein_scores


def format_features(protein_scores, expected_features):
    """
    Monta o DataFrame de entrada para o modelo.

    Garante que o DataFrame tenha:
    - Exatamente as colunas que o modelo espera
    - Na ordem correta
    - Com 0.0 para proteínas não encontradas pelo BLAST
    - Ignorando proteínas encontradas que o modelo não conhece
    """
    row = {feat: 0.0 for feat in expected_features}
    for prot, score in protein_scores.items():
        if prot in row:
            row[prot] = score
    return pd.DataFrame([row], columns=expected_features)


def check_feature_name_match(protein_scores, log):
    """
    Confere se os IDs que o BLAST devolveu batem com os nomes das colunas
    da tabela de features. Se nenhum bater, o modelo receberia só zeros.
    """
    step = "FEATURES"
    matched = [name for name in GLOBAL_FEATURES if name in protein_scores]
    log.info(step, f"{len(matched)} of {len(GLOBAL_FEATURES)} general features were found "
                   f"in the genome (the others are set to 0.0)")
    if not matched:
        log.error(step, "NONE of the BLAST protein IDs match the feature table columns.",
                  detail=f"BLAST IDs (sample): {list(protein_scores)[:3]} | "
                         f"table columns (sample): {GLOBAL_FEATURES[:3]}")
        return ("Serogroup prediction is not configured correctly on the server.",
                CONTACT_ADMIN)
    return None, None


def run_serogroup_model(genome_fasta, work_dir, log):
    """
    Pipeline completo de predição de serogrupo em 4 etapas:

    Etapa 1 — Cria banco BLAST do genoma query
    Etapa 2 — Busca as proteínas do locus rfb via tblastn → protein_scores
    Etapa 3 — Estágio 1: qual das 4 seroclasses?
    Etapa 4 — Estágio 2: qual serogrupo dentro da classe predita?

    Retorna dicionário com resultado ou {"error": mensagem, "hint": dica}.
    """
    start = time.time()

    # ── Etapa 1: banco BLAST ──────────────────────────────────────────────
    db_name, db_error = create_blast_db(genome_fasta, work_dir, log)
    if not db_name:
        return {"error": "Failed to create BLAST database for serogroup prediction.",
                "hint": db_error}

    # ── Etapa 2: extração de features via tblastn ─────────────────────────
    protein_scores = run_tblastn(genome_fasta, db_name, work_dir, log)
    if not protein_scores:
        return {"error": "No rfb locus proteins were found in this genome.",
                "hint": "The genome may not contain the rfb locus, or the server's protein "
                        "set is unavailable. If you expected a result, " +
                        CONTACT_ADMIN[0].lower() + CONTACT_ADMIN[1:]}

    mismatch_error, mismatch_hint = check_feature_name_match(protein_scores, log)
    if mismatch_error:
        return {"error": mismatch_error, "hint": mismatch_hint}

    # ── Etapa 3: predição de seroclasse (estágio 1) ───────────────────────
    step = "MODEL-CLASS"
    class_probs = {}
    for class_idx in range(1, 5):
        model_path = os.path.join(MODEL_DIR, f"brf_model_class_{class_idx}.pkl")
        if not os.path.exists(model_path):
            log.warning(step, f"model file for class {class_idx} not found.",
                        detail=f"path={model_path}")
            continue
        try:
            model   = joblib.load(model_path)
            X_input = format_features(protein_scores, GLOBAL_FEATURES)
            prob    = model.predict_proba(X_input)[0][1]
            class_probs[f"Class {class_idx}"] = prob
        except Exception:
            log.exception(step, f"model for class {class_idx} failed",
                          detail=f"file={os.path.basename(model_path)}")

    if not class_probs:
        return {"error": "No seroclass model is available on the server.",
                "hint": CONTACT_ADMIN}

    log.debug(step, "class probabilities: " +
                    ", ".join(f"{k}={v:.3f}" for k, v in class_probs.items()))

    predicted_class_name = max(class_probs, key=class_probs.get)
    predicted_class_num  = predicted_class_name.split(" ")[1]
    class_confidence     = class_probs[predicted_class_name] * 100
    log.info(step, f"predicted {predicted_class_name} ({class_confidence:.1f}%)")

    # ── Etapa 4: predição de serogrupo (estágio 2) ───────────────────────
    step = "MODEL-SEROGROUP"
    sg_model_pattern = os.path.join(MODEL_DIR, f"brf_model_class{predicted_class_num}_*.pkl")
    sg_model_files   = sorted(glob.glob(sg_model_pattern))
    log.debug(step, f"{len(sg_model_files)} serogroup model file(s) found for class "
                    f"{predicted_class_num}", detail=f"pattern={sg_model_pattern}")

    if not sg_model_files:
        log.error(step, f"no serogroup models found for Class {predicted_class_num}.",
                  detail=f"pattern={sg_model_pattern}")
        return {"error": f"Serogroup models for Class {predicted_class_num} are not "
                         f"available on the server.",
                "hint": CONTACT_ADMIN}

    sg_fallback_features = CLASS_FEATURES[int(predicted_class_num)]

    serogroup_probs = {}
    for sg_path in sg_model_files:
        # nome do serogrupo = parte do nome do arquivo depois de "brf_model_classN_"
        prefix  = f"brf_model_class{predicted_class_num}_"
        sg_name = os.path.basename(sg_path)[len(prefix):-len(".pkl")]
        try:
            model = joblib.load(sg_path)
            if hasattr(model, "feature_names_in_"):
                feats = list(model.feature_names_in_)
            else:
                feats = sg_fallback_features
                log.debug(step, f"{sg_name}: model has no feature_names_in_, "
                                f"using the class feature table")
            X_input = format_features(protein_scores, feats)
            prob    = model.predict_proba(X_input)[0][1]
            serogroup_probs[sg_name] = prob
        except Exception:
            log.exception(step, f"model for serogroup '{sg_name}' failed",
                          detail=f"file={os.path.basename(sg_path)}")

    if not serogroup_probs:
        return {"error": "All serogroup models failed.",
                "hint": CONTACT_ADMIN}

    predicted_sg  = max(serogroup_probs, key=serogroup_probs.get)
    sg_confidence = serogroup_probs[predicted_sg] * 100

    # Top-3 para mostrar na tabela de resultados
    top3 = sorted(serogroup_probs.items(), key=lambda x: x[1], reverse=True)[:3]
    log.info(step, f"predicted serogroup {predicted_sg} ({sg_confidence:.1f}%); top 3: " +
                   ", ".join(f"{sg}={p * 100:.1f}%" for sg, p in top3))

    return {
        "predicted_class":      predicted_class_num,
        "class_confidence":     round(class_confidence, 1),
        "predicted_serogroup":  predicted_sg,
        "serogroup_confidence": round(sg_confidence, 1),
        "top3_serogroups":      [(sg, round(p * 100, 1)) for sg, p in top3],
        "elapsed_seconds":      round(time.time() - start, 2),
        "protein_scores":       protein_scores,   # usado para projetar a amostra no PCA
    }


# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────

@app.post("/analyze", response_class=HTMLResponse)
async def analyze(file: UploadFile = File(...)):
    job_id = uuid.uuid4().hex
    log    = JobLog(job_id)
    step   = "UPLOAD"
    # O nome do arquivo do usuário pode conter IDs de amostra/paciente:
    # fica só no terminal (detail), nunca na página.
    log.info(step, "file received",
             detail=f"name={file.filename!r} content_type={file.content_type!r}")

    # Extensão checada sem diferenciar maiúsculas (.FNA também vale)
    if not file.filename or not file.filename.lower().endswith(ALLOWED_EXTENSIONS):
        log.error(step, "invalid file extension", detail=f"name={file.filename!r}")
        return HTMLResponse(html_pages.build_error_page(
            "Error: invalid format",
            "Please upload a .fa, .fasta or .fna file.",
            None, log), status_code=400)

    # Pasta temporária isolada por job (UUID garante que dois uploads
    # simultâneos não sobrescrevam arquivos um do outro)
    job_dir = os.path.join(TEMP_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    try:
        safe_name = make_safe_file_name(file.filename)
        if safe_name != file.filename:
            log.debug(step, "file name sanitized",
                      detail=f"{file.filename!r} -> {safe_name!r}")
        file_path = os.path.join(job_dir, safe_name)

        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        log.debug(step, f"file saved ({len(content):,} bytes)", detail=f"path={file_path}")

        # ── Validação do FASTA ────────────────────────────────────────────
        step = "VALIDATION"
        validation_error, validation_hint = validate_genome_fasta(file_path, log)
        if validation_error:
            log.error(step, validation_error)
            return HTMLResponse(html_pages.build_error_page(
                "Error: invalid genome file", validation_error, validation_hint, log),
                status_code=400)

        # ── Análises (rodam em thread para não travar o servidor) ─────────
        step = "FASTANI"
        ani_result = await run_in_threadpool(run_fastani, file_path, log)

        # Erro fatal no FastANI
        if "error" in ani_result:
            log.error(step, ani_result["error"])
            return HTMLResponse(html_pages.build_error_page(
                "Error (FastANI)", ani_result["error"], ani_result.get("hint"), log),
                status_code=500)

        step = "SEROGROUP"
        sg_work_dir = os.path.join(job_dir, "sg_work")
        os.makedirs(sg_work_dir, exist_ok=True)
        sg_result = await run_in_threadpool(run_serogroup_model, file_path, sg_work_dir, log)
        if "error" in sg_result:
            log.error(step, sg_result["error"])

        # ── Montagem da página ────────────────────────────────────────────
        step = "RENDER"
        sg_html  = html_pages.build_serogroup_html(sg_result)
        pca_html = html_pages.build_pca_html(sg_result, PCAS, log)
        return HTMLResponse(html_pages.build_results_page(ani_result, sg_html, pca_html, log))

    except Exception:
        # Erro NÃO previsto: traceback completo só no terminal.
        # A página recebe apenas uma mensagem genérica + o código do job.
        log.exception(step, "unexpected error")
        return HTMLResponse(html_pages.build_error_page(
            "Unexpected error",
            "An internal error occurred while processing your file.",
            f"Please try again. If the problem persists, contact the administrator "
            f"and mention the job code {job_id[:8]}.", log),
            status_code=500)

    finally:
        # Sempre limpa os arquivos temporários, mesmo se der erro
        shutil.rmtree(job_dir, ignore_errors=True)


# Página com os 5 PCAs de referência (sem precisar enviar genoma)
@app.get("/pca", response_class=HTMLResponse)
def pca_page():
    return html_pages.build_pca_reference_page(PCAS)


@app.get("/", response_class=HTMLResponse)
def home():
    return html_pages.HOME_PAGE