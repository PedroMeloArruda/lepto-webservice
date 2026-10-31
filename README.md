# Leptospira Species and Serogroup Identification Webservice

A web service that identifies the species of *Leptospira*, predicts its serogroup, and then places it on a Principal Component Analysis (PCA).

Species identification is done using FastANI (Average Nucleotide Identity) against a set of reference genomes downloaded for free from the NCBI page.

Serogroup prediction is made by pre-trained Machine Learning (ML) models, using features extracted from the rfb-locus proteins (BLAST+).

Note: The Machine Learning models can be found on the GitHub page: https://github.com/Spidey2004/Modelo_Leptospira

PCA visualization is made by placing the sample onto reference PCAs.

Note: The general PCA can also be found on the GitHub page: https://github.com/Spidey2004/Modelo_Leptospira

IMPORTANT: The serological classification was trained only on types from clade P1. Results for other clades may be unreliable.

usefull links: 
https://github.com/PedroMeloArruda/lepto-webservice
https://hub.docker.com/r/pedromeloarruda/lepto-webservice

## Table of Contents

1. How it works
2. Contents of the repository
3. Requirements
4. How to run 
5. Usage
6. Understanding the results
7. Implementation details and design decisions
8. Configuration
9. Error handling
10. Limitations
11. Troubleshooting
12. Citations

## 1. How it works

When the user uploads a genome assembly (FASTA DNA sample), the service runs the following pipeline:

```text
Upload -> Input validation -> FastANI (species identification) -> BLAST+ and ML models -> PCA Projection -> HTML reports (visualization)
```

**Validation:** Checks that the file extension is accepted (.fa, .fasta and .fna), and that the file is a nucleotide FASTA.

**FastANI:** The input genome is compared against all reference genomes. The three best matches are saved to be reported to the user.

**Serogroup identification:**

1. A BLAST database is built from the input genome.
2. Reference proteins from the rfb locus are searched against it (using `tblastn`).
3. Four class models predict the seroclass.
4. Within the predicted seroclass, one model per serogroup is evaluated, and the most probable one is chosen.

**PCA:** The protein scores of the samples are projected onto reference PCAs and onto the PCA of the predicted seroclass. The input genome is represented by a black star with yellow borders.

**HTML reports:** Renders everything as an HTML page.

## 2. Contents of the repository

```text
lepto/
    main.py: FastAPI and analysis pipeline
    geradorPCA.py: Reference PCAs, sample projections and plots
    html_pages.py: All HTML pages
    Dockerfile: Image definition
    environment.yml: Conda environment
    .dockerignore: Files not to be included in the Docker build
    .gitignore: Files not to be included in the GitHub push
    data/
        reference_genomes/: 74 reference Leptospira genomes
        models/: Pre-trained ML models
        features/: Feature tables (general + classes 1-4), used as model inputs and to build the PCAs
        multi_fasta/: rfb locus proteins
```

## 3. Requirements

1. Docker, or Conda/Mamba (recommended for running without Docker).
2. Everything else is installed from `environment.yml`:

   - Python
   - FastANI
   - BLAST+
   - FastAPI
   - uvicorn
   - python-multipart
   - pandas
   - numpy
   - joblib
   - scikit-learn (1.6.1)
   - imbalanced-learn (0.13.0)
   - matplotlib

## 4. How to run


### Option 1: Docker (recommended)

The image contains the code, all dependencies (including FastANI and BLAST+) and the reference data, so nothing else needs to be installed or downloaded.

```bash
docker pull pedromeloarruda/lepto-webservice:latest
docker run --rm -p 8000:8000 pedromeloarruda/lepto-webservice:latest
```

Then open <http://localhost:8000> in your browser. The first seconds are slower because the reference PCAs are built at start-up. The image is large because it includes the data, so it may take a while to download it.

To run it in the background and read the logs (the logs hold the details of any error, searchable by job code):

```bash
docker run -d --name lepto -p 8000:8000 pedromeloarruda/lepto-webservice:latest
docker logs -f lepto
docker stop lepto
```

To use other data instead of the image mount your own foldder at `/data`:

```bash
docker run --rm -p 8000:8000 -v /path/to/data:/data:ro pedromeloarruda/lepto-webservice:latest
```

Build yourself the image from the source code

```bash
git clone https://github.com/PedroMeloArruda/lepto-webservice.git
cd lepto-webservice
docker build -t lepto-webservice .
docker run --rm -p 8000:8000 lepto-webservice
```

### Option 2: locally with Conda/Mamba (without Docker)

```bash
git clone https://github.com/PedroMeloArruda/lepto-webservice.git
cd lepto-webservice
conda env create -n lepto -f environment.yml
conda activate lepto
uvicorn main:app --host 0.0.0.0 --port 8000
```

The data is already in `./data` after cloning. To use another folder, set `LEPTO_DATA_DIR`

```bash
export LEPTO_DATA_DIR=/path/to/data
```

### Check that it works

- Open <http://localhost:8000> and upload a genome (`.fna`, `.fasta` or `.fa`).
- Open <http://localhost:8000/pca> to see the reference PCAs without uploading anything.
- Or use the command line:

```bash
curl -X POST -F "file=@my_genome.fna" http://localhost:8000/analyze -o report.html
```

Stop the service with `Ctrl+C`.

## 5. Usage

**Web interface:**

- Open the home page.
- Choose a genome assembly (.fna, .fasta, .fa).
- Read the report.

**Endpoints:**

- `/`: Home page for uploads.
- `/analyze`: Receives a genome and returns the HTML report.
- `/pca`: Reference PCAs (general and seroclasses 1-4).

**Input requirements:**

- FASTA nucleotide genome assembly (not compressed).
- Accepted extensions: .fa, .fasta and .fna (case-insensitive).
- The first line must start with > (standard for all accepted extensions).

## 6. Understanding the results

**Species identification:** Closest reference genome, species, ANI, mapped/total fragments, and the three closest genomes. A green badge means the species was identified (ANI >= 95%), while a red badge means it was not identified.

**Serogroup prediction:** Predicted seroclass and serogroup (each shown with a confidence value), plus the top 3 serogroup candidates. Badge index: green for >=70%, orange for 40%-70% and red for <40%.

**PCA:** Two plots with the input represented by a black star with a yellow edge: general PCA and predicted class PCA. The axes label the variance explained by PC1 and PC2.

**Debug log:** Collapsible log, shown only when a warning or error occurs. Contains safe messages and the job code.

Visit the ML GitHub page to understand more about the ML models.

## 7. Implementation details and design decisions

### main.py

**Isolation and cleaning:** Every request gets a unique `job_id` (UUID) and its own temporary folder. This is necessary so that simultaneous requests cannot overwrite each other's files. The folder is always deleted, even in case of errors (`finally` block).

**Privacy protection:** In the job log, the text is split into two categories:

- Message: Safe text, printed on the terminal and shown on the page.
- Detail: Sensitive text (server paths, uploaded file names, exceptions), printed only on the server terminal.

Raw errors may show sensitive server information, so users can only see a clear message and a short job code. Using the job code, an admin can see the full details.

**Preloading on start-up:** Feature names and the five PCAs are loaded ONCE when the server starts, not on every request.

**FASTA validation:** `validate_genome_fasta` reads the file line by line and checks that the file is non-empty, starts with '>', contains sequences and is mostly nucleotides.

Warns for genomes under 1 MB.

**Species identification:** The paths to all genomes are written to a list (`refs.txt`) and passed to a single FastANI call (`--rl`), which uses 8 threads. Results are sorted by ANI, and the top 3 are kept. The species name is acquired from the FASTA header only for the top 3. >= 95% is the cut-off for the species boundary.

**When FastANI returns nothing:** When no reference reaches a minimum ANI, FastANI returns no result. When this happens, the service:

- Checks every reference file (empty, gzip-compressed, not FASTA, protein).
- Compares the input genome against itself. If it fails, the input file is the problem; if it works, the genome is probably unrelated to Leptospira.

**Serogroup prediction:**

**Why BLAST:** The ML models use features describing similarity to rfb locus proteins. `tblastn` searches those proteins against the genome, so the input genome needs no gene annotation.

**Protein score:** For each protein, the best hit is scored as:

`percent_identity * (aligned_length / protein_length)`

A high score means the protein is present with high identity and good coverage. 0 means absent.

**format_features:** Builds a one-row table with the columns in exactly the order that the models expect (missing proteins = 0.0). Necessary because it is a scikit-learn model.

**check_feature_name_match:** If no BLAST protein ID matches the feature columns, the model would receive only zeros, so the service stops (configuration error) instead of making a useless prediction.

**Two stages:** Stage 1 chooses among 4 seroclasses. Stage 2 chooses the serogroup within the chosen seroclass.

**Stage 2 features:** If a model stores `feature_names_in_`, those columns are used; otherwise, the class feature table is used.

**Responsiveness:** The analyses block while external programs run, so they are executed with `run_in_threadpool`. The server keeps answering other requests (`/pca`) during long analyses.

**Partial results:** If FastANI fails, an error page is returned. If serogroup prediction (or only PCA) fails, the ANI result is still shown, together with an error message.

**Caching:** The `/pca` page is generated once and kept in memory (`lru_cache`).

### geradorPCA.py

**PCAModelo:** Wraps one reference PCA (2 components, `random_state = 0`). Standardization is available, but OFF BY DEFAULT.

**Validation of tables:** Errors (missing files, invalid TSV, small number of samples) raise `PCAError` with a message that never contains server paths. The original cause stays in the server logs.

**Fixed orientation (`_orientar`):** The sign of a principal component is arbitrary, so the orientation is fixed to keep plots identical between runs (seroclass III on the left and I on top).

**Projection (`projetar`):** The sample protein scores are aligned with the PCA feature list (missing = 0) and transformed with the same PCA.

**Plots (`grafico`):** Standard X-axis ranges from -600 to 200, and Y-axis from -400 to 400. The axes are expanded if needed. Figures are drawn under a lock (Matplotlib pyplot is NOT THREAD-SAFE). Always closed afterward to avoid memory leaks, and returns a base64 PNG so no files are written.

**carregar_todos:** Builds the five PCAs once at startup.

### html_pages.py

- All HTML is here to keep things "clean".
- Protection against HTML injection: Every value that comes from user input is escaped with `html.escape`.
- Images are embedded as base64 so the report is a single self-contained page.
- Blocks are built separately (`build_ani_top3_html`, `build_serogroup_html`, etc.) so one failure does not hide the others.

### Dockerfile and environment.yml

- Based on `mambaorg/micromamba` because FastANI and BLAST+ come from Bioconda.
- Dependencies are built before the code to avoid unnecessary reinstallation.
- Data is baked into the image (it can be overridden by mounting a folder at `/data`).

## 8. Configuration

**`LEPTO_DATA_DIR`** (environment variable): Default: `<folder of main.py>/data`, location of the data directory (genomes, models, multi_fasta, features).

**`FASTANI`** (constant in `main.py`): Default: `fastANI`, FastANI executable.

**`THREADS`** (constant in `main.py`): Default: 8, threads used by FastANI.

**`ALLOWED_EXTENSIONS`** (constant in `main.py`): Default: .fa, .fna, .fasta. Accepted input extensions.

Temporary job folders are created under the system temporary directory `lepto_jobs/`.

## 9. Error handling

- 400: Invalid extension or invalid/non-nucleotide FASTA.
- 500: FastANI failure or unexpected internal error.

Pages show a short, safe message. For server-side problems, the user is asked to contact the administrator and quote the job code shown on the page. The full details are in the server log under that code.

## 10. Limitations

- The serological classification was trained only on types from clade P1. Results for other clades (P2, S1, S2) may be unreliable.
- Species can only be identified among the genomes in `reference_genomes/`.
- Serogroup prediction covers only the 22 serogroups trained with models and requires the rfb locus to be present and well assembled.
- Small, fragmented, or low-quality inputs may produce no FastANI result or unreliable predictions.
- Results should always be double-checked if possible.

## 11. Troubleshooting

- "Feature table not found": Data folder is missing or `LEPTO_DATA_DIR` / Docker volume is wrong.
- "The FastANI program is not available on the server": FastANI is not installed or not on PATH.
- "A required program (BLAST+) is not available": `makeblastdb` / `tblastn` is not installed.
- "`reference_genomes/` directory not found": Missing `reference_genomes/` folder.
- "Serogroup prediction is not configured correctly": Protein IDs in `multi_fasta/` do not match feature table columns.
- "Models fail to load": scikit-learn / imbalanced-learn versions differ from those stated in `environment.yml`.
- "No results found" from FastANI: Sample is too divergent, too small, or reference files are invalid (see server log).
- Upload fails with multipart error: python-multipart is not installed.

## 12. Citations

If you use this service, please also cite FastANI (Jain et al., 2018, Nature Communications) and BLAST+ (Camacho et al., 2009, BMC Bioinformatics).

**Authors:**

- Testu Sakamoto
- Edson de Carvalho Ferreira Filho
- Leonardo Cabral Afonso Ferreira
- Maria Raquel Venturim Cosate
- Pedro Melo Arruda