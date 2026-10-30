"""
PCA dos sorogrupos de Leptospira (geral + por seroclasse).

Uso no webservice:
    import geradorPCA
    PCAS = geradorPCA.carregar_todos(FEATURES_DIR)   # uma vez, na inicialização
    png_b64 = PCAS["geral"].grafico(protein_scores)   # com a amostra do usuário

Todos os erros previstos levantam PCAError, sempre com uma mensagem clara.

PRIVACIDADE: as mensagens de PCAError NÃO trazem caminhos do servidor nem o
texto da exceção original. O detalhe técnico continua disponível no terminal,
porque cada erro é encadeado com "raise ... from exc" (o traceback mostra a
causa completa).
"""
import io
import os
import base64
import threading

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

COL_SOROGRUPO = "True_serogroup"
COL_CLASSE = "class"
COLUNAS_NAO_FEATURE = {"Unnamed: 0", COL_SOROGRUPO, COL_CLASSE, "Serogroup"}

_MARCADORES = ["o", "s", "^", "D", "v", "P", "X", "<", ">"]

# Estilo das seroclasses (igual à figura do artigo)
_ROMANOS = {"1": "I", "2": "II", "3": "III", "4": "IV"}
_ESTILO_CLASSE = {
    "1": ("#66c2a5", "o"),   # verde-água, círculo
    "2": ("#fc8d62", "s"),   # laranja, quadrado
    "3": ("#8da0cb", "D"),   # azul-lavanda, losango
    "4": ("#e78ac3", "^"),   # rosa, triângulo
}
# Escala padrão dos gráficos (mesma para o geral e para cada classe).
# Os eixos nunca ficam menores que isso; se algum ponto (ou a amostra enviada)
# cair fora, o eixo é ampliado só o suficiente para ele aparecer.
LIM_X = (-600, 200)
LIM_Y = (-400, 400)
PASSO_TICKS = 200

_lock = threading.Lock()  # matplotlib (pyplot) não é thread-safe


class PCAError(Exception):
    """Erro previsto na geração/projeção do PCA (a mensagem já é amigável)."""


class PCAModelo:
    def __init__(self, df, titulo, padronizar=False, n_comp=2, colorir_padrao="sorogrupo"):
        self.titulo = titulo
        self.colorir_padrao = colorir_padrao

        # ---- validações da tabela ----------------------------------------
        if df is None or df.empty:
            raise PCAError(f"[{titulo}] the feature table is empty.")
        if df.shape[1] < 2:
            raise PCAError(f"[{titulo}] the feature table has no usable columns.")
        if COL_SOROGRUPO not in df.columns:
            raise PCAError(f"[{titulo}] column '{COL_SOROGRUPO}' is missing from the TSV.")

        self.ids = df.iloc[:, 0].astype(str).values
        self.sorogrupo = df[COL_SOROGRUPO].astype(str).values
        self.classe = df[COL_CLASSE].astype(str).values if COL_CLASSE in df.columns else None

        X = df.drop(columns=[c for c in COLUNAS_NAO_FEATURE if c in df.columns])
        X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        self.features = list(X.columns)          # mantém todas p/ alinhar com a amostra nova

        if X.shape[0] < 2 or X.shape[1] < 2:
            raise PCAError(f"[{titulo}] PCA needs at least 2 samples and 2 features "
                           f"(found {X.shape[0]} samples x {X.shape[1]} features).")

        # ---- ajuste do PCA -----------------------------------------------
        try:
            self.scaler = StandardScaler() if padronizar else None
            Xv = self.scaler.fit_transform(X) if self.scaler else X.values
            self.pca = PCA(n_components=min(n_comp, *X.shape), random_state=0)
            self.scores = self.pca.fit_transform(Xv)
        except Exception as exc:
            raise PCAError(f"[{titulo}] could not fit the PCA.") from exc

        if self.scores.shape[1] < 2:
            raise PCAError(f"[{titulo}] the PCA produced fewer than 2 components.")
        self.var = self.pca.explained_variance_ratio_
        self._orientar()

    def _orientar(self):
        """
        O sinal de cada componente é arbitrário; fixamos uma orientação para
        os gráficos ficarem sempre iguais (e parecidos com a figura do artigo):
          - geral: seroclasse III à esquerda (PC1) e seroclasse I em cima (PC2)
          - demais: o lado mais longo do PC1 fica à esquerda
        """
        inverter = [False, False]
        if self.classe is not None and "3" in set(self.classe):
            inverter[0] = self.scores[self.classe == "3", 0].mean() > 0
            if "1" in set(self.classe):
                inverter[1] = self.scores[self.classe == "1", 1].mean() < 0
        else:
            inverter[0] = abs(self.scores[:, 0].min()) < abs(self.scores[:, 0].max())
        for i, inv in enumerate(inverter):
            if inv:
                self.scores[:, i] *= -1
                self.pca.components_[i] *= -1   # mantém projetar() consistente

    # ---- projeção de amostra nova ---------------------------------------
    def projetar(self, protein_scores: dict):
        """protein_scores = dict {proteína: score} (saída do run_tblastn)."""
        if not isinstance(protein_scores, dict) or not protein_scores:
            raise PCAError("no protein scores available to project the sample.")
        try:
            x = np.array([[protein_scores.get(f, 0.0) for f in self.features]], dtype=float)
            if not np.isfinite(x).all():
                raise PCAError("the sample has non-numeric (NaN/inf) protein scores.")
            if self.scaler:
                x = self.scaler.transform(x)
            return self.pca.transform(x)[0]
        except PCAError:
            raise
        except Exception as exc:
            raise PCAError(f"could not project the sample onto '{self.titulo}'.") from exc

    # ---- gráfico ----------------------------------------------------------
    def grafico(self, protein_scores=None, colorir_por=None) -> str:
        """
        PNG em base64. Se protein_scores for passado, desenha a amostra como estrela preta.
        colorir_por: "sorogrupo" ou "classe". Se None, usa o padrão do modelo
                     (geral -> classe; por classe -> sorogrupo).
        """
        # projeta ANTES de abrir a figura (se falhar, nada fica aberto)
        p = self.projetar(protein_scores) if protein_scores is not None else None

        colorir_por = colorir_por or self.colorir_padrao
        por_classe = colorir_por == "classe" and self.classe is not None
        rot = self.classe if por_classe else self.sorogrupo
        grupos = sorted(set(rot), key=lambda g: (len(g), g))
        cmap = plt.get_cmap("tab10" if len(grupos) <= 10 else "tab20")

        with _lock:
            fig = None
            try:
                fig, ax = plt.subplots(figsize=(8, 6))
                for i, g in enumerate(grupos):
                    m = rot == g
                    if por_classe:
                        cor, marc = _ESTILO_CLASSE.get(g, ("gray", "o"))
                        nome = f"Seroclass {_ROMANOS.get(g, g)} (n={m.sum()})"
                        ax.scatter(self.scores[m, 0], self.scores[m, 1], color=cor, marker=marc,
                                   s=50, alpha=0.85, edgecolors="black", linewidths=0.5, label=nome)
                    else:
                        ax.scatter(self.scores[m, 0], self.scores[m, 1],
                                   color=cmap(i % cmap.N),
                                   marker=_MARCADORES[(i // cmap.N) % len(_MARCADORES)],
                                   s=45, alpha=0.85, edgecolors="black", linewidths=0.5,
                                   label=f"{g} (n={m.sum()})")
                if p is not None:
                    ax.scatter(p[0], p[1], marker="*", s=400, c="black", edgecolors="yellow",
                               linewidths=1.2, label="Your sample", zorder=10)
                # escala: mínimo = LIM_X/LIM_Y, ampliada se houver pontos fora
                pts = self.scores[:, :2] if p is None else np.vstack([self.scores[:, :2], p[:2]])
                ax.set_xlim(*self._limite(LIM_X, pts[:, 0]))
                ax.set_ylim(*self._limite(LIM_Y, pts[:, 1]))
                ax.set_xticks(self._ticks(ax.get_xlim()))
                ax.set_yticks(self._ticks(ax.get_ylim()))
                ax.set_xlabel(f"PC1 ({self.var[0]*100:.1f}% variance)")
                ax.set_ylabel(f"PC2 ({self.var[1]*100:.1f}% variance)")
                ax.set_title(self.titulo)
                ax.grid(alpha=0.3)
                ax.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
                fig.tight_layout()
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=130)
            except PCAError:
                raise
            except Exception as exc:
                raise PCAError(f"could not draw '{self.titulo}'.") from exc
            finally:
                if fig is not None:
                    plt.close(fig)      # nunca deixa figura aberta (vazamento de memória)
        return base64.b64encode(buf.getvalue()).decode()

    @staticmethod
    def _limite(base, valores):
        margem = 0.05 * (valores.max() - valores.min() + 1e-9)
        return (min(base[0], valores.min() - margem), max(base[1], valores.max() + margem))

    @staticmethod
    def _ticks(lim):
        ini = int(np.ceil(lim[0] / PASSO_TICKS)) * PASSO_TICKS
        return np.arange(ini, lim[1] + 1, PASSO_TICKS)

    def resumo(self):
        return {"titulo": self.titulo, "n_amostras": len(self.ids),
                "n_features": len(self.features),
                "variancia_explicada": [round(float(v) * 100, 2) for v in self.var]}


def _ler_tsv(features_dir, nome):
    """Lê um TSV de features. As mensagens citam só o NOME do arquivo (nunca o
    caminho completo); a causa original fica no traceback (raise ... from exc)."""
    caminho = os.path.join(features_dir, nome)
    if not os.path.isfile(caminho):
        raise PCAError(f"feature table not found: {nome}")
    try:
        return pd.read_csv(caminho, sep="\t")
    except pd.errors.EmptyDataError as exc:
        raise PCAError(f"feature table is empty: {nome}") from exc
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise PCAError(f"feature table is not a valid TSV: {nome}") from exc
    except OSError as exc:
        raise PCAError(f"cannot read feature table: {nome}") from exc


def carregar_todos(features_dir, padronizar=False):
    """Carrega as 5 tabelas e devolve {"geral": PCAModelo, 1: ..., 2: ..., 3: ..., 4: ...}."""
    pcas = {"geral": PCAModelo(_ler_tsv(features_dir, "general_table_identity_rep.tsv"),
                               "General PCA — seroclasses", padronizar,
                               colorir_padrao="classe")}
    for c in range(1, 5):
        pcas[c] = PCAModelo(_ler_tsv(features_dir, f"table_identity_cl_{c}.tsv"),
                            f"PCA — Seroclass {c}", padronizar)
    return pcas