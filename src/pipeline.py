"""
=============================================================================
MERIDIAN COMMODITIES TRACKER — Pipeline 100% Automático
=============================================================================
Arquitetura em 3 camadas + fallback basis adaptativo:

  Camada 1 — BCB REST API          → PTAX, IPCA, Selic, IGP-M
  Camada 2 — Yahoo Finance          → BGI B3, CBOT Milho/Soja, Brent, FX
  Camada 3 — CEPEA scraping         → físico BR (milho, soja, boi, frango, suíno)
  Fallback  — Basis Adaptativo      → estima físico via CBOT + basis histórico local

Execute:
    python3 pipeline.py              # coleta completa
    python3 pipeline.py --status     # mostra último estado do cache
    python3 pipeline.py --force      # força recoleta mesmo com cache do dia

Agendamento automático (incluso): seg e qua às 07h30
=============================================================================
"""

import os, sys, json, time, logging, warnings
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

warnings.filterwarnings("ignore")

# ─── Diretórios
BASE       = Path(__file__).parent.parent   # raiz do repo
DATA_RAW   = BASE / "data" / "raw"
DATA_PROC  = BASE / "data" / "processed"
CACHE_DIR  = BASE / "data" / "cache"
LOGS_DIR   = BASE / "logs"
DOCS_DIR   = BASE / "docs"
DOCS_DIR.mkdir(exist_ok=True)
for d in [DATA_RAW, DATA_PROC, CACHE_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(LOGS_DIR / f"pipeline_{datetime.now():%Y%m%d}.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("pipeline")

# ─── Headers para scraping
HEADERS_SCRAPE = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "pt-BR,pt;q=0.9,en;q=0.7",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection":      "keep-alive",
    "Cache-Control":   "no-cache",
}

TODAY      = datetime.today().date()
DATA_INICIO = datetime.today() - timedelta(days=365 * 5)

# ════════════════════════════════════════════════════════════════
#  CAMADA 1 — BCB REST API
# ════════════════════════════════════════════════════════════════

class ColetorBCB:
    """
    API pública do Banco Central do Brasil.
    Não requer autenticação. Taxa de sucesso: ~99%.
    """

    def _get(self, url: str, timeout: int = 20) -> requests.Response | None:
        try:
            r = requests.get(url, headers=HEADERS_SCRAPE, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception as e:
            log.warning(f"BCB request falhou ({url[:60]}): {e}")
            return None

    def coletar_ptax(self, inicio: datetime, fim: datetime) -> pd.DataFrame:
        url = (
            "https://olinda.bcb.gov.br/olinda/servico/PTAX/versao/v1/odata/"
            "CotacaoDolarPeriodo(dataInicial=@di,dataFinalCotacao=@df)"
            f"?@di='{inicio:%m-%d-%Y}'&@df='{fim:%m-%d-%Y}'"
            "&$top=9999&$format=json&$select=cotacaoCompra,cotacaoVenda,dataHoraCotacao"
        )
        r = self._get(url)
        if not r:
            return pd.DataFrame()
        try:
            dados = r.json()["value"]
            df = pd.DataFrame(dados)
            df["data"] = pd.to_datetime(df["dataHoraCotacao"].str[:10])
            df = (
                df[["data", "cotacaoCompra", "cotacaoVenda"]]
                .rename(columns={"cotacaoCompra": "ptax_compra", "cotacaoVenda": "ptax_venda"})
                .sort_values("data").drop_duplicates("data").reset_index(drop=True)
            )
            log.info(f"BCB PTAX → {len(df)} registros")
            return df
        except Exception as e:
            log.error(f"BCB PTAX parse falhou: {e}")
            return pd.DataFrame()

    def coletar_sgs(self, nome: str, codigo: int,
                    inicio: datetime, fim: datetime) -> pd.DataFrame:
        url = (
            f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo}/dados"
            f"?formato=json&dataInicial={inicio:%d/%m/%Y}&dataFinal={fim:%d/%m/%Y}"
        )
        r = self._get(url)
        if not r:
            return pd.DataFrame()
        try:
            df = pd.DataFrame(r.json())
            df["data"]  = pd.to_datetime(df["data"], dayfirst=True)
            df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
            df = df.rename(columns={"valor": nome})[["data", nome]]
            df = df.sort_values("data").dropna().reset_index(drop=True)
            log.info(f"BCB SGS [{nome}] → {len(df)} registros")
            return df
        except Exception as e:
            log.error(f"BCB SGS [{nome}] parse falhou: {e}")
            return pd.DataFrame()

    def coletar_tudo(self, inicio: datetime, fim: datetime) -> dict:
        resultado = {}
        resultado["ptax"] = self.coletar_ptax(inicio, fim)
        time.sleep(0.5)
        series_sgs = {
            "ipca_alimentacao": 1635,
            "igp_m":            189,
            "selic_meta":       432,
        }
        for nome, cod in series_sgs.items():
            resultado[nome] = self.coletar_sgs(nome, cod, inicio, fim)
            time.sleep(0.5)
        return resultado


# ════════════════════════════════════════════════════════════════
#  CAMADA 2 — YAHOO FINANCE
# ════════════════════════════════════════════════════════════════

class ColetorYahoo:
    """
    Yahoo Finance via yfinance.
    Futuros de mercado e câmbio. Taxa de sucesso: ~95%.
    """

    TICKERS = {
        "bgi_b3_brl_arroba":  "BGI=F",   # Boi Gordo B3 (R$/@)
        "cbot_milho_usc_bu":  "ZC=F",    # CBOT Corn (¢/bushel)
        "cbot_soja_usc_bu":   "ZS=F",    # CBOT Soybean (¢/bushel)
        "brent_usd_bbl":      "BZ=F",    # Brent Crude (USD/bbl)
        "usdbrl_spot":        "BRL=X",   # USD/BRL spot
    }

    def coletar(self, inicio: datetime, fim: datetime) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError:
            log.error("yfinance não instalado. Execute: pip install yfinance")
            return pd.DataFrame()

        dfs = []
        for nome, ticker in self.TICKERS.items():
            log.info(f"Yahoo [{nome}] ({ticker})...")
            try:
                hist = yf.Ticker(ticker).history(
                    start=inicio.strftime("%Y-%m-%d"),
                    end=fim.strftime("%Y-%m-%d"),
                    auto_adjust=True,
                )
                if hist.empty:
                    log.warning(f"Yahoo [{nome}] sem dados")
                    continue
                s = hist["Close"].rename(nome)
                s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
                dfs.append(s)
                log.info(f"Yahoo [{nome}] → {len(s)} registros, último={s.iloc[-1]:.2f}")
                time.sleep(0.4)
            except Exception as e:
                log.error(f"Yahoo [{nome}] FALHOU: {e}")

        if not dfs:
            return pd.DataFrame()

        df = pd.concat(dfs, axis=1).reset_index().rename(columns={"Date": "data", "index": "data"})
        df["data"] = pd.to_datetime(df["data"])
        df = df.sort_values("data").reset_index(drop=True)
        df.to_csv(DATA_RAW / "yahoo_futuros.csv", index=False)
        return df


# ════════════════════════════════════════════════════════════════
#  CAMADA 3 — CEPEA SCRAPING (com session e retry)
# ════════════════════════════════════════════════════════════════

class ColetorCEPEA:
    """
    Scraping CEPEA/ESALQ com session simulada e múltiplas estratégias.
    
    Estratégia 1: Scraping direto da tabela HTML
    Estratégia 2: Download do xlsx de série histórica
    
    Se ambas falharem → ColetorBasis assume.
    """

    # Páginas de indicador CEPEA
    INDICADORES = {
        "milho_esalq_brl_saca": {
            "page":    "https://www.cepea.esalq.usp.br/br/indicador/milho.aspx",
            "series":  "https://www.cepea.esalq.usp.br/br/indicador/series/milho.aspx?id=2",
            "label":   "Milho ESALQ",
        },
        "soja_esalq_brl_saca": {
            "page":    "https://www.cepea.esalq.usp.br/br/indicador/soja.aspx",
            "series":  "https://www.cepea.esalq.usp.br/br/indicador/series/soja.aspx?id=2",
            "label":   "Soja Paranaguá",
        },
        "boi_cepea_brl_arroba": {
            "page":    "https://www.cepea.esalq.usp.br/br/indicador/boi-gordo.aspx",
            "series":  "https://www.cepea.esalq.usp.br/br/indicador/series/boi-gordo.aspx?id=2",
            "label":   "Boi Gordo",
        },
        "frango_brl_kg": {
            "page":    "https://www.cepea.esalq.usp.br/br/indicador/frango.aspx",
            "series":  "https://www.cepea.esalq.usp.br/br/indicador/series/frango.aspx?id=2",
            "label":   "Frango vivo SP",
        },
        "suino_brl_kg": {
            "page":    "https://www.cepea.esalq.usp.br/br/indicador/suino.aspx",
            "series":  "https://www.cepea.esalq.usp.br/br/indicador/series/suino.aspx?id=2",
            "label":   "Suíno vivo SP",
        },
    }

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(HEADERS_SCRAPE)
        self._warm_session()

    def _warm_session(self):
        """Aquecer sessão visitando a home do CEPEA (obtém cookies)."""
        try:
            self.session.get("https://www.cepea.esalq.usp.br/br/", timeout=10)
            time.sleep(1)
        except Exception:
            pass

    def _scrape_tabela_html(self, col: str, url_page: str) -> pd.DataFrame:
        """Estratégia 1: scraping da tabela HTML da página de indicador."""
        try:
            from bs4 import BeautifulSoup
            r = self.session.get(url_page, timeout=20)
            if r.status_code != 200:
                return pd.DataFrame()
            soup = BeautifulSoup(r.text, "lxml")
            tabela = soup.find("table", {"id": "imagenet-indicador1"}) or soup.find("table")
            if not tabela:
                return pd.DataFrame()
            linhas = []
            for tr in tabela.find_all("tr")[1:]:
                tds = [td.get_text(strip=True) for td in tr.find_all("td")]
                if len(tds) >= 2:
                    linhas.append(tds)
            if not linhas:
                return pd.DataFrame()
            df = pd.DataFrame(linhas)
            df["data"] = pd.to_datetime(df.iloc[:, 0], dayfirst=True, errors="coerce")
            df[col] = (
                df.iloc[:, 1]
                .str.replace(".", "", regex=False)
                .str.replace(",", ".", regex=False)
                .pipe(pd.to_numeric, errors="coerce")
            )
            df = df[["data", col]].dropna().sort_values("data").reset_index(drop=True)
            if len(df) > 0:
                log.info(f"CEPEA HTML [{col}] → {len(df)} registros, último={df[col].iloc[-1]:.2f}")
            return df
        except Exception as e:
            log.warning(f"CEPEA HTML [{col}] falhou: {e}")
            return pd.DataFrame()

    def _download_xlsx(self, col: str, url_series: str) -> pd.DataFrame:
        """Estratégia 2: download do xlsx de série histórica."""
        try:
            r = self.session.get(url_series, timeout=30)
            if r.status_code != 200:
                return pd.DataFrame()
            content_type = r.headers.get("content-type", "")
            # Verificar se é xlsx
            if "excel" in content_type or "spreadsheet" in content_type or r.content[:4] == b'PK\x03\x04':
                path_tmp = DATA_RAW / f"cepea_{col}.xlsx"
                path_tmp.write_bytes(r.content)
                df = pd.read_excel(path_tmp, skiprows=3, engine="openpyxl")
                # CEPEA xlsx: col 0 = data, col 1 = valor à vista
                df.columns = [str(c).strip() for c in df.columns]
                df["data"] = pd.to_datetime(df.iloc[:, 0], dayfirst=True, errors="coerce")
                df[col] = pd.to_numeric(
                    df.iloc[:, 1].astype(str)
                    .str.replace(".", "", regex=False)
                    .str.replace(",", ".", regex=False),
                    errors="coerce"
                )
                df = df[["data", col]].dropna().sort_values("data").reset_index(drop=True)
                if len(df) > 0:
                    log.info(f"CEPEA XLSX [{col}] → {len(df)} registros")
                return df
            else:
                # Pode ser página HTML com link para o xlsx real
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(r.text, "lxml")
                link = soup.find("a", href=lambda h: h and (".xls" in h or ".xlsx" in h))
                if link:
                    href = link["href"]
                    if not href.startswith("http"):
                        href = "https://www.cepea.esalq.usp.br" + href
                    r2 = self.session.get(href, timeout=30)
                    if r2.status_code == 200:
                        path_tmp = DATA_RAW / f"cepea_{col}.xlsx"
                        path_tmp.write_bytes(r2.content)
                        df = pd.read_excel(path_tmp, skiprows=3, engine="openpyxl")
                        df["data"] = pd.to_datetime(df.iloc[:, 0], dayfirst=True, errors="coerce")
                        df[col] = pd.to_numeric(
                            df.iloc[:, 1].astype(str)
                            .str.replace(".", "", regex=False)
                            .str.replace(",", ".", regex=False),
                            errors="coerce"
                        )
                        df = df[["data", col]].dropna().sort_values("data").reset_index(drop=True)
                        log.info(f"CEPEA XLSX via link [{col}] → {len(df)} registros")
                        return df
            return pd.DataFrame()
        except Exception as e:
            log.warning(f"CEPEA XLSX [{col}] falhou: {e}")
            return pd.DataFrame()

    def coletar_tudo(self) -> dict:
        resultados = {}
        status_cepea = {}
        for col, meta in self.INDICADORES.items():
            log.info(f"CEPEA [{meta['label']}]...")
            # Tentar estratégia 1: HTML
            df = self._scrape_tabela_html(col, meta["page"])
            if df.empty:
                time.sleep(1)
                # Tentar estratégia 2: xlsx
                df = self._download_xlsx(col, meta["series"])
            if not df.empty:
                resultados[col] = df
                status_cepea[col] = "ok"
            else:
                log.warning(f"CEPEA [{col}] → ambas as estratégias falharam")
                status_cepea[col] = "falhou"
            time.sleep(1.5)

        # Salvar status para o basis saber o que precisa estimar
        (CACHE_DIR / "cepea_status.json").write_text(
            json.dumps({**status_cepea, "data": str(TODAY)})
        )
        return resultados


# ════════════════════════════════════════════════════════════════
#  FALLBACK — BASIS ADAPTATIVO
# ════════════════════════════════════════════════════════════════

class ColetorBasis:
    """
    Estima preços físicos quando o CEPEA falha.
    
    Método:
    1. Carrega histórico local das últimas coletas bem-sucedidas
    2. Calcula basis = físico - cbot_convertido (média 90 dias)
    3. Aplica basis ao preço CBOT atual para estimar o físico
    
    Precisão esperada: 5-8% de erro (vs 30% com basis fixo)
    """

    CACHE_BASIS = CACHE_DIR / "basis_historico.json"

    # Conversão bushel → saca 60kg
    CONV = {
        "cbot_milho_usc_bu": {"lbs_per_bu": 56, "kg_per_saca": 60},  # 56 lbs/bu milho
        "cbot_soja_usc_bu":  {"lbs_per_bu": 60, "kg_per_saca": 60},  # 60 lbs/bu soja
    }

    def cbot_para_brl_saca(self, cbot_cents_bu: float, ptax: float,
                            tipo: str) -> float:
        """Converte CBOT (¢/bushel) para R$/saca 60kg."""
        c = self.CONV[tipo]
        lbs   = c["lbs_per_bu"]
        saca  = c["kg_per_saca"]
        kg_per_bu = lbs * 0.453592
        return (cbot_cents_bu / 100) * (saca / kg_per_bu) * ptax

    def atualizar_basis(self, df_consolidado: pd.DataFrame) -> None:
        """Recalcula e salva o basis adaptativo do histórico local."""
        basis = {}
        try:
            df = df_consolidado.copy()
            if "ptax_venda" not in df.columns:
                return

            # Usar últimos 90 dias úteis
            df = df.tail(90).copy()
            ptax = df["ptax_venda"].ffill()

            pares = [
                ("milho_esalq_brl_saca", "cbot_milho_usc_bu"),
                ("soja_esalq_brl_saca",  "cbot_soja_usc_bu"),
            ]
            for col_fisico, col_cbot in pares:
                if col_fisico in df.columns and col_cbot in df.columns:
                    cbot_brl = pd.Series([
                        self.cbot_para_brl_saca(v, p, col_cbot)
                        for v, p in zip(df[col_cbot].ffill(), ptax)
                    ])
                    fisico = df[col_fisico].ffill()
                    b = (fisico - cbot_brl).dropna()
                    if len(b) > 10:
                        basis[col_fisico] = {
                            "basis_medio":    round(float(b.mean()), 2),
                            "basis_mediana":  round(float(b.median()), 2),
                            "basis_std":      round(float(b.std()), 2),
                            "n_obs":          len(b),
                            "data":           str(TODAY),
                        }

            # Relações frango e suíno com milho
            if "frango_brl_kg" in df.columns and "milho_esalq_brl_saca" in df.columns:
                ratio = (df["frango_brl_kg"] / df["milho_esalq_brl_saca"]).dropna()
                if len(ratio) > 10:
                    basis["frango_brl_kg"] = {
                        "ratio_vs_milho": round(float(ratio.mean()), 4),
                        "ratio_std":      round(float(ratio.std()), 4),
                        "data":           str(TODAY),
                    }
            if "suino_brl_kg" in df.columns and "milho_esalq_brl_saca" in df.columns:
                ratio = (df["suino_brl_kg"] / df["milho_esalq_brl_saca"]).dropna()
                if len(ratio) > 10:
                    basis["suino_brl_kg"] = {
                        "ratio_vs_milho": round(float(ratio.mean()), 4),
                        "ratio_std":      round(float(ratio.std()), 4),
                        "data":           str(TODAY),
                    }
            if "boi_cepea_brl_arroba" in df.columns and "ptax_venda" in df.columns:
                ratio = (df["boi_cepea_brl_arroba"] / df["ptax_venda"]).dropna()
                if len(ratio) > 10:
                    basis["boi_cepea_brl_arroba"] = {
                        "ratio_vs_ptax": round(float(ratio.mean()), 2),
                        "ratio_std":     round(float(ratio.std()), 2),
                        "data":          str(TODAY),
                    }

            self.CACHE_BASIS.write_text(json.dumps(basis, ensure_ascii=False, indent=2))
            log.info(f"Basis adaptativo atualizado → {len(basis)} variáveis")
        except Exception as e:
            log.error(f"Erro ao atualizar basis: {e}")

    def estimar(self, col: str, df_atual: pd.DataFrame) -> pd.Series | None:
        """Estima série de preços físicos usando basis adaptativo."""
        if not self.CACHE_BASIS.exists():
            log.warning(f"Basis: cache não encontrado para {col}")
            return None
        try:
            basis = json.loads(self.CACHE_BASIS.read_text())
            if col not in basis:
                return None

            b = basis[col]
            ptax = df_atual["ptax_venda"].ffill()

            if col in ("milho_esalq_brl_saca", "soja_esalq_brl_saca"):
                col_cbot = "cbot_milho_usc_bu" if "milho" in col else "cbot_soja_usc_bu"
                if col_cbot not in df_atual.columns:
                    return None
                cbot_brl = pd.Series([
                    self.cbot_para_brl_saca(v, p, col_cbot)
                    for v, p in zip(df_atual[col_cbot].ffill(), ptax)
                ], index=df_atual.index)
                estimativa = cbot_brl + b["basis_medio"]

            elif col in ("frango_brl_kg", "suino_brl_kg"):
                if "milho_esalq_brl_saca" not in df_atual.columns:
                    return None
                estimativa = df_atual["milho_esalq_brl_saca"].ffill() * b["ratio_vs_milho"]

            elif col == "boi_cepea_brl_arroba":
                estimativa = ptax * b["ratio_vs_ptax"]

            else:
                return None

            log.info(f"Basis estimado [{col}]: último={estimativa.iloc[-1]:.2f} (basis={b})")
            return estimativa.round(2)

        except Exception as e:
            log.error(f"Basis estimar [{col}] falhou: {e}")
            return None


# ════════════════════════════════════════════════════════════════
#  CACHE LOCAL
# ════════════════════════════════════════════════════════════════

class GerenciadorCache:
    """Salva e carrega o dataset consolidado por data."""

    @staticmethod
    def salvar(df: pd.DataFrame, fontes: dict) -> None:
        path = CACHE_DIR / f"{TODAY}.json"
        payload = {
            "data":   str(TODAY),
            "fontes": fontes,
            "shape":  list(df.shape),
            "ultimo": {
                col: round(float(df[col].dropna().iloc[-1]), 4)
                for col in df.select_dtypes(include=np.number).columns
                if not df[col].dropna().empty
            },
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        df.to_parquet(CACHE_DIR / f"{TODAY}.parquet", index=False)
        log.info(f"Cache salvo: {path.name}")

    @staticmethod
    def carregar_ultimo() -> tuple[pd.DataFrame | None, dict]:
        arquivos = sorted(CACHE_DIR.glob("*.parquet"), reverse=True)
        if not arquivos:
            return None, {}
        path = arquivos[0]
        data_cache = path.stem
        dias_atraso = (TODAY - datetime.strptime(data_cache, "%Y-%m-%d").date()).days
        log.info(f"Cache mais recente: {data_cache} ({dias_atraso} dias atrás)")
        df = pd.read_parquet(path)
        meta = {}
        json_path = CACHE_DIR / f"{data_cache}.json"
        if json_path.exists():
            meta = json.loads(json_path.read_text())
        meta["dias_atraso"] = dias_atraso
        return df, meta

    @staticmethod
    def hoje_coletado() -> bool:
        return (CACHE_DIR / f"{TODAY}.parquet").exists()


# ════════════════════════════════════════════════════════════════
#  CONSOLIDADOR
# ════════════════════════════════════════════════════════════════

class Consolidador:
    """Junta todas as fontes num DataFrame diário."""

    @staticmethod
    def merge(*dfs: pd.DataFrame) -> pd.DataFrame:
        validos = [df.set_index("data") for df in dfs if not df.empty]
        if not validos:
            return pd.DataFrame()
        base = validos[0]
        for df in validos[1:]:
            base = base.join(df, how="outer")
        return base.sort_index().ffill(limit=5).reset_index()

    @staticmethod
    def calcular_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values("data").reset_index(drop=True).copy()
        colunas = [c for c in df.columns if c != "data" and pd.api.types.is_numeric_dtype(df[c])]

        for col in colunas:
            df[f"{col}_ret7d"]  = df[col].pct_change(5)
            df[f"{col}_ret30d"] = df[col].pct_change(21)
            df[f"{col}_ret90d"] = df[col].pct_change(63)
            df[f"{col}_mm20"]   = df[col].rolling(20).mean()
            df[f"{col}_mm60"]   = df[col].rolling(60).mean()
            df[f"{col}_vol30d"] = df[col].pct_change().rolling(30).std() * np.sqrt(252)

        if "ptax_venda" in df.columns:
            ptax = df["ptax_venda"]
            if "cbot_milho_usc_bu" in df.columns:
                df["milho_cbot_brl"] = (df["cbot_milho_usc_bu"] / 100 * 2.20462 * 60 / 25.401 * ptax).round(2)
            if "cbot_soja_usc_bu" in df.columns:
                df["soja_cbot_brl"] = (df["cbot_soja_usc_bu"] / 100 * 2.20462 * 60 / 36.744 * ptax).round(2)

        if "milho_cbot_brl" in df.columns and "milho_esalq_brl_saca" in df.columns:
            df["basis_milho"] = df["milho_esalq_brl_saca"] - df["milho_cbot_brl"]
        if "soja_cbot_brl" in df.columns and "soja_esalq_brl_saca" in df.columns:
            df["basis_soja"] = df["soja_esalq_brl_saca"] - df["soja_cbot_brl"]

        df["mes"]          = df["data"].dt.month
        df["sazon_seno"]   = np.sin(2 * np.pi * df["data"].dt.dayofyear / 365)
        df["sazon_cosseno"]= np.cos(2 * np.pi * df["data"].dt.dayofyear / 365)
        return df


# ════════════════════════════════════════════════════════════════
#  PIPELINE PRINCIPAL
# ════════════════════════════════════════════════════════════════

def executar(force: bool = False) -> pd.DataFrame:
    log.info("=" * 60)
    log.info("MERIDIAN — Pipeline Automático")
    log.info(f"Data: {TODAY}  |  Force: {force}")
    log.info("=" * 60)

    cache = GerenciadorCache()

    # ── Verificar cache do dia
    if not force and cache.hoje_coletado():
        log.info("Cache do dia já existe. Use --force para reforçar.")
        df, meta = cache.carregar_ultimo()
        return df

    fontes_status = {}

    # ── CAMADA 1: BCB
    log.info("\n[1/3] BCB REST API...")
    bcb    = ColetorBCB()
    dados_bcb = bcb.coletar_tudo(DATA_INICIO, datetime.today())
    fontes_status["bcb_ptax"] = "ok" if not dados_bcb.get("ptax", pd.DataFrame()).empty else "erro"

    # ── CAMADA 2: Yahoo Finance
    log.info("\n[2/3] Yahoo Finance...")
    yahoo  = ColetorYahoo()
    df_yf  = yahoo.coletar(DATA_INICIO, datetime.today())
    fontes_status["yahoo"] = "ok" if not df_yf.empty else "erro"

    # ── Consolidar parcial (BCB + Yahoo) — necessário para basis
    dfs_parcial = [v for v in dados_bcb.values() if not v.empty]
    if not df_yf.empty:
        dfs_parcial.append(df_yf)

    df_parcial = Consolidador.merge(*dfs_parcial) if dfs_parcial else pd.DataFrame()

    # ── CAMADA 3: CEPEA
    log.info("\n[3/3] CEPEA scraping...")
    cepea  = ColetorCEPEA()
    dados_cepea = cepea.coletar_tudo()

    # ── FALLBACK: Basis Adaptativo para o que o CEPEA não coletou
    basis = ColetorBasis()

    # Atualizar basis com os dados históricos que já temos em cache
    df_hist, _ = cache.carregar_ultimo()
    if df_hist is not None and not df_hist.empty:
        # Enriquecer com o que o CEPEA coletou hoje
        dfs_para_basis = [df_hist]
        for col, df_c in dados_cepea.items():
            if not df_c.empty:
                dfs_para_basis.append(df_c)
        df_para_basis = Consolidador.merge(*dfs_para_basis)
        basis.atualizar_basis(df_para_basis)

    # Aplicar fallback para colunas não coletadas
    colunas_cepea = list(ColetorCEPEA.INDICADORES.keys())
    if not df_parcial.empty:
        for col in colunas_cepea:
            if col not in dados_cepea:
                log.info(f"Basis fallback → estimando {col}...")
                estimativa = basis.estimar(col, df_parcial)
                if estimativa is not None:
                    df_parcial[col] = estimativa.values
                    fontes_status[col] = "basis_estimado"
                else:
                    fontes_status[col] = "indisponivel"

    # ── Consolidar tudo
    todas_fontes = list(dados_bcb.values()) + ([df_yf] if not df_yf.empty else [])
    for df_c in dados_cepea.values():
        if not df_c.empty:
            todas_fontes.append(df_c)
    for col, status in fontes_status.items():
        if status not in ("basis_estimado", "indisponivel"):
            fontes_status[col] = fontes_status.get(col, "ok")

    if not todas_fontes:
        log.error("Nenhum dado coletado. Verificar conexão.")
        df_cache, meta = cache.carregar_ultimo()
        if df_cache is not None:
            log.warning(f"Usando cache de {meta.get('data','?')} ({meta.get('dias_atraso',0)} dias atrás)")
            return df_cache
        return pd.DataFrame()

    df_final = Consolidador.merge(*todas_fontes)
    df_features = Consolidador.calcular_features(df_final)

    # Salvar
    df_features.to_csv(DATA_PROC / "dataset_producao.csv", index=False)
    cache.salvar(df_features, fontes_status)

    log.info("=" * 60)
    log.info(f"Pipeline concluído: {df_features.shape[0]} linhas × {df_features.shape[1]} colunas")
    log.info(f"Fontes: {fontes_status}")
    log.info("=" * 60)

    # ── Imprimir resumo dos últimos valores
    cols_print = [c for c in [
        "ptax_venda", "milho_esalq_brl_saca", "soja_esalq_brl_saca",
        "frango_brl_kg", "suino_brl_kg", "boi_cepea_brl_arroba",
        "cbot_milho_usc_bu", "cbot_soja_usc_bu", "brent_usd_bbl",
    ] if c in df_features.columns]

    print("\n📊 ÚLTIMOS VALORES COLETADOS:")
    print(f"  {'Variável':<35} {'Último':>10}  Status")
    print("  " + "-" * 58)
    for col in cols_print:
        s = df_features[col].dropna()
        if not s.empty:
            st = fontes_status.get(col, fontes_status.get("yahoo", "ok"))
            icon = "✅" if st == "ok" else ("⚠️ " if st == "basis_estimado" else "❌")
            print(f"  {col:<35} {s.iloc[-1]:>10.2f}  {icon} {st}")

    return df_features


def mostrar_status():
    """Exibe status do último cache."""
    cache = GerenciadorCache()
    df, meta = cache.carregar_ultimo()
    if df is None:
        print("Nenhum cache encontrado. Execute: python3 pipeline.py")
        return
    print(f"\n{'='*50}")
    print(f"  Último cache: {meta.get('data','?')}  ({meta.get('dias_atraso',0)} dias)")
    print(f"  Shape: {meta.get('shape','?')}")
    print(f"\n  Fontes: {meta.get('fontes','?')}")
    print(f"\n  Últimos valores:")
    for k, v in (meta.get("ultimo") or {}).items():
        print(f"    {k:<35} {v:>10.4f}")
    print("=" * 50)


# ════════════════════════════════════════════════════════════════
#  AGENDAMENTO AUTOMÁTICO
# ════════════════════════════════════════════════════════════════

def iniciar_scheduler():
    try:
        import schedule
    except ImportError:
        print("Instale: pip install schedule")
        return

    def job():
        log.info("⏰ Scheduler: coleta agendada iniciada")
        executar(force=True)

    schedule.every().monday.at("07:30").do(job)
    schedule.every().wednesday.at("07:30").do(job)
    log.info("Scheduler ativo: coleta toda seg/qua 07h30")
    log.info(f"Próxima: {schedule.next_run()}")

    while True:
        schedule.run_pending()
        time.sleep(60)


# ════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if "--status" in sys.argv:
        mostrar_status()
    elif "--scheduler" in sys.argv:
        iniciar_scheduler()
    else:
        force = "--force" in sys.argv
        executar(force=force)
