"""
MERIDIAN COMMODITIES TRACKER — Pipeline v2
Coleta: BCB + Yahoo Finance + CEPEA Widget (14 produtos)
"""
import os, sys, json, time, logging, warnings
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

warnings.filterwarnings("ignore")

BASE       = Path(__file__).parent.parent
DATA_RAW   = BASE / "data" / "raw"
DATA_PROC  = BASE / "data" / "processed"
CACHE_DIR  = BASE / "data" / "cache"
LOGS_DIR   = BASE / "logs"
for d in [DATA_RAW, DATA_PROC, CACHE_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

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

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/123.0.0.0 Safari/537.36",
    "Accept-Language": "pt-BR,pt;q=0.9",
}
TODAY      = datetime.today().date()
DATA_INICIO = datetime.today() - timedelta(days=365 * 5)

# ══════════════════════════════════════════════════
#  CEPEA WIDGET (nova fonte principal)
# ══════════════════════════════════════════════════

WIDGET_BASE = "https://cepea.org.br/br/widgetproduto.js.php"

INDICADORES_WIDGET = {
    "2":                           ("boi_cepea_brl_arroba",    "Boi Gordo",              "R$/@"),
    "23":                          ("cafe_arabica_brl_sc",     "Café Arábica",           "R$/sc"),
    "24":                          ("cafe_robusta_brl_sc",     "Café Robusta",           "R$/sc"),
    "181":                         ("frango_congelado_brl_kg", "Frango Congelado",       "R$/kg"),
    "130":                         ("frango_resfriado_brl_kg", "Frango Resfriado",       "R$/kg"),
    "92":                          ("soja_paranagua_brl_sc",   "Soja Paranaguá",         "R$/sc"),
    "12":                          ("soja_pr_brl_sc",          "Soja PR",                "R$/sc"),
    "159-Bastos+(SP)+-+FOB-branco":("ovos_brancos_brl_cx",    "Ovos Brancos Bastos SP", "R$/cx"),
    "129-1":                       ("suino_sp_brl_kg",         "Suíno SP",               "R$/kg"),
    "178":                         ("trigo_pr_brl_t",          "Trigo PR",               "R$/t"),
    "leitep":                      ("leite_brl_litro",         "Leite",                  "R$/litro"),
    "77":                          ("milho_brl_sc",            "Milho ESALQ",            "R$/sc"),
    "381-1":                       ("feijao_carioca_brl_sc",   "Feijão Carioca SP",      "R$/sc"),
    "54":                          ("algodao_brl_lp",          "Algodão",                "¢R$/lp"),
    "91":                          ("arroz_brl_sc",            "Arroz IRGA-RS",          "R$/sc"),
    "53":                          ("acucar_brl_sc",           "Açúcar Cristal SP",      "R$/sc"),
}

IDS_WIDGET = list(INDICADORES_WIDGET.keys())

# CEPEA bloqueia servidores cloud — tentamos múltiplos Referers
WIDGET_HEADERS_LIST = [
    {**HEADERS, "Referer": "https://www.noticiasagricolas.com.br/", "Accept": "*/*"},
    {**HEADERS, "Referer": "https://www.agrolink.com.br/", "Accept": "*/*"},
    {**HEADERS, "Referer": "https://www.farmnews.com.br/", "Accept": "*/*"},
    {**HEADERS, "Referer": "https://www.cepea.org.br/", "Accept": "*/*"},
]
WIDGET_HEADERS = WIDGET_HEADERS_LIST[0]


class ColetorCEPEAWidget:
    def _valor_float(self, v: str) -> float:
        import re
        v = re.sub(r'[R\$¢\s]', '', v).strip()
        v = v.replace(".", "").replace(",", ".")
        return float(v)

    def _montar_url(self) -> str:
        ids_str = "&".join([f"id_indicador[]={id_}" for id_ in IDS_WIDGET])
        return (f"{WIDGET_BASE}?fonte=arial&tamanho=10&largura=400px"
                f"&corfundo=ffffff&cortexto=333333&corlinha=eeeeee&{ids_str}")

    def _fetch_direto(self, url: str) -> str | None:
        """Tenta acesso direto com múltiplos headers."""
        for i, headers in enumerate(WIDGET_HEADERS_LIST):
            try:
                r = requests.get(url, headers=headers, timeout=20)
                r.raise_for_status()
                if len(r.text) > 200:
                    log.info(f"CEPEA direto → {len(r.text)} bytes (header {i+1})")
                    return r.text
            except Exception as e:
                log.warning(f"CEPEA direto tentativa {i+1}: {e}")
                time.sleep(1)
        return None

    def _fetch_scraperapi(self, url: str) -> str | None:
        """Usa ScraperAPI como proxy residencial (fallback)."""
        import os
        from urllib.parse import quote
        api_key = os.environ.get("SCRAPER_API_KEY", "")
        if not api_key:
            log.warning("SCRAPER_API_KEY não configurada — pulando ScraperAPI")
            return None
        proxy_url = (f"http://api.scraperapi.com"
                     f"?api_key={api_key}"
                     f"&country_code=br"
                     f"&render=false"
                     f"&url={quote(url)}")
        try:
            r = requests.get(proxy_url, timeout=30)
            r.raise_for_status()
            if len(r.text) > 200:
                log.info(f"CEPEA via ScraperAPI → {len(r.text)} bytes ✅")
                return r.text
        except Exception as e:
            log.error(f"ScraperAPI falhou: {e}")
        return None

    def coletar(self) -> dict:
        import re
        url = self._montar_url()

        # Tentativa 1: acesso direto
        log.info("CEPEA Widget → tentando acesso direto...")
        js = self._fetch_direto(url)

        # Tentativa 2: ScraperAPI (proxy residencial)
        if not js:
            log.info("CEPEA Widget → tentando ScraperAPI...")
            js = self._fetch_scraperapi(url)

        if not js:
            log.error("CEPEA Widget indisponível — dashboard usará histórico CSV")
            return {}

        # Salvar JS bruto para debug
        (DATA_RAW / "cepea_widget_raw.js").write_text(js, encoding="utf-8")

        # Extrair datas e valores na ordem
        datas  = re.findall(r"(\d{2}/\d{2}/\d{4})", js)
        valores = re.findall(r"(R\$\s*[\d.]+,\d{2}|¢R\$\s*[\d.]+,\d{2})", js)

        resultados = {}
        for i, id_ in enumerate(IDS_WIDGET):
            if id_ not in INDICADORES_WIDGET:
                continue
            col, nome, unid = INDICADORES_WIDGET[id_]
            if i < len(valores):
                try:
                    v = self._valor_float(valores[i])
                    d = datas[i] if i < len(datas) else str(TODAY)
                    resultados[col] = {"valor": v, "data": d, "nome": nome, "unidade": unid}
                    log.info(f"  {nome}: {v} {unid} ({d})")
                except Exception as ex:
                    log.warning(f"  {nome}: erro parse — {ex}")

        # Salvar JSON do resultado
        (CACHE_DIR / "cepea_widget_ultimo.json").write_text(
            json.dumps(resultados, ensure_ascii=False, indent=2)
        )
        log.info(f"CEPEA Widget → {len(resultados)}/{len(IDS_WIDGET)} produtos coletados")
        return resultados

    def para_dataframe(self, resultados: dict) -> pd.DataFrame:
        if not resultados:
            return pd.DataFrame()
        row = {"data": pd.Timestamp(TODAY)}
        for col, info in resultados.items():
            row[col] = info["valor"]
        return pd.DataFrame([row])


# ══════════════════════════════════════════════════
#  BCB REST API
# ══════════════════════════════════════════════════

class ColetorBCB:
    def _get(self, url):
        try:
            r = requests.get(url, headers=HEADERS, timeout=20)
            r.raise_for_status()
            return r
        except Exception as e:
            log.warning(f"BCB request falhou: {e}")
            return None

    def coletar_ptax(self, inicio, fim):
        url = (
            "https://olinda.bcb.gov.br/olinda/servico/PTAX/versao/v1/odata/"
            f"CotacaoDolarPeriodo(dataInicial=@di,dataFinalCotacao=@df)"
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
            df = (df[["data","cotacaoCompra","cotacaoVenda"]]
                  .rename(columns={"cotacaoCompra":"ptax_compra","cotacaoVenda":"ptax_venda"})
                  .sort_values("data").drop_duplicates("data").reset_index(drop=True))
            log.info(f"BCB PTAX → {len(df)} registros")
            return df
        except Exception as e:
            log.error(f"BCB PTAX parse falhou: {e}")
            return pd.DataFrame()

    def coletar_sgs(self, nome, codigo, inicio, fim):
        url = (f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo}/dados"
               f"?formato=json&dataInicial={inicio:%d/%m/%Y}&dataFinal={fim:%d/%m/%Y}")
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
            log.error(f"BCB SGS [{nome}] falhou: {e}")
            return pd.DataFrame()

    def coletar_tudo(self, inicio, fim):
        resultado = {"ptax": self.coletar_ptax(inicio, fim)}
        time.sleep(0.5)
        for nome, cod in [("ipca_alimentacao",1635),("igp_m",189),("selic_meta",432)]:
            resultado[nome] = self.coletar_sgs(nome, cod, inicio, fim)
            time.sleep(0.5)
        return resultado


# ══════════════════════════════════════════════════
#  YAHOO FINANCE
# ══════════════════════════════════════════════════

class ColetorYahoo:
    TICKERS = {
        "cbot_milho_usc_bu": "ZC=F",
        "cbot_soja_usc_bu":  "ZS=F",
        "brent_usd_bbl":     "BZ=F",
        "usdbrl_spot":       "BRL=X",
    }

    def coletar(self, inicio, fim):
        try:
            import yfinance as yf
        except ImportError:
            log.error("yfinance não instalado.")
            return pd.DataFrame()

        dfs = []
        for nome, ticker in self.TICKERS.items():
            log.info(f"Yahoo [{nome}] ({ticker})...")
            try:
                hist = yf.Ticker(ticker).history(
                    start=inicio.strftime("%Y-%m-%d"),
                    end=fim.strftime("%Y-%m-%d"),
                    auto_adjust=True
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
        df = pd.concat(dfs, axis=1).reset_index().rename(columns={"Date":"data","index":"data"})
        df["data"] = pd.to_datetime(df["data"])
        return df.sort_values("data").reset_index(drop=True)


# ══════════════════════════════════════════════════
#  CONSOLIDADOR + FEATURES
# ══════════════════════════════════════════════════

class Consolidador:
    @staticmethod
    def merge(*dfs):
        validos = [df.set_index("data") for df in dfs if not df.empty]
        if not validos:
            return pd.DataFrame()
        base = validos[0]
        for df in validos[1:]:
            base = base.join(df, how="outer")
        return base.sort_index().ffill(limit=5).reset_index()

    @staticmethod
    def features(df):
        df = df.sort_values("data").reset_index(drop=True).copy()
        cols = [c for c in df.columns if c != "data" and pd.api.types.is_numeric_dtype(df[c])]
        for col in cols:
            df[f"{col}_var7d"]  = df[col].pct_change(5)
            df[f"{col}_var30d"] = df[col].pct_change(21)
            df[f"{col}_var90d"] = df[col].pct_change(63)
            df[f"{col}_mm20"]   = df[col].rolling(20).mean()
            df[f"{col}_mm60"]   = df[col].rolling(60).mean()
            df[f"{col}_vol30d"] = df[col].pct_change().rolling(30).std() * np.sqrt(252)
        if "ptax_venda" in df.columns:
            ptax = df["ptax_venda"]
            if "cbot_milho_usc_bu" in df.columns:
                df["milho_cbot_brl"] = (df["cbot_milho_usc_bu"]/100 * 2.20462*60/25.401 * ptax).round(2)
            if "cbot_soja_usc_bu" in df.columns:
                df["soja_cbot_brl"] = (df["cbot_soja_usc_bu"]/100 * 2.20462*60/36.744 * ptax).round(2)
        df["mes"]          = df["data"].dt.month
        df["sazon_seno"]   = np.sin(2 * np.pi * df["data"].dt.dayofyear / 365)
        df["sazon_cosseno"]= np.cos(2 * np.pi * df["data"].dt.dayofyear / 365)
        return df


# ══════════════════════════════════════════════════
#  CACHE
# ══════════════════════════════════════════════════

class Cache:
    @staticmethod
    def salvar(df, fontes):
        path = CACHE_DIR / f"{TODAY}.json"
        payload = {
            "data": str(TODAY), "fontes": fontes, "shape": list(df.shape),
            "ultimo": {
                col: round(float(df[col].dropna().iloc[-1]), 4)
                for col in df.select_dtypes(include=np.number).columns
                if not df[col].dropna().empty
            },
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        try:
            df.to_parquet(CACHE_DIR / f"{TODAY}.parquet", index=False)
        except Exception as e:
            log.warning(f"Parquet não salvo: {e}")
        log.info(f"Cache salvo: {path.name}")

    @staticmethod
    def hoje_coletado():
        return (CACHE_DIR / f"{TODAY}.parquet").exists()

    @staticmethod
    def carregar_ultimo():
        arquivos = sorted(CACHE_DIR.glob("*.parquet"), reverse=True)
        if not arquivos:
            return None, {}
        df = pd.read_parquet(arquivos[0])
        meta = {}
        jp = CACHE_DIR / f"{arquivos[0].stem}.json"
        if jp.exists():
            meta = json.loads(jp.read_text())
        meta["dias_atraso"] = (TODAY - datetime.strptime(arquivos[0].stem, "%Y-%m-%d").date()).days
        return df, meta


# ══════════════════════════════════════════════════
#  PIPELINE PRINCIPAL
# ══════════════════════════════════════════════════

def executar(force=False):
    log.info("=" * 60)
    log.info("MERIDIAN — Pipeline v2 (CEPEA Widget + BCB + Yahoo)")
    log.info(f"Data: {TODAY}  |  Force: {force}")
    log.info("=" * 60)

    cache = Cache()
    if not force and cache.hoje_coletado():
        log.info("Cache do dia já existe. Use --force para reforçar.")
        df, _ = cache.carregar_ultimo()
        return df

    fontes = {}

    # 1 — BCB
    log.info("\n[1/3] BCB REST API...")
    bcb = ColetorBCB()
    dados_bcb = bcb.coletar_tudo(DATA_INICIO, datetime.today())
    fontes["bcb_ptax"] = "ok" if not dados_bcb.get("ptax", pd.DataFrame()).empty else "erro"

    # 2 — Yahoo Finance
    log.info("\n[2/3] Yahoo Finance...")
    yahoo = ColetorYahoo()
    df_yf = yahoo.coletar(DATA_INICIO, datetime.today())
    fontes["yahoo"] = "ok" if not df_yf.empty else "erro"

    # 3 — CEPEA Widget
    log.info("\n[3/3] CEPEA Widget...")
    cepea_widget = ColetorCEPEAWidget()
    dados_cepea = cepea_widget.coletar()
    df_cepea = cepea_widget.para_dataframe(dados_cepea)

    for col in dados_cepea:
        fontes[col] = "ok"
    if df_cepea.empty:
        log.warning("CEPEA Widget sem dados — usando basis/referência como fallback")
        fontes["cepea"] = "indisponivel"

    # Salvar status CEPEA para gerar_dashboard usar
    (CACHE_DIR / "cepea_status.json").write_text(
        json.dumps({v[0]: fontes.get(v[0], "indisponivel") for v in INDICADORES_WIDGET.values()},
                   ensure_ascii=False)
    )

    # Consolidar
    todas_fontes = [v for v in dados_bcb.values() if not v.empty]
    if not df_yf.empty:
        todas_fontes.append(df_yf)
    if not df_cepea.empty:
        todas_fontes.append(df_cepea)

    if not todas_fontes:
        log.error("Nenhum dado coletado.")
        df_cache, _ = cache.carregar_ultimo()
        return df_cache or pd.DataFrame()

    cons = Consolidador()
    df_final   = cons.merge(*todas_fontes)

    # ── Integrar histórico CEPEA (CSV local com 15 anos)
    hist_path = BASE / "data" / "historico" / "cepea_historico_consolidado.csv"
    if hist_path.exists():
        try:
            df_hist = pd.read_csv(hist_path, parse_dates=["data"])
            df_hist_idx = df_hist.set_index("data")
            df_final_idx = df_final.set_index("data") if "data" in df_final.columns else df_final

            # Colunas presentes no histórico mas ausentes ou com lacunas no dataset atual
            for col in df_hist.columns:
                if col == "data":
                    continue
                if col not in df_final_idx.columns:
                    # Coluna nova — adicionar do histórico
                    df_final_idx[col] = df_hist_idx[col].reindex(df_final_idx.index)
                else:
                    # Preencher NaN com histórico
                    mask = df_final_idx[col].isna()
                    if mask.any():
                        df_final_idx.loc[mask, col] = df_hist_idx[col].reindex(
                            df_final_idx.index[mask]
                        )
            df_final = df_final_idx.reset_index()
            cols_hist = [c for c in df_hist.columns if c != "data"]
            log.info(f"Histórico CEPEA integrado: {len(cols_hist)} séries | "
                     f"{df_hist['data'].min():%d/%m/%Y} → {df_hist['data'].max():%d/%m/%Y}")
        except Exception as e:
            log.warning(f"Histórico CEPEA não carregado: {e}")
    else:
        log.warning(f"CSV histórico não encontrado: {hist_path}")

    df_features = cons.features(df_final)

    df_features.to_csv(DATA_PROC / "dataset_producao.csv", index=False)
    cache.salvar(df_features, fontes)

    log.info("=" * 60)
    log.info(f"Concluído: {df_features.shape[0]} linhas × {df_features.shape[1]} colunas")
    log.info(f"Fontes: {fontes}")
    log.info("=" * 60)

    print("\n📊 ÚLTIMOS VALORES COLETADOS:")
    print(f"  {'Variável':<35} {'Último':>10}  Status")
    print("  " + "-" * 55)
    cols_print = [c for c in list(INDICADORES_WIDGET.values()) + [("ptax_venda","PTAX",""),("cbot_milho_usc_bu","CBOT Milho",""),("cbot_soja_usc_bu","CBOT Soja","")]
                  if isinstance(c, str) and c in df_features.columns]
    # simpler:
    for col_info in list(INDICADORES_WIDGET.values()):
        col = col_info[0]
        if col in df_features.columns:
            s = df_features[col].dropna()
            if not s.empty:
                st = fontes.get(col, "ok")
                icon = "✅" if st == "ok" else "⚠️ "
                print(f"  {col:<35} {s.iloc[-1]:>10.2f}  {icon} {st}")
    for col in ["ptax_venda", "cbot_milho_usc_bu", "cbot_soja_usc_bu", "usdbrl_spot"]:
        if col in df_features.columns:
            s = df_features[col].dropna()
            if not s.empty:
                print(f"  {col:<35} {s.iloc[-1]:>10.2f}  ✅ ok")

    return df_features


if __name__ == "__main__":
    import sys
    force = "--force" in sys.argv
    executar(force=force)
