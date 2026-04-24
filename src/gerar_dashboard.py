"""
MERIDIAN — Gerador do Dashboard v6.3
Lê dataset_producao.csv + historico CEPEA e gera docs/index.html
"""
import json, logging, warnings, re
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("dashboard")

BASE      = Path(__file__).parent.parent
DATA_PROC = BASE / "data" / "processed"
CACHE_DIR = BASE / "data" / "cache"
HIST_DIR  = BASE / "data" / "historico"
DOCS_DIR  = BASE / "docs"
DOCS_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

# ── Todos os produtos
PRODUTOS = {
    "boi_cepea_brl_arroba":    {"nome":"Boi Gordo",           "emoji":"🐄","unidade":"R$/@",    "grupo":"Proteínas"},
    "frango_congelado_brl_kg": {"nome":"Frango Congelado",    "emoji":"🐔","unidade":"R$/kg",   "grupo":"Proteínas"},
    "suino_sp_brl_kg":         {"nome":"Suíno SP",            "emoji":"🐷","unidade":"R$/kg",   "grupo":"Proteínas"},
    "ovos_bastos_brl_cx":      {"nome":"Ovos Bastos SP",      "emoji":"🥚","unidade":"R$/cx",   "grupo":"Proteínas"},
    "milho_brl_sc":            {"nome":"Milho ESALQ",         "emoji":"🌽","unidade":"R$/sc",   "grupo":"Grãos"},
    "soja_paranagua_brl_sc":   {"nome":"Soja Paranaguá",      "emoji":"🫘","unidade":"R$/sc",   "grupo":"Grãos"},
    "soja_pr_brl_sc":          {"nome":"Soja PR",             "emoji":"🫘","unidade":"R$/sc",   "grupo":"Grãos"},
    "trigo_pr_brl_t":          {"nome":"Trigo PR",            "emoji":"🌾","unidade":"R$/t",    "grupo":"Grãos"},
    "arroz_brl_sc":            {"nome":"Arroz IRGA-RS",       "emoji":"🍚","unidade":"R$/sc",   "grupo":"Grãos"},
    "cafe_arabica_brl_sc":     {"nome":"Café Arábica",        "emoji":"☕","unidade":"R$/sc",   "grupo":"Outros"},
    "acucar_brl_sc":           {"nome":"Açúcar Cristal SP",   "emoji":"🍬","unidade":"R$/sc",   "grupo":"Outros"},
    "ptax_venda":              {"nome":"PTAX",                "emoji":"💵","unidade":"R$/USD",  "grupo":"Macro"},
}

SAZON = {
    "boi_cepea_brl_arroba":    [2.1,0.2,1.3,0.6,-0.5,-3.8,1.7,-1.1,-2.2,0.2,0.7,1.2],
    "frango_congelado_brl_kg": [0.5,3.5,1.7,1.3,0.2,-2.3,-4.2,-2.9,0.2,0.2,2.4,2.3],
    "suino_sp_brl_kg":         [0.3,-0.7,-0.2,0.6,-1.5,-0.7,-1.6,-0.8,-1.5,0.8,0.1,3.1],
    "ovos_bastos_brl_cx":      [2.0,1.5,-0.5,-1.5,-2.0,-1.0,0.5,1.5,2.0,1.5,1.0,2.5],
    "milho_brl_sc":            [4.5,3.2,0.5,-5.2,-6.1,-4.8,-2.3,-1.1,1.8,3.4,4.1,5.0],
    "soja_paranagua_brl_sc":   [1.8,6.2,-0.1,0.2,-1.7,-4.7,-2.4,0.1,3.0,2.6,3.5,5.5],
    "soja_pr_brl_sc":          [1.8,6.2,-0.1,0.2,-1.7,-4.7,-2.4,0.1,3.0,2.6,3.5,5.5],
    "trigo_pr_brl_t":          [1.5,2.0,0.5,-1.0,-2.0,-1.5,0.5,2.0,3.0,2.5,1.0,0.5],
    "arroz_brl_sc":            [1.0,1.5,0.5,-0.5,-2.0,-2.5,-1.5,0.5,2.0,2.5,2.0,1.5],
    "cafe_arabica_brl_sc":     [1.0,2.5,1.5,0.5,-1.0,-2.0,-1.5,0.5,2.0,3.0,2.5,1.5],
    "acucar_brl_sc":           [1.0,1.5,0.5,-0.5,-1.5,-2.0,-1.0,0.5,1.5,2.0,1.5,1.0],
    "ptax_venda":              [0.2,0.9,2.2,0.6,-3.0,0.7,1.0,1.5,2.8,2.7,-0.4,3.2],
}

FONTE_LABEL = {
    "boi_cepea_brl_arroba":    "CEPEA/B3",
    "frango_congelado_brl_kg": "CEPEA",
    "suino_sp_brl_kg":         "CEPEA",
    "ovos_bastos_brl_cx":      "CEPEA",
    "milho_brl_sc":            "CEPEA/ESALQ",
    "soja_paranagua_brl_sc":   "CEPEA/ESALQ",
    "soja_pr_brl_sc":          "CEPEA/ESALQ",
    "trigo_pr_brl_t":          "CEPEA",
    "arroz_brl_sc":            "CEPEA/IRGA",
    "cafe_arabica_brl_sc":     "CEPEA/ESALQ",
    "acucar_brl_sc":           "CEPEA/ESALQ",
    "ptax_venda":              "BCB PTAX",
}

def _var(s, n):
    return round(float((s.iloc[-1]/s.iloc[-1-n]-1)*100),1) if len(s)>n else 0.0

def _score(s):
    if len(s)<120: return 0
    v=s.iloc[-1]; m20=s.rolling(20).mean().iloc[-1]; m60=s.rolling(60).mean().iloc[-1]
    m120=s.rolling(120).mean().iloc[-1]
    r30=(v/s.iloc[-22]-1) if len(s)>22 else 0
    r90=(v/s.iloc[-63]-1) if len(s)>63 else 0
    return sum([1 if v>m20 else -1,1 if v>m60 else -1,1 if v>m120 else -1,
                1 if r30>0 else -1,1 if r90>0 else -1])

def _probs(sc):
    if sc>=4: return [5.0,15.0,80.0]
    elif sc==3: return [10.0,25.0,65.0]
    elif sc>=1: return [20.0,45.0,35.0]
    elif sc==0: return [33.0,34.0,33.0]
    elif sc>=-2: return [35.0,45.0,20.0]
    elif sc==-3: return [65.0,25.0,10.0]
    else: return [80.0,15.0,5.0]

def _sinal(sc): return "ALTA" if sc>=3 else ("BAIXA" if sc<=-3 else "LATERAL")


def carregar_dataset() -> pd.DataFrame:
    """Carrega dataset: CSV histórico + produção + cache widget."""
    frames = []

    # 1. Histórico CEPEA (base principal)
    hist = HIST_DIR / "cepea_historico_consolidado.csv"
    if hist.exists():
        df_h = pd.read_csv(hist, parse_dates=["data"])
        frames.append(df_h)
        log.info(f"Histórico CEPEA: {len(df_h)} linhas ({df_h['data'].min():%d/%m/%Y} → {df_h['data'].max():%d/%m/%Y})")

    # 2. Dataset produção (BCB + Yahoo + widget hoje)
    csv = DATA_PROC / "dataset_producao.csv"
    if csv.exists():
        df_p = pd.read_csv(csv, parse_dates=["data"])
        frames.append(df_p)
        log.info(f"Dataset produção: {len(df_p)} linhas")

    # 3. Cache widget (último snapshot CEPEA)
    widget_json = CACHE_DIR / "cepea_widget_ultimo.json"
    if widget_json.exists():
        try:
            dados = json.loads(widget_json.read_text())
            hoje = datetime.today().date()
            row = {"data": pd.Timestamp(hoje)}
            for col, info in dados.items():
                row[col] = info["valor"]
            frames.append(pd.DataFrame([row]))
            log.info(f"Widget CEPEA: {len(dados)} produtos ({hoje})")
        except Exception as e:
            log.warning(f"Widget JSON: {e}")

    if not frames:
        log.error("Nenhum dataset encontrado!")
        return pd.DataFrame()

    # Merge: outer join por data, depois forward-fill
    df = frames[0].set_index("data")
    for f in frames[1:]:
        if f.empty: continue
        f2 = f.set_index("data")
        # Atualizar colunas existentes e adicionar novas
        for col in f2.columns:
            if col in df.columns:
                # Priorizar dados mais recentes (não sobrescrever com NaN)
                mask = f2[col].notna()
                df.loc[f2.index[mask], col] = f2.loc[mask, col]
            else:
                df[col] = f2[col]
    
    df = df.sort_index().ffill(limit=5).reset_index()
    log.info(f"Dataset final: {df.shape}")
    return df


def gerar_raw(df: pd.DataFrame) -> dict:
    hoje = datetime.today()
    stats = {}

    for col, cfg in PRODUTOS.items():
        if col not in df.columns:
            log.warning(f"Coluna não encontrada: {col}")
            continue
        s = df[col].dropna()
        if len(s) < 5:
            log.warning(f"Dados insuficientes: {col} ({len(s)} obs)")
            continue

        # Série semanal 26 semanas
        df_w = (df[["data", col]].dropna()
                .set_index("data").resample("W").last()
                .ffill().tail(26).reset_index())

        sc = _score(s)
        fl = FONTE_LABEL.get(col, "CEPEA")

        try:
            last_idx = s.last_valid_index()
            dr = pd.Timestamp(
                df["data"].iloc[last_idx] if isinstance(last_idx, int) else last_idx
            ).strftime("%d/%m/%Y")
        except:
            dr = hoje.strftime("%d/%m/%Y")

        stats[col] = {
            "nome":        cfg["nome"],
            "emoji":       cfg["emoji"],
            "unidade":     cfg["unidade"],
            "grupo":       cfg["grupo"],
            "fonte_label": fl,
            "fonte_desc":  f"{fl} · {cfg['nome']}",
            "data_ref":    dr,
            "estimado":    len(s) < 30,
            "last":        round(float(s.iloc[-1]), 2),
            "var7":        _var(s, 6),
            "var30":       _var(s, 21),
            "var90":       _var(s, 63),
            "mm20":        round(float(s.rolling(20).mean().iloc[-1]), 2) if len(s)>=20 else round(float(s.iloc[-1]),2),
            "mm60":        round(float(s.rolling(60).mean().iloc[-1]), 2) if len(s)>=60 else round(float(s.iloc[-1]),2),
            "vol30":       round(float(s.pct_change().rolling(30).std().iloc[-1]*np.sqrt(252)*100), 1) if len(s)>=30 else 0.0,
            "score":       sc,
            "sinal":       _sinal(sc),
            "probs":       _probs(sc),
            "acc":         None,
            "f1":          None,
            "sazon":       SAZON.get(col, [0]*12),
            "min":         round(float(s.tail(252).min()), 2),
            "max":         round(float(s.tail(252).max()), 2),
            "hist":        [round(float(x), 2) for x in df_w[col].tolist()],
            "dates":       df_w["data"].dt.strftime("%d/%m/%y").tolist(),
        }
        stats[col]["nome"] = cfg["nome"]  # garantir

    return {
        "data_base":   hoje.strftime("%d/%m/%Y"),
        "hora_coleta": hoje.strftime("%H:%M"),
        "stats":       stats,
    }


def main():
    log.info("Gerando dashboard v6.3...")

    df = carregar_dataset()
    if df.empty:
        log.error("Dataset vazio — abortando.")
        return

    raw = gerar_raw(df)
    log.info(f"RAW: {len(raw['stats'])} produtos · {raw['data_base']} {raw['hora_coleta']}")

    # Injetar no template
    tpl_path = DOCS_DIR / "index.html"
    if not tpl_path.exists():
        log.error("Template docs/index.html não encontrado!")
        return

    tpl = tpl_path.read_text(encoding="utf-8")
    raw_json = json.dumps(raw, ensure_ascii=False, separators=(",", ":"))
    novo = re.sub(r"const RAW = \{.*?\};", f"const RAW = {raw_json};", tpl, flags=re.DOTALL)

    if novo == tpl:
        log.error("Padrão 'const RAW = {...};' não encontrado no template!")
        return

    tpl_path.write_text(novo, encoding="utf-8")
    (DOCS_DIR / "data.json").write_text(
        json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Status
    (CACHE_DIR / "ultimo_status.json").write_text(json.dumps({
        "data_base":   raw["data_base"],
        "hora_coleta": raw["hora_coleta"],
        "produtos":    {k: v["sinal"]       for k, v in raw["stats"].items()},
        "fontes":      {k: v["fonte_label"] for k, v in raw["stats"].items()},
    }, ensure_ascii=False, indent=2))

    log.info(f"docs/index.html atualizado ({len(novo)//1024}kb)")
    print(f"\n✅ Dashboard: {raw['data_base']} {raw['hora_coleta']} — {len(raw['stats'])} produtos")
    for col, s in raw["stats"].items():
        est = " ⚠" if s["estimado"] else ""
        print(f"  {s['emoji']} {s['nome']:<25} {s['last']:>9.2f} {s['unidade']:<10} "
              f"{s['var30']:>+6.1f}%  {s['sinal']:<12}  {s['fonte_label']}{est}")


if __name__ == "__main__":
    main()
