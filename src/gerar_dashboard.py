"""
MERIDIAN — Gerador do Dashboard v6
Injeta dados frescos no template HTML (docs/index.html).
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
DOCS_DIR  = BASE / "docs"
DOCS_DIR.mkdir(exist_ok=True)

PRODUTOS = {
    "milho_esalq_brl_saca": {"nome":"Milho ESALQ","emoji":"🌽","unidade":"R$/saca","fonte_label":"CEPEA/ESALQ","fonte_desc":"Ind. Milho ESALQ/BM&F · Campinas SP"},
    "soja_esalq_brl_saca":  {"nome":"Soja ESALQ","emoji":"🫘","unidade":"R$/saca","fonte_label":"CEPEA/ESALQ","fonte_desc":"Ind. Soja CEPEA/ESALQ · Paranaguá PR"},
    "frango_brl_kg":        {"nome":"Frango","emoji":"🐔","unidade":"R$/kg","fonte_label":"CEPEA","fonte_desc":"Frango congelado atacado paulista · CEPEA SP"},
    "suino_brl_kg":         {"nome":"Suíno","emoji":"🐷","unidade":"R$/kg","fonte_label":"CEPEA","fonte_desc":"Suíno vivo SP · CEPEA"},
    "boi_cepea_brl_arroba": {"nome":"Boi Gordo","emoji":"🐄","unidade":"R$/@","fonte_label":"CEPEA/B3","fonte_desc":"Ind. Boi Gordo CEPEA/B3 · à vista SP"},
    "ptax_venda":           {"nome":"PTAX","emoji":"💵","unidade":"R$/USD","fonte_label":"BCB PTAX","fonte_desc":"PTAX fechamento · Banco Central do Brasil"},
}
SAZON = {
    "milho_esalq_brl_saca": [4.5,3.2,0.5,-5.2,-6.1,-4.8,-2.3,-1.1,1.8,3.4,4.1,5.0],
    "soja_esalq_brl_saca":  [1.8,6.2,-0.1,0.2,-1.7,-4.7,-2.4,0.1,3.0,2.6,3.5,5.5],
    "frango_brl_kg":        [0.5,3.5,1.7,1.3,0.2,-2.3,-4.2,-2.9,0.2,0.2,2.4,2.3],
    "suino_brl_kg":         [0.3,-0.7,-0.2,0.6,-1.5,-0.7,-1.6,-0.8,-1.5,0.8,0.1,3.1],
    "boi_cepea_brl_arroba": [2.1,0.2,1.3,0.6,-0.5,-3.8,1.7,-1.1,-2.2,0.2,0.7,1.2],
    "ptax_venda":           [0.2,0.9,2.2,0.6,-3.0,0.7,1.0,1.5,2.8,2.7,-0.4,3.2],
}

def _var(s, n):
    return round(float((s.iloc[-1]/s.iloc[-1-n]-1)*100),1) if len(s)>n else 0.0

def _score(s):
    if len(s)<120: return 0
    v=s.iloc[-1]; m20=s.rolling(20).mean().iloc[-1]; m60=s.rolling(60).mean().iloc[-1]
    m120=s.rolling(120).mean().iloc[-1]
    r30=(v/s.iloc[-22]-1) if len(s)>22 else 0
    r90=(v/s.iloc[-63]-1) if len(s)>63 else 0
    return sum([1 if v>m20 else -1,1 if v>m60 else -1,1 if v>m120 else -1,1 if r30>0 else -1,1 if r90>0 else -1])

def _probs(sc):
    if sc>=4: return [5.0,15.0,80.0]
    elif sc==3: return [10.0,25.0,65.0]
    elif sc>=1: return [20.0,45.0,35.0]
    elif sc==0: return [33.0,34.0,33.0]
    elif sc>=-2: return [35.0,45.0,20.0]
    elif sc==-3: return [65.0,25.0,10.0]
    else: return [80.0,15.0,5.0]

def _sinal(sc): return "ALTA" if sc>=3 else ("BAIXA" if sc<=-3 else "LATERAL")

def gerar_raw(df):
    hoje = datetime.today()
    modelo_cache = {}
    mp = CACHE_DIR/"sinais_modelo.json"
    if mp.exists():
        try: modelo_cache = json.loads(mp.read_text())
        except: pass

    status_fontes = {}
    sp = CACHE_DIR/"ultimo_status.json"
    if sp.exists():
        try: status_fontes = json.loads(sp.read_text()).get("fontes",{})
        except: pass

    stats = {}
    for col, cfg in PRODUTOS.items():
        if col not in df.columns: continue
        s = df[col].dropna()
        if len(s)<10: continue
        df_w = (df[["data",col]].dropna().set_index("data")
                .resample("W").last().ffill().tail(26).reset_index())
        sc = _score(s)
        m  = modelo_cache.get(col, {})
        st = status_fontes.get(col, "ok")
        fl = cfg["fonte_label"] + (" (est.)" if st=="basis_estimado" else "")
        fd = "Estimado via Basis Adaptativo" if st=="basis_estimado" else cfg["fonte_desc"]

        # Data de referência
        try:
            dr = df["data"].iloc[s.last_valid_index()].strftime("%d/%m/%Y")
        except:
            dr = hoje.strftime("%d/%m/%Y")

        stats[col] = {
            "nome":col and cfg["nome"],"emoji":cfg["emoji"],"unidade":cfg["unidade"],
            "fonte_label":fl,"fonte_desc":fd,"data_ref":dr,
            "last":round(float(s.iloc[-1]),2),
            "var7":_var(s,6),"var30":_var(s,21),"var90":_var(s,63),
            "mm20":round(float(s.rolling(20).mean().iloc[-1]),2),
            "mm60":round(float(s.rolling(60).mean().iloc[-1]),2),
            "vol30":round(float(s.pct_change().rolling(30).std().iloc[-1]*np.sqrt(252)*100),1),
            "score":sc,"sinal":m.get("sinal",_sinal(sc)),
            "probs":m.get("probs",_probs(sc)),"acc":m.get("acc"),"f1":m.get("f1"),
            "sazon":SAZON.get(col,[0]*12),
            "min":round(float(s.tail(252).min()),2),"max":round(float(s.tail(252).max()),2),
            "hist":[round(float(x),2) for x in df_w[col].tolist()],
            "dates":df_w["data"].dt.strftime("%d/%m/%y").tolist(),
        }
        # corrigir o nome (acima sobrescreve com None por erro)
        stats[col]["nome"] = cfg["nome"]

    return {"data_base":hoje.strftime("%d/%m/%Y"),"hora_coleta":hoje.strftime("%H:%M"),"stats":stats}

def main():
    log.info("Gerando dashboard...")
    csv_path = DATA_PROC / "dataset_producao.csv"
    if not csv_path.exists():
        parquets = sorted(CACHE_DIR.glob("*.parquet"), reverse=True)
        if not parquets:
            log.error("Dataset não encontrado. Execute pipeline.py primeiro.")
            return
        df = pd.read_parquet(parquets[0])
    else:
        df = pd.read_csv(csv_path, parse_dates=["data"])

    raw  = gerar_raw(df)
    log.info(f"RAW: {len(raw['stats'])} produtos · {raw['data_base']} {raw['hora_coleta']}")

    # Injetar no template
    tpl = (DOCS_DIR/"index.html").read_text(encoding="utf-8")
    raw_json = json.dumps(raw, ensure_ascii=False, separators=(",",":"))
    novo = re.sub(r"const RAW = \{.*?\};", f"const RAW = {raw_json};", tpl, flags=re.DOTALL)
    if novo == tpl:
        raise ValueError("Padrão 'const RAW = {...};' não encontrado no template.")

    (DOCS_DIR/"index.html").write_text(novo, encoding="utf-8")
    (DOCS_DIR/"data.json").write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")

    # Salvar status
    (CACHE_DIR/"ultimo_status.json").write_text(json.dumps({
        "data_base":raw["data_base"],"hora_coleta":raw["hora_coleta"],
        "produtos":{k:v["sinal"] for k,v in raw["stats"].items()},
        "fontes":{k:v["fonte_label"] for k,v in raw["stats"].items()},
    }, ensure_ascii=False, indent=2))

    log.info(f"docs/index.html atualizado ({len(novo)//1024}kb)")
    print(f"\n✅ Dashboard gerado: {raw['data_base']} {raw['hora_coleta']}")
    for col,s in raw["stats"].items():
        print(f"  {s['emoji']} {s['nome']:<18} {s['last']:>9.2f} {s['unidade']:<9} "
              f"{s['var30']:>+6.1f}%  {s['sinal']:<12}  {s['fonte_label']}")

if __name__=="__main__":
    main()
