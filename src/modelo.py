"""
MERIDIAN — Modelo Preditivo v1.0
Treina GBM para produtos onde o modelo supera o baseline.
Salva sinais em data/cache/sinais_modelo.json para o dashboard.
"""
import json, logging, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("modelo")

BASE       = Path(__file__).parent.parent
HIST_PATH  = BASE / "data" / "historico" / "cepea_historico_consolidado.csv"
PROC_PATH  = BASE / "data" / "processed"  / "dataset_producao.csv"
CACHE_DIR  = BASE / "data" / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# Resultados do backtesting — define se usamos modelo ou score técnico
BACKTEST = {
    "milho_brl_sc":            {"usar_modelo": False, "acc": 40.0, "prec_alta": 44.7, "prec_baixa": 38.6},
    "soja_paranagua_brl_sc":   {"usar_modelo": False, "acc": 39.0, "prec_alta": 36.9, "prec_baixa": 32.5},
    "boi_cepea_brl_arroba":    {"usar_modelo": False, "acc": 43.0, "prec_alta": 38.2, "prec_baixa": 24.0},
    "frango_congelado_brl_kg": {"usar_modelo": True,  "acc": 41.8, "prec_alta": 47.6, "prec_baixa": 39.2},
    "suino_sp_brl_kg":         {"usar_modelo": True,  "acc": 55.7, "prec_alta": 59.9, "prec_baixa": 55.9},
    "arroz_brl_sc":            {"usar_modelo": False, "acc": 42.6, "prec_alta": 38.0, "prec_baixa": 54.6},
    "cafe_arabica_brl_sc":     {"usar_modelo": False, "acc": 36.9, "prec_alta": 45.9, "prec_baixa": 34.1},
    "acucar_brl_sc":           {"usar_modelo": True,  "acc": 42.8, "prec_alta": 57.6, "prec_baixa": 36.0},
    "trigo_pr_brl_t":          {"usar_modelo": True,  "acc": 46.6, "prec_alta": 49.3, "prec_baixa": 54.5},
    "soja_pr_brl_sc":          {"usar_modelo": False, "acc": 39.5, "prec_alta": 38.0, "prec_baixa": 33.0},
    "ovos_bastos_brl_cx":      {"usar_modelo": False, "acc": None, "prec_alta": None, "prec_baixa": None},
    "ptax_venda":              {"usar_modelo": False, "acc": None, "prec_alta": None, "prec_baixa": None},
}

HORIZONTE = 30
LIMIAR    = 3.0


def carregar_dataset() -> pd.DataFrame:
    """Carrega e mescla histórico + produção."""
    frames = []
    if HIST_PATH.exists():
        frames.append(pd.read_csv(HIST_PATH, parse_dates=["data"]))
        log.info(f"Histórico: {len(frames[-1])} linhas")
    if PROC_PATH.exists():
        frames.append(pd.read_csv(PROC_PATH, parse_dates=["data"]))
        log.info(f"Produção:  {len(frames[-1])} linhas")
    if not frames:
        return pd.DataFrame()
    base = frames[0].set_index("data")
    for f in frames[1:]:
        f2 = f.set_index("data")
        for col in f2.columns:
            if col in base.columns:
                mask = f2[col].notna()
                base.loc[f2.index[mask], col] = f2.loc[mask, col]
            else:
                base[col] = f2[col]
    df = base.sort_index().ffill(limit=5).reset_index()
    log.info(f"Dataset final: {df.shape}")
    return df


def build_features(s: np.ndarray, i: int, mes: int) -> list:
    if i < 120 or np.isnan(s[i]):
        return None
    v = s[i]
    f = []
    for n in [5, 10, 20, 30, 60, 120]:
        f.append((v/s[i-n]-1)*100 if i>=n and not np.isnan(s[i-n]) and s[i-n]>0 else 0.0)
    for n in [20, 60, 120]:
        mm = np.nanmean(s[max(0,i-n+1):i+1])
        f.append((v/mm-1)*100 if mm > 0 else 0.0)
    ret30 = np.diff(np.log(np.maximum(s[max(0,i-30):i+1], 1e-8)))
    ret30 = ret30[~np.isnan(ret30)]
    f.append(ret30.std() * np.sqrt(252) * 100 if len(ret30) > 2 else 0.0)
    f.extend([np.sin(2*np.pi*mes/12), np.cos(2*np.pi*mes/12)])
    s252 = s[max(0,i-252):i+1]; s252 = s252[~np.isnan(s252)]
    f.append((v-s252.min())/(s252.max()-s252.min()+1e-8) if len(s252) > 0 else 0.5)
    return f


def score_tecnico(s: np.ndarray, i: int) -> tuple:
    """Score técnico por médias móveis. Retorna (sinal, probs)."""
    if i < 120 or np.isnan(s[i]):
        return "LATERAL", [33.0, 34.0, 33.0]
    v = s[i]
    score = 0
    for n, col in [(20,0),(60,0),(120,0)]:
        mm = np.nanmean(s[max(0,i-n+1):i+1])
        score += 1 if v > mm else -1
    r30 = (v/s[i-22]-1) if i>=22 and not np.isnan(s[i-22]) and s[i-22]>0 else 0
    r90 = (v/s[i-63]-1) if i>=63 and not np.isnan(s[i-63]) and s[i-63]>0 else 0
    score += (1 if r30 > 0 else -1) + (1 if r90 > 0 else -1)
    if score >= 3:   return "ALTA",    [10.0, 20.0, 70.0]
    if score >= 1:   return "LATERAL", [25.0, 45.0, 30.0]
    if score == 0:   return "LATERAL", [33.0, 34.0, 33.0]
    if score >= -2:  return "LATERAL", [30.0, 45.0, 25.0]
    return "BAIXA", [70.0, 20.0, 10.0]


def treinar_e_prever(df: pd.DataFrame, col: str) -> tuple:
    """Treina GBM com todo histórico disponível e prediz o momento atual."""
    try:
        from sklearn.ensemble import GradientBoostingClassifier
    except ImportError:
        return None, None

    s     = df[col].values.copy()
    datas = df["data"].tolist()
    idx   = [i for i in range(len(s)) if not np.isnan(s[i])]

    if len(idx) < 600:
        return None, None

    # Treino: tudo menos os últimos 30 dias
    treino_ate = len(idx) - 30
    X_train, y_train = [], []
    for ii in idx[:treino_ate]:
        ft = build_features(s, ii, datas[ii].month)
        if ii + HORIZONTE >= len(s) or np.isnan(s[ii+HORIZONTE]):
            continue
        r = (s[ii+HORIZONTE]/s[ii]-1)*100
        lb = "ALTA" if r > LIMIAR else ("BAIXA" if r < -LIMIAR else "LATERAL")
        if ft:
            X_train.append(ft)
            y_train.append(lb)

    if len(set(y_train)) < 2 or len(X_train) < 200:
        return None, None

    clf = GradientBoostingClassifier(
        n_estimators=100, max_depth=3,
        learning_rate=0.05, subsample=0.8,
        random_state=42
    )
    clf.fit(X_train, y_train)

    # Predição no ponto mais recente
    i_atual = idx[-1]
    ft_atual = build_features(s, i_atual, datas[i_atual].month)
    if not ft_atual:
        return None, None

    pred   = clf.predict([ft_atual])[0]
    proba  = clf.predict_proba([ft_atual])[0]
    classes = list(clf.classes_)
    probs  = [
        float(round(proba[classes.index("BAIXA")]*100, 1)) if "BAIXA" in classes else 33.0,
        float(round(proba[classes.index("LATERAL")]*100, 1)) if "LATERAL" in classes else 34.0,
        float(round(proba[classes.index("ALTA")]*100, 1)) if "ALTA" in classes else 33.0,
    ]
    return pred, probs


def main():
    log.info("Modelo Preditivo Meridian v1.0")
    log.info("="*50)

    df = carregar_dataset()
    if df.empty:
        log.error("Dataset vazio — abortando.")
        return

    sinais = {}

    for col, bt in BACKTEST.items():
        if col not in df.columns:
            continue
        s     = df[col].values
        datas = df["data"].tolist()
        idx   = [i for i in range(len(s)) if not np.isnan(s[i])]
        if not idx:
            continue
        i_atual = idx[-1]

        if bt["usar_modelo"]:
            # Usar modelo GBM
            pred, probs = treinar_e_prever(df, col)
            if pred and probs:
                sinais[col] = {
                    "sinal":   pred,
                    "probs":   probs,
                    "metodo":  "GBM",
                    "acc":     bt["acc"],
                    "prec_alta":  bt["prec_alta"],
                    "prec_baixa": bt["prec_baixa"],
                }
                log.info(f"  GBM  {col:<33} → {pred:<8} probs={probs}")
            else:
                # Fallback para técnico
                sinal, probs = score_tecnico(s, i_atual)
                sinais[col] = {"sinal":sinal,"probs":probs,"metodo":"técnico (fallback)","acc":bt["acc"],"prec_alta":bt["prec_alta"],"prec_baixa":bt["prec_baixa"]}
                log.warning(f"  FALL {col:<33} → {sinal}")
        else:
            # Score técnico puro
            sinal, probs = score_tecnico(s, i_atual)
            sinais[col] = {
                "sinal":   sinal,
                "probs":   probs,
                "metodo":  "técnico",
                "acc":     bt["acc"],
                "prec_alta":  bt["prec_alta"],
                "prec_baixa": bt["prec_baixa"],
            }
            log.info(f"  TEC  {col:<33} → {sinal:<8} probs={probs}")

    # Salvar sinais para o dashboard
    (CACHE_DIR / "sinais_modelo.json").write_text(
        json.dumps(sinais, ensure_ascii=False, indent=2)
    )
    log.info(f"Sinais salvos: {len(sinais)} produtos")

    print("\n✅ Sinais do modelo:")
    print(f"  {'Produto':<33} {'Sinal':<10} {'Prob Alta':>10}  {'Método':<20}  {'Acc backtest'}")
    print("  " + "-"*75)
    for col, s in sinais.items():
        acc_str = f"{s['acc']:.1f}%" if s['acc'] else "—"
        print(f"  {col:<33} {s['sinal']:<10} {s['probs'][2]:>9.1f}%  {s['metodo']:<20}  {acc_str}")


if __name__ == "__main__":
    main()
