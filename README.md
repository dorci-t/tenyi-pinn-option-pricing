# Black-Scholes Option Pricing with Physics-Informed Neural Networks

## Rövid projektleírás

## TODO: ez most szerintem picit zavaros, át kell még írni

Alapötlet: https://arxiv.org/abs/2312.06711

A projekt célja a Black-Scholes parciális differenciálegyenlet numerikus közelítése Physics-Informed Neural Network (PINN) segítségével.

A teljes cikk reprodukciója helyett a projekt egy részfeladatra fókuszál:

> Európai call opció árának közelítése PINN-nel, majd összehasonlítás az analitikus Black-Scholes képlettel.

Ez azért jó kiindulópont, mert európai call opcióra ismert zárt alakú megoldás van, ezért pontos benchmarkkal tudjuk ellenőrizni a neurális háló eredményét.

A projekt alapját egy szintetikus példa adja, nem piaci adathalmaz. A PINN tanításához a Black-Scholes PDE értelmezési tartományából választunk pontokat, az eredményeket pedig az analitikus Black-Scholes képlettel hasonlítjuk össze. Ha marad idő, megnézünk valódi piaci opciós adatokat is.

## Matematikai háttér

A Black-Scholes PDE (parciális differenciálegyenlet):

```math
\frac{\partial V}{\partial t}
+
\frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2}
+
rS\frac{\partial V}{\partial S}
-
rV
=
0.
```

ahol:

- `S`: részvényárfolyam,
- `t`: idő,
- `V(t,S)`: az opció ára,
- `K`: kötési árfolyam / strike price,
- `T`: lejárati idő,
- `r`: kockázatmentes kamatláb,
- `sigma`: volatilitás.

Európai call opció lejáratkori feltétele:

```math
V(T,S) = \max(S-K,0).
```

## Projektterv

### 1. Analitikus benchmark

Implementáljuk az európai call opció Black-Scholes képletét, majd kirajzoljuk az árfüggvényt különböző időpontokban.

### 2. PINN-modell

PyTorch segítségével neurális hálót építünk, amelynek bemenete `(t, S)`, kimenete pedig `V(t,S)`.

### 3. Loss-függvény

A loss három fő részből áll majd:

1. lejáratkori feltétel hibája,
2. peremfeltételek hibája,
3. Black-Scholes PDE residual hibája.

### 4. Kiértékelés

A PINN eredményét az analitikus Black-Scholes árral hasonlítjuk majd össze. A tervezett kiértékelési elemek:

- MSE és MAE a PINN-predikció és az analitikus ár között,
- a PINN tanítási veszteségének görbéje,
- analitikus és prediktált árfelület,
- hibafelület,
- metszetek fix időpontokban.

## Jelenlegi státusz

- [x] projekt scope meghatározva
- [x] GitHub-repo struktúra előkészítve
- [x] analitikus Black-Scholes képlet implementálva
- [x] első ábrák generálása megvan
- [ ] PINN-modell betanítása
- [ ] PINN vs analitikus benchmark összehasonlítás
- [ ] felkészülés a prezentációra

## Futtatás

Telepítés:

```bash
python -m venv .venv
```

Windows alatt:

```bash
.venv\Scripts\activate
```

macOS/Linux alatt:

```bash
source .venv/bin/activate
```

Csomagok telepítése:

```bash
pip install -r requirements.txt
```

Első analitikus benchmark futtatása:

```bash
python main.py
```

A generált ábrák a `figures/` mappába kerülnek.

## Forrás

A projekt alapötlete a következő cikkből származik:

Dhiman, A. and Hu, Y. (2023) Physics informed Neural Network for option pricing, arXiv.org. Available at: https://arxiv.org/abs/2312.06711 (Accessed: 03 May 2026). 

## Tervezett programcsomagok
## TODO: véglegesítés (majd a projekt végén)

- Python
- NumPy
- SciPy
- Matplotlib
- PyTorch
- Jupyter Notebook (nem programcsomag, de ugyanúgy installálni kell)
