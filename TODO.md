# TASKS

## Dorci - 2026-05-01

- [x] projekt scope-jának pontosítása
- [x] cikk fő ötletének leszűkítése
- [x] GitHub repo létrehozása
- [x] alap fájlstruktúra kialakítása
- [x] benchmark implementálása
- [x] egyszerű plotting utility-k megírása
- [x] `main.py` elkészítése a benchmarkhoz
- [x] `main.py` lefuttatása
    - [x] analitikus árfüggvény metszeteinek generálása
    - [x] analitikus árfüggvény 3D felületének generálása
- [x] PINN skeleton

## Dorci - 2026-05-03

- [ ] README
- [ ] első commit
- [ ] GitHub link elküldése a tanárnak

## TODO

### Black-Scholes benchmark

- [ ] átnézni még egyszer a Black-Scholes képlet implementációját
- [ ] pontosítani, milyen `(t, S)` tartományon dolgozunk
- [ ] létrehozni a tanításhoz használt szintetikus pontokat
- [ ] külön kezelni a belső pontokat, a lejárati pontokat és a perempontokat
- [ ] létrehozni egy tesztrácsot a kiértékeléshez
- [ ] kiszámolni az analitikus Black-Scholes értékeket ezen a rácson
- [ ] megírni az MSE/MAE kiértékelést

### PINN rész

- [ ] véglegesíteni az egyszerű PyTorch modellt
- [ ] kipróbálni, hogy a `forward(t, S)` tényleg a megfelelő alakú outputot adja
- [ ] megírni a PDE residual számítását autograddal
- [ ] megírni a loss külön részeit:
  - [ ] payoff feltétel
  - [ ] peremfeltételek
  - [ ] PDE residual
- [ ] összerakni a teljes loss-függvényt

### Training és eredmények

- [ ] megírni az első training loopot
- [ ] eltárolni a loss értékeket tanítás közben
- [ ] kirajzolni a loss görbét
- [ ] futtatni egy rövid próba-traininget, hogy látszódjon, működik-e
- [ ] ha stabil, futtatni egy hosszabb tanítást
- [ ] kiszámolni a PINN predikcióját a tesztrácson
- [ ] összehasonlítani az analitikus megoldással
- [ ] hibákat ábrázolni metszeteken és/vagy felületen
- [ ] eredmények rövid értelmezése

### Ha marad idő

- [ ] megnézni, tudunk-e egyszerűen valós opciós adatot szerezni (pl. `yfinance`)
- [ ] esetleg megnézni, tudjuk-e ugyanazt az adathalmazt használni, mint a cikk szerzői?
- [ ] választani egy részvényt és egy lejáratot
- [ ] letölteni egy call opciós láncot
- [ ] bid/ask alapján kiszámolni egy egyszerű mid price-t
- [ ] összevetni a piaci árakat a Black-Scholes/PINN árakkal

### Dokumentáció / leadás

- [ ] README frissítése, ha elkészül a PINN rész
- [ ] futtatási lépések ellenőrzése friss környezetben
- [ ] végső ábrák kiválasztása
- [ ] rövid beszámoló megírása
- [ ] prezentáció összerakása, felkészülés