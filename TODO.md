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

- [x] README
- [x] első commit
- [x] GitHub link elküldése a tanárnak

## Dorci - 2026-05-04

- [x] README rendbetétele
- [x] `src/__init__.py` hozzáadása
- [x] PINN bemenetek skálázása
- [x] szintetikus pontmintavételezés megírása
- [x] PINN loss komponenseinek előkészítése
- [x] egyszeri loss-számítás kipróbálása egy még nem tanított modellen
- [x] második commit és push

## Dorci - 2026-05-05

- [x] első egyszerű PINN training script megírása
- [x] rövid training futtatása
- [x] loss curve generálása

## Dorci - 2026-05-06
- [x] PINN evaluation script megírása
- [x] MSE és MAE kiszámítása az analitikus benchmarkhoz képest
- [x] PINN vs analitikus metszetábra generálása
- [x] PINN hibaábra generálása

## TODO

### Kiértékelés / eredmények

- [ ] első evaluation eredmények rövid értelmezése
- [ ] hibák vizsgálata különböző tartományokban
- [ ] finomított modell kiértékelése MSE/MAE metrikákkal
- [ ] végleges metszet- és hibaábrák generálása
- [ ] eredmények összefoglalása a beszámolóhoz

### Modell javítása

- [ ] training hosszának és learning rate-nek finomítása
- [ ] néhány egyszerű hiperparaméter-kísérlet futtatása
- [ ] hidden layer méret és PDE loss súlyának kipróbálása
- [ ] loss komponensek skálázásának / súlyozásának átgondolása
- [ ] stabilabb training futtatása a kiválasztott beállításokkal
- [ ] végső eredmények újragenerálása

### Ha marad idő

- [ ] megnézni, tudunk-e egyszerűen valós opciós adatot szerezni (pl. `yfinance`)
- [ ] esetleg megnézni, tudjuk-e ugyanazt az adathalmazt használni, mint a cikk szerzői?
- [ ] választani egy részvényt és egy lejáratot
- [ ] letölteni egy call opciós láncot
- [ ] bid/ask alapján kiszámolni egy egyszerű mid price-t
- [ ] összevetni a piaci árakat a Black-Scholes/PINN árakkal

### Dokumentáció / prezentáció

- [ ] README frissítése, ha elkészül a PINN rész
- [ ] futtatási lépések ellenőrzése friss környezetben
- [ ] végső ábrák kiválasztása
- [ ] rövid beszámoló megírása
- [ ] prezentáció összerakása, felkészülés