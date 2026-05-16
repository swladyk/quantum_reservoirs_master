# Plan dalszych prac — ocena pod cele pracy magisterskiej

Cele pracy:
1. zbadanie działania **ukrytej oraz jawnej pamięci** w zaproponowanym modelu hybrydowego rezerwuaru kwantowego
2. zbadanie **efektywnej wymiarowości** jaką kreuje ten model
3. **porównanie wyników do klasycznego rezerwuaru typu ESN** (Echo State Networks)

---

# Część 1 — Ocena merytoryczna obecnego stanu

## Ocena per cel pracy

### Cel 1: ukryta (leaky) vs jawna (window) pamięć

**Co mamy:**
- W gridzie są ablacje `leakage_rate=1.0` (czysta pamięć okna) i `window_size=1` (czysta pamięć leaky, brak splątania)
- Notebook 04 — tabela per-dataset z 4 konfiguracjami pokazuje liczbowo wyniki obu skrajnych przypadków
- Heatmapy z notebook 04 pokazują 2D krajobraz interakcji obu mechanizmów

**Czego brakuje:**
- **Dedykowana analiza dekompozycji pamięci** — narracyjnie jak każdy mechanizm wpływa na NMSE per profil
- Test statystyczny: czy okno daje istotnie lepszą pamięć niż leaky? (paired Wilcoxon na per-sub-seed wynikach)
- Wykres pasujący do narracji magisterskiej: dla każdego profilu pasek/słupek pokazujący NMSE w trzech reżimach (pure leaky / pure window / combined) — szybkie wizualne porównanie

### Cel 2: efektywna wymiarowość

**Co mamy:**
- PCA scatter per profil (z notebook 03)
- `effective_dim` policzone dla wszystkich QRC i Classical (nowo dodane)
- Concordance % z bootstrap CI

**Czego brakuje:**
- **Heatmapa `effective_dim` per (lr, ws)** — analogiczna do heatmap NMSE, pokazuje gdzie w przestrzeni hiperparametrów model kreuje bogate reprezentacje
- **Empiryczna weryfikacja teoretycznej predykcji**: dyskutowaliśmy że przy `window_size=1` musi być `effective_dim ≤ 2` (bo cechy żyją na 2D wariacji `sin θ, cos θ`). Warto to pokazać empirycznie — to mocny argument że rozumiesz mechanikę modelu

### Cel 3: porównanie z klasycznym ESN

**Co mamy:**
- Tabela porównawcza per-dataset (właśnie dodana)
- CV vs Test scatter z obu modeli
- Concordance osobno dla QRC i Classical

**Czego brakuje:**
- **Heatmapa różnicy QRC − Classical** w przestrzeni (lr, ws) — była w pierwotnym planie (Faza 1.4) ale nie została zrobiona. Wizualnie pokazuje **w którym rogu** przestrzeni hiperparametrów QRC ma przewagę. Mocna karta dla pracy
- **Heatmapy NMSE dla Classical** — mamy tylko dla QRC. Dla uczciwości symetrii warto mieć też klasyczne
- **Test istotności**: czy różnica QRC vs Classical jest statystycznie znacząca? Paired test na per-sub-seed wynikach przy zafiksowanych hiperparametrach

---

## Zaległości z poprzedniego planu

| Pozycja z planu | Status |
|---|---|
| Faza 0 — notebook 00 methodology | ✓ |
| Faza 1.1 — NMSE jako kolumny | ✓ (ale tylko w notebook 04, nie w 02) |
| Faza 1.2 — rozszerzona tabela porównawcza | ✓ (różne miejsca: 04 sekcja 3, 04 sekcja 4) |
| Faza 1.3 — heatmapy NMSE per model per profil | **częściowo** — tylko QRC, brak Classical |
| Faza 1.4 — heatmapa różnicy QRC − Classical | **nie zrobione** |
| Faza 1.5 — tabela ablacji pamięci | **częściowo** — wbudowana w tabelę 4-config, nie samodzielna |
| Faza 2.1 — effective_dim dla Classical | ✓ |
| Faza 2.2 — concordance per profil | ✓ |
| Faza 2.3 — concordance per window_size-bin (kontrolne) | **nie zrobione** — ważne! odpowiada na zarzut confounding |
| Faza 2.4 — bootstrap CI | ✓ |

**Trzy konkretne dziury, które warto zamknąć:** 1.3 (Classical heatmapy), 1.4 (różnica), 2.3 (concordance kontrolne).

---

## Nowe sugestie

### A. Wykres dekompozycji pamięci (Cel 1)

Per profil, słupkowy wykres pokazujący NMSE w trzech reżimach:

```
                    NMSE (test, log scale)
Profile             pure leaky | pure window | combined
Mackey-Glass τ=17   ▓▓▓▓▓▓     | ▓▓▓        | ▓
Mackey-Glass τ=30   ▓▓▓▓▓▓▓    | ▓▓▓        | ▓
Mackey-Glass τ=100  ▓▓▓▓▓▓▓▓   | ▓▓▓▓▓▓     | ▓▓
NARMA-10            ▓▓▓        | ▓▓▓        | ▓
...
```

Czytelnik od razu widzi że dla MG τ=100 ani pure leaky ani pure window same nie wystarczają (ślepa pamięć krótka), ale ich połączenie daje sukces. **Bezpośrednia odpowiedź na Cel 1.**

### B. Heatmapa QRC − Classical (Cel 3, Faza 1.4)

Per profil, kolorowa mapa różnicy: niebieski = QRC lepszy, czerwony = Classical lepszy, biały = remis. Wartości to log10(NMSE_QRC / NMSE_Classical). Pokazuje że QRC dominuje np. przy małych oknach a Classical przy dużych — albo odwrotnie. **To by była najmocniejsza pojedyncza figura w sekcji wyników.**

### C. Heatmapa effective_dim per (lr, ws) (Cel 2)

Analogiczna do heatmap NMSE, ale kolor = effective_dim. Pokazuje krajobraz wymiarowości. Wraz z heatmapą NMSE → reader widzi że niskie NMSE koresponduje z wysokim effective_dim (lub nie, jeśli concordance jest słabe).

### D. Empiryczna weryfikacja sufitu `effective_dim ≤ 2` przy ws=1

Mini-tabela:
```
window_size=1: effective_dim measured: 2 (matches theoretical bound)
window_size=2: effective_dim measured: 5
window_size=10: effective_dim measured: 27
```

Dwie linijki tekstu w pracy, ale stanowczy argument że rozumiesz model.

### E. Concordance kontrolne per window_size-bin (Faza 2.3)

Per profil × window_size (6 × 6 = 36 mini-statystyk). Pokazuje że nawet po kontroli `window_size` (eliminacja confoundera), wyższe `effective_dim` → niższe NMSE. Mocny argument metodologiczny.

### F. Test istotności statystycznej QRC vs Classical (Cel 3)

Per profil, paired Wilcoxon signed-rank na MSE z 11 sub-seedów dla najlepszych hiperparametrów obu modeli. Daje p-value w tabeli "czy QRC jest istotnie lepszy/gorszy". W literaturze magisterskiej często się tego oczekuje.

---

## Rekomendowana priorytetyzacja

| Priorytet | Co | Cel pracy | Szacowany czas |
|---|---|---|---|
| 1 — must | B. heatmapa QRC − Classical | 3 | 30 min |
| 2 — must | A. dekompozycja pamięci (wykres + tabela) | 1 | 1h |
| 3 — strong | 1.3 dokończone — Classical heatmapy NMSE | 3 | 30 min |
| 4 — strong | C. heatmapa effective_dim | 2 | 30 min |
| 5 — nice | E. concordance per window_size-bin | 2 | 30 min |
| 6 — nice | F. test istotności (Wilcoxon) | 3 | 30 min |
| 7 — nice | D. weryfikacja eff_dim ≤ 2 przy ws=1 | 2 | 15 min |

**Must-have (1-3)** to ~2 godziny pracy i pokrywają największe luki w trzech celach. Reszta to "polish" wzmacniający narrację.

---

# Część 2 — Szczegóły testów statystycznych

Trzy różne mechanizmy statystyczne. Dwa wymagają per-sub-seed danych których obecnie nie zachowujemy w CSV, jeden można zrobić post-hoc.

---

## Test 1: Paired Wilcoxon — okno vs leaky integrator

### Co właściwie chcemy stwierdzić

Czy dla danego profilu danych "czysta pamięć okna" (`leakage_rate=1, ws=best`) daje systematycznie inny błąd niż "czysta pamięć leaky" (`ws=1, lr=best`). "Systematycznie" = z wykluczeniem szumu losowego pomiędzy sub-seedami.

### Mechanika testu

Dla każdej z **11 par sub-seedów** w tym samym profilu:

```
sub_seed_i (i = 0..10):
   MSE_window_i  = test MSE z "pure window" przy sub_seed_i
   MSE_leaky_i   = test MSE z "pure leaky"  przy sub_seed_i
   diff_i        = MSE_window_i − MSE_leaky_i
```

Otrzymujemy 11 wartości `diff_i`. Wilcoxon signed-rank test bada czy mediana tych różnic jest istotnie różna od zera:

- H₀: rozkład różnic jest symetryczny wokół 0 (nie ma systematycznej przewagi)
- H₁: jest skośny w jedną stronę

Test daje `p-value`. Jeśli `p < 0.05` → odrzucamy H₀, jeden mechanizm pamięci jest istotnie lepszy.

### Dlaczego **paired** a nie zwykły t-test

- **Paired** uwzględnia że sub_seed_i wprowadza tę samą losowość do obu konfiguracji — usuwa "szum z inicjalizacji rezerwuaru" jako wspólny czynnik
- **Wilcoxon** zamiast t-test bo MSE w skali log nie jest normalny + 11 prób to mało żeby zakładać CLT
- **Signed-rank** zamiast sign-test bo wykorzystuje też wielkość różnicy, nie tylko znak

### Co dostajesz na wyjściu

Tabela per profil:
```
Profile          p-value    Winner          Konkluzja
MG τ=17          0.003      pure leaky      lepszy istotnie
MG τ=100         0.812      brak różnicy    nie ma istotnej
NARMA-10         0.001      pure window     lepszy istotnie
...
```

To są **konkretne stwierdzenia** do wstawienia w sekcji wyników: "Dla profilu Mackey-Glass τ=100 nie znaleziono istotnej różnicy między mechanizmami pamięci (p=0.81)", itd.

### ❌ Problem: nie mamy per-sub-seed danych w CSV

Obecnie `experiment.py` agreguje 11 sub-seedów do `median_test_mse`, `std_test_mse`, `cv_test_mse` i odrzuca surowe wartości. Wilcoxon potrzebuje **11 indywidualnych liczb**, nie mediany.

**Co trzeba zrobić:**

1. Zmodyfikować `run_qrc_experiment_with_cv` i `run_classical_experiment_with_cv` aby zwracały dodatkowo listę 11 wartości test MSE per sub-seed (`test_scores_raw`)
2. Zapisać tę listę jako kolumnę w CSV (np. jako serializowany JSON lub osobne kolumny `test_mse_seed_0`, `test_mse_seed_1`, ...)
3. **Puścić nową pętlę** — niestety nie ma post-hoc obejścia

To jedyna sytuacja w której zmiana wymaga ponownej pętli.

---

## Test 2: Paired Wilcoxon — QRC vs Classical

### Mechanika

Analogiczna do Testu 1, ale porównujemy dwa modele zamiast dwóch konfiguracji jednego modelu:

```
Wybieramy "matched" hiperparametry — np. window_size=10, leakage_rate=0.9
  (te same dla QRC i Classical, możliwe dzięki ostatniej refaktorze gridów)
Dla i = 0..10:
   MSE_QRC_i        = test MSE QRC przy (ws=10, lr=0.9, sub_seed=i)
   MSE_Classical_i  = test MSE Classical przy (ws=10, lr=0.9, sub_seed=i)
   diff_i           = MSE_QRC_i − MSE_Classical_i
Wilcoxon na 11 diff_i  → p-value
```

### Subtelność: "sub_seed" znaczy co innego dla obu modeli

- W QRC sub_seed=2025+i seeduje losowe wagi obwodu kwantowego (`weights`, `biases`)
- W Classical sub_seed=2025+i seeduje `W_in` i `W_res`

Czyli "sub_seed_i" w QRC vs Classical to **nie ta sama losowość**. Ale paired pairing po `i` ma sens jako "porównujemy modele które dostały i-te źródło stochastyczności" — to standardowe ujęcie w benchmarkach.

### Co dostajesz

```
Profile          p-value    Winner       NMSE_QRC      NMSE_Classical
MG τ=17          0.001      QRC          1.0e-5        2.5e-5
MG τ=100         0.015      QRC          6.8e-6        1.2e-5
NARMA-10         0.234      brak różnicy 9.0e-3        9.5e-3
...
```

**Bardzo silny argument do dyskusji w pracy**: zamiast pisać "QRC ma niższe NMSE niż Classical na 4 z 6 profili" piszesz "QRC ma istotnie niższe NMSE niż Classical na 4 z 6 profili (p < 0.05)".

### ❌ Ten sam problem co Test 1: brak per-sub-seed w CSV

Wymaga tych samych modyfikacji `experiment.py` i ponownej pętli. Jeśli i tak robisz to dla Testu 1, robisz dla obu naraz — żaden dodatkowy koszt.

---

## Test 3: Concordance per `window_size`-bin (kontrolne)

### Co właściwie jest problemem (confounder)

Globalna concordance(eff_dim, NMSE) per profil mierzy "czy wyższa wymiarowość daje lepszy wynik". Ale w naszym gridzie:

- Większe `window_size` → więcej qubitów → wyższy sufit `effective_dim` (3·n_qubits)
- Większe `window_size` → też (zwykle) lepsze NMSE bo model widzi więcej historii

Czyli wysoka concordance globalna może mówić **"większe okno daje wyższy eff_dim ORAZ niższe MSE"**, a nie **"wyższy eff_dim daje niższe MSE niezależnie od okna"**. To jest klasyczny problem **confoundera** w statystyce — `window_size` to zmienna ukryta wpływająca jednocześnie na obie analizowane zmienne.

### Jak działa kontrolowanie

Dzielimy dane na biny po `window_size` i liczymy concordance **w obrębie każdego binu osobno**:

```
Dla profilu Mackey-Glass τ=17:
  rows with window_size=1:   concordance(eff_dim, NMSE) wśród 6 wartości lr → 72%, CI=[60%, 85%]
  rows with window_size=2:   concordance(eff_dim, NMSE) wśród 6 wartości lr → 68%, CI=[55%, 80%]
  rows with window_size=4:   concordance(eff_dim, NMSE) wśród 6 wartości lr → 75%, CI=[62%, 87%]
  rows with window_size=6:   ...
  rows with window_size=8:   ...
  rows with window_size=10:  ...
```

W każdym binie `window_size` jest **stały**, więc eliminujemy go jako confounder. Różnice w `effective_dim` w obrębie binu pochodzą wyłącznie z innych zmiennych (głównie `leakage_rate`, w QRC też w mniejszym stopniu inicjalizacji `weights` przez seed).

### Co to mówi merytorycznie

- **Jeśli concordance pozostaje > 50% w każdym binie**: relacja "wyższy eff_dim → niższe NMSE" jest **autentyczna**, nie tylko artefakt rozmiaru okna. Mocna konkluzja: model wykorzystuje wymiarowość kwantową niezależnie od ilości qubitów
- **Jeśli concordance spada do ~50% w binach**: globalny efekt był głównie napędzany przez `window_size`. Wniosek: większe okno pomaga, ale "wymiarowość kwantowa" sama w sobie nie jest sterownikiem wyników

Druga konkluzja byłaby też wartościowa naukowo, choć mniej spektakularna.

### Format tabeli LaTeX

```
Profile           ws=1        ws=2        ws=4        ws=6        ws=8        ws=10       Median
MG τ=17           72% [60,85] 68% [55,80] 75% [62,87] 71% [58,84] 73% [60,86] 70% [57,83] 71%
...
```

Wartość mediany z 6 binów daje syntetyczną liczbę "po kontroli window_size".

### ✓ Post-hoc — bez ponownej pętli

Concordance jest funkcją (effective_dim, NMSE) per wiersz. Mamy już oba w CSV (po dodaniu w notebook 03). Filtrowanie po `window_size` i ponowne liczenie concordance + bootstrap = sekundy.

---

## Podsumowanie kosztów i wykonalności

| Test | Co wymaga | Czas implementacji |
|---|---|---|
| Test 1 (Wilcoxon memory) | **Modyfikacja `experiment.py` + ponowna pętla** ~3h | + 1h kodowania |
| Test 2 (Wilcoxon QRC vs Classical) | Razem z Testem 1, ten sam koszt | + 30 min kodowania |
| Test 3 (concordance kontrolne) | **Post-hoc na obecnym CSV** | 30 min kodowania |

**Test 3 jest tani i wartościowy** — rekomendowałbym wprowadzić od razu. Testy 1 i 2 wymagają planowania (kolejna pętla), ale gdyby ją puścić, dostajesz w pakiecie dwa mocne testy istotności statystycznej do pracy.

Jeśli zdecydujesz się na nową pętlę dla Testów 1 i 2 — modyfikacja `experiment.py` to dodanie ~5 linii (return surowych list per sub-seed) i kolumny serializującej je w CSV. Po nowej pętli odpalasz analizy w istniejących notebookach bez zmian.
