# Genomic Foundation Model Benchmark

Benchmark modulare di **rappresentazioni congelate**: NTv3 650M pre/post,
Evo2, DNABERT-2 e frequenze k-mer. Ogni rappresentazione alimenta la stessa
famiglia di probe lineari, con selezione su validation e valutazione finale su test.
Supporta classificazione binaria/multiclasse e regressione a singolo target.

## Avvio rapido (CPU)

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e '.[test]'
pytest -q
gfm-bench run configs/demo.yaml
```

La demo usa 14 sequenze sintetiche con diversa composizione: verifica la pipeline,
non misura capacità biologiche. Nessun peso o dataset viene scaricato dalla demo.

## Organizzazione

```text
configs/                  esperimenti YAML, percorsi relativi al file YAML
data/examples/            fixture sintetica versionata
data/raw/                 download locali, esclusi da git
data/processed/           subset annotati, esclusi da git
src/gfm_benchmark/
  data.py                 lettura, normalizzazione, controlli split
  encoders.py             protocollo Encoder e registro backend
  evaluation.py           probe e metriche
  runner.py               windowing, cache, manifest e risultati
  cli.py                  comandi prepare / embed / run
tests/                    verifiche CPU senza download di modelli
docs/                     protocollo sperimentale e ambienti GPU
outputs/                  artefatti generati, esclusi da git
```

## Dati del collega e OpenGenome

Il formato canonico è JSONL (anche CSV in ingresso), una riga per sequenza:

```json
{"id":"seq_001","sequence":"ACGTACGT","label":"promoter","split":"train","group":"genome_001"}
```

`split` vale `train`, `validation` o `test`; `group` identifica il genoma, paziente,
specie o cluster di omologia da mantenere in un solo split. Per la regressione
`label` contiene il valore numerico. Non vengono creati split casuali implicitamente.
Sono rifiutati ID duplicati e sequenze uguali o reverse complement tra split;
anche i gruppi, quando specificati, devono essere disgiunti.

Assumiamo OpenGenome2 di Arc Institute; il loader funziona anche con OpenGenome
precedente o dataset privati esportati in FASTA, CSV o JSONL, anche `.gz`.
OpenGenome2 contiene FASTA e JSONL preformattati per pretraining: usare preferibilmente
FASTA, perché i JSONL possono contenere tag e token speciali. Il loader accetta
solo A/C/G/T/N e rifiuta esplicitamente il testo con tag.

Scaricare un subset selezionato dalla [pagina ufficiale OpenGenome2](https://huggingface.co/datasets/arcinstitute/opengenome2),
poi unirlo a metadati del collega indicizzati dallo stesso `id`:

```bash
gfm-bench prepare data/raw/subset.fasta.gz data/processed/opengenome.jsonl \
  --metadata data/raw/labels.csv --limit 10000
# JSONL senza tag con DNA nella colonna text:
gfm-bench prepare data/raw/subset.jsonl.gz data/processed/unlabelled.jsonl \
  --sequence-column text --limit 1000
gfm-bench run configs/opengenome.yaml --models kmer_6
```

`labels.csv`: colonne `id,label,split,group`. Il primo token dell'header FASTA è l'ID.
Senza un ID sorgente, il loader usa `record_0`, `record_1`, ecc.: mantenere stabile
l'ordine e preferire ID espliciti per il join. `prepare` non sovrascrive file esistenti.
Il corpus non fornisce automaticamente etichette downstream: per dati senza etichette
usare `gfm-bench embed CONFIG.yaml`, che salva soltanto le rappresentazioni.

OpenGenome2 è stato usato nel pretraining di Evo2 e NTv3: risultati su sue sequenze
non dimostrano generalizzazione fuori dal pretraining. Vedere il
[protocollo](docs/protocol.md) per holdout e contaminazioni.

## Modelli ed estensibilità

I preset sono in [configs/opengenome.yaml](configs/opengenome.yaml). Gli ambienti
per [NTv3, Evo2 e DNABERT-2](docs/models.md) richiedono installazioni separate.
Eseguire un modello per processo con `--models NOME` permette di usare ambienti
diversi e condividere dataset e output.

Un nuovo encoder implementa `encode(list[str]) -> np.ndarray` di forma `[N,D]`
e si registra in `encoders.REGISTRY`. Il backend `hf` copre modelli compatibili
con `AutoModel` che restituiscono hidden state `[batch,tokens,features]`.
Architetture con conditioning, output o tokenizzazione diversi richiedono un adapter.

## Risultati

Ogni run salva `outputs/<esperimento>/<modello>/<sha256>/` con `features.npy`,
`manifest.json`, `metrics.json` e `predictions.jsonl`. Il manifest include configurazione,
ordine degli ID, dimensione delle features e versioni dei pacchetti installati.
`summary-*.json` raccoglie i risultati dei modelli selezionati in quella invocazione.
La cache dipende dal contenuto dei dati, configurazione encoder, window e versione
del benchmark. Cambiare versione o eliminare la specifica cache dopo modifiche
al codice degli encoder; fissare revisioni e ambienti per gli esperimenti finali.

I backend FM sono implementati sulle API ufficiali ma richiedono validazione
con i pesi reali sul sistema GPU di destinazione; i test CPU coprono dati,
baseline, aggregazione, cache e valutazione, senza eseguire i grandi modelli.

Nell'ambiente iniziale sono stati eseguiti controlli di sintassi e smoke check
NumPy su dati, leakage, baseline, windowing e cache. La suite pytest e il probe
supervisionato non sono stati eseguiti qui: scikit-learn non è installato e il
download delle dipendenze è impedito dalla risoluzione DNS.
