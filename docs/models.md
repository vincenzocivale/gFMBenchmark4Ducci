# Ambienti e backend

Usare Python 3.11 e ambienti separati. Le dipendenze CUDA vanno scelte in base a
driver e GPU del cluster. Non viene installato Evo2 come dipendenza del core.

| Metodo | Backend | Rappresentazione |
| --- | --- | --- |
| k-mer | NumPy | frequenze normalizzate su finestre valide A/C/G/T |
| NTv3 650M pre | JAX ufficiale | media pesata delle posizioni transformer del layer scelto |
| NTv3 650M post | JAX ufficiale | stesso layer, con conditioning di specie esplicito |
| Evo2 7B (configurabile) | pacchetto Evo2 ufficiale | media dell'attivazione del modulo scelto |
| DNABERT-2 117M | Hugging Face | ultimo hidden state, media senza special token/padding |

## NTv3

Installare il [repository ufficiale](https://github.com/instadeepai/nucleotide-transformer)
seguendo le istruzioni NTv3 e il runtime JAX appropriato; installare poi questa repo
nello stesso ambiente con `uv pip install -e .`. Consultare la
[documentazione NTv3](https://github.com/instadeepai/nucleotide-transformer/blob/main/docs/nucleotide_transformer_v3.md).

I loader usati sono `get_pretrained_ntv3_model` e `get_posttrained_ntv3_model`.
Il post richiede una specie/assembly supportata: il valore `human` nel preset
è un esempio, da sostituire prima di usare dati di altre specie. Per corpus misti,
creare esperimenti omogenei per specie o estendere l'adapter con conditioning per record.
Non usare una condizione umana per specie sconosciute senza dichiararlo come esperimento.

La lunghezza deve essere multipla di `2**num_downsamples` (128 nei modelli principali).
L'adapter aggiunge N alla fine di ogni window e pesa la posizione finale in base
alle basi reali: il padding può comunque influenzare le attivazioni contestuali.
Il layer 10 è una scelta iniziale comune, non un layer ottimizzato. L'API ufficiale
scarica codice remoto e non espone qui una revisione dei pesi: per riproducibilità
archiviare snapshot HF e commit del backend e usare un ambiente/cache immutabile.

## Evo2

Seguire l'[installazione ufficiale Evo2](https://github.com/ArcInstitute/evo2),
inclusi i requisiti CUDA e Vortex. Scegliere il checkpoint e verificare che il modulo
`layer` esista nel modello: `blocks.28` è il preset per il 7B, non universale.
L'estrazione usa hook ufficiali `return_embeddings=True, layer_names=[...]`.
Le sequenze vengono elaborate singolarmente per evitare padding. La memoria richiesta
dipende da checkpoint, lunghezza e runtime; il 7B richiede un ambiente GPU adeguato.

## DNABERT-2 e altri Hugging Face

```bash
uv pip install -e '.[hf]'
gfm-bench run configs/opengenome.yaml --models dnabert2
```

Il [quickstart ufficiale](https://github.com/MAGICS-LAB/DNABERT_2) richiede `BertConfig`
esplicita per Transformers successivo alla 4.28; l'adapter applica questa convenzione.
Il codice remoto DNABERT-2 può richiedere versioni specifiche di Transformers/Triton
e ulteriori dipendenze: fissare l'ambiente dopo la verifica sul cluster.
`trust_remote_code` è esplicito nei preset e disabilitato per default nel backend
generico. Impostare `revision` a un commit verificato del modello prima di produrre risultati.
Non viene effettuato troncamento silenzioso: il superamento di `max_tokens` genera errore.

## Esecuzione separata

```bash
gfm-bench run configs/opengenome.yaml --models ntv3_650m_pre
gfm-bench run configs/opengenome.yaml --models ntv3_650m_post
gfm-bench run configs/opengenome.yaml --models evo2_7b
gfm-bench run configs/opengenome.yaml --models dnabert2
```

La compatibilità con le API è implementata; il funzionamento con checkpoint reali,
le dipendenze esatte e le prestazioni GPU devono essere verificati sul cluster.
