# Protocollo sperimentale

1. Definire task, etichette, unità biologica e insieme di specie. Conservare versioni,
   sorgenti, licenze e hash dei file scaricati.
2. Assegnare train/validation/test a genomi, cromosomi, soggetti o cluster di omologia
   **prima** di generare frammenti. Per finestre sullo stesso cromosoma usare regioni
   disgiunte con margine e controllare le coordinate. Il controllo software di duplicati
   e gruppi non rileva omologia approssimata o sovrapposizioni senza metadati adeguati.
3. Congelare gli stessi record e split per tutti i metodi. Annotare la possibile
   sovrapposizione col pretraining: OpenGenome2 è nel training di Evo2 e NTv3.
   Per NTv3 post controllare anche la sovrapposizione con le annotazioni supervisionate.
4. Scegliere una lunghezza comune `window_bp` supportata da tutti i modelli. La pipeline
   attraversa l'intera sequenza con finestre contigue non sovrapposte e aggrega gli
   embedding pesandoli per numero di basi. L'ultima finestra può essere più corta.
   Questo misura una rappresentazione locale aggregata, non l'intero contesto nativo
   dei modelli long-context. Fare esperimenti separati per studiare il long-context.
5. La baseline conta k-mer overlapping sull'intera sequenza: ignora finestre con N,
   normalizza per numero di k-mer validi e può unire reverse complement canonici.
   Per sequenze più corte di k o senza k-mer validi restituisce un vettore zero;
   filtrare o rendicontare questi casi. La dimensione cresce esponenzialmente con k.
6. Usare probe identici: StandardScaler stimato solo su train, logistic regression
   per classificazione o Ridge per regressione. Selezionare C/alpha tra
   0.01, 0.1, 1, 10, 100 usando balanced accuracy o MSE su validation.
   Non rifittare su train+validation; valutare una volta su test. In classificazione
   ogni split deve contenere tutte le classi, per evitare metriche non definite.
7. Rendicontare accuracy, balanced accuracy, macro-F1, MCC; per binario anche AUROC
   e average precision. La classe positiva è `classes_[1]` nell'ordinamento delle
   etichette stringa di scikit-learn. Per regressione: MAE, RMSE, R².
8. Per confronti scientifici ripetere su split biologici indipendenti fissati a priori,
   riportando distribuzione delle metriche e intervalli di confidenza. Il seed del
   probe non sostituisce questa variabilità. Multilabel, task per base, fine-tuning,
   clustering e intervalli bootstrap non sono implementati in questa versione.

Non mescolare scelta di k, layer, specie, lunghezza o checkpoint con l'osservazione
dei risultati test: queste scelte devono essere predefinite o selezionate su validation.
Per esperimenti finali fissare commit backend, revisioni pesi, dipendenze e hardware.
I manifest raccolgono l'ambiente osservato, ma non sono un lockfile dei runtime GPU.

Gli input sono iterati durante la lettura, ma `prepare` e il runner materializzano
il subset selezionato e le features in RAM: usare `limit` per la preparazione e
shard esterni per corpus grandi. Non caricare l'intero OpenGenome2 in questa pipeline.
