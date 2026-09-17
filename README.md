# Explainable E-Commerce Analytics via RAG

Il progetto analizza ordini e recensioni del dataset pubblico Olist. Spark
calcola indicatori su ordini, rating e consegne; il sistema RAG recupera
recensioni pertinenti e le usa per costruire una spiegazione in italiano.
Metriche, testi di supporto e sintesi rimangono separati e verificabili.

## Architettura

```text
CSV Olist
   |
   v
HDFS locale oppure Amazon S3
   |
   v
Apache Spark -----------------> KPI e confronti temporali
   |
   v
recensioni -> E5-small -> Chroma -> reranker multilingue
                                         |
                                         v
                                contesto per l'LLM
                                         |
                                         v
                                  app Streamlit
```

Spark è la fonte dei valori numerici. Le recensioni aggiungono esempi testuali,
ma non vengono usate per dimostrare rapporti di causa ed effetto.

Le tecnologie principali sono Spark 3.5.5, PySpark, HDFS, Chroma, Streamlit e
Docker Compose. Il modello predefinito per gli embedding è
`intfloat/multilingual-e5-small`. GPT-5.6 Terra è il generatore principale,
mentre Qwen 2.5 1.5B Q4 permette l'esecuzione locale tramite Ollama. La
traduzione può usare Google Cloud Translation oppure OPUS-MT.

## Avvio rapido

Copiare i nove CSV Olist in `data/raw/olist/`, quindi eseguire:

```powershell
Copy-Item .env.example .env
.\scripts\start.ps1
.\scripts\run_pipeline.ps1 -Pipeline phase1
.\scripts\rag\prepare_rag.ps1
.\scripts\rag\index_reviews.ps1
.\scripts\rag\start_ui.ps1
```

L'applicazione sarà disponibile su http://localhost:8501. La prima
indicizzazione scarica i modelli necessari; le esecuzioni successive
riutilizzano dati, modelli e cache persistenti.

Comandi utili:

```powershell
.\scripts\status.ps1       # stato dei servizi
.\scripts\smoke_test.ps1   # controllo rapido
.\scripts\stop.ps1         # arresto senza eliminare i volumi
```

Per riavviare l'ambiente senza ricostruire l'immagine:

```powershell
.\scripts\start.ps1 -NoBuild
```

Non utilizzare `docker compose down -v` se si vogliono conservare HDFS,
Chroma, Ollama e le cache.

## Configurazione

`.env.example` contiene la configurazione iniziale. La copia locale `.env`
permette di scegliere provider, porte e risorse senza pubblicare credenziali.

Le variabili principali sono:

- `APP_ENV`, per scegliere l'ambiente;
- `LLM_PROVIDER`, per scegliere Ollama oppure OpenAI;
- `TRANSLATION_PROVIDER`, per scegliere la traduzione locale oppure Google;
- `OPENAI_API_KEY` e `GOOGLE_TRANSLATE_API_KEY`, entrambe opzionali.

La configurazione predefinita funziona localmente con Qwen e OPUS-MT.

Interfacce principali:

| Servizio | Indirizzo |
|---|---|
| Streamlit | http://localhost:8501 |
| HDFS NameNode | http://localhost:9870 |
| Spark Master | http://localhost:18080 |
| Chroma | http://localhost:8000 |

## Preparazione e analisi dei dati

La pipeline verifica i CSV, li carica in HDFS e produce dataset Parquet
tipizzati:

```powershell
.\scripts\run_pipeline.ps1 -Pipeline phase1
```

I file originali non vengono modificati. Gli output principali sono
`orders_enriched`, `reviews_enriched`, `review_order_links`,
`review_themes` e `rag_documents`.

Il corpus RAG contiene 42.114 recensioni con contenuto informativo. I testi
formati soltanto da simboli, numeri o emoji restano disponibili per le metriche
basate sul rating, ma non vengono indicizzati.

Per rigenerare l'analisi esplorativa:

```powershell
.\scripts\analytics\run_eda.ps1
```

Esempi di query Spark:

```powershell
.\scripts\analytics\run_analytics.ps1 -QueryId overall

.\scripts\analytics\run_analytics.ps1 `
  -QueryId office-furniture `
  -ProductCategory office_furniture
```

Categoria, stato e periodo possono essere combinati. Se viene indicato un
periodo, il sistema calcola anche il confronto con i tre mesi precedenti.

## RAG e applicazione

Preparazione del corpus e indicizzazione:

```powershell
.\scripts\rag\prepare_rag.ps1
.\scripts\rag\index_reviews.ps1
```

Esempio di ricerca:

```powershell
.\scripts\rag\query_reviews.ps1 `
  -Question "Il mio ordine non è mai arrivato" `
  -QueryId non-delivery `
  -TopK 5
```

Il retrieval recupera 50 candidati da Chroma, applica eventuali filtri e usa il
reranker per selezionare i primi cinque risultati.

Flusso completo con metriche Spark e generazione:

```powershell
.\scripts\rag\explain.ps1 `
  -Question "Perché il rating è diminuito a marzo 2018?" `
  -QueryId march-rating-drop
```

L'interfaccia Streamlit si avvia con:

```powershell
.\scripts\rag\start_ui.ps1
```

La pagina mostra l'ambito riconosciuto, le metriche Spark, l'eventuale confronto
temporale, i temi e le recensioni di supporto. Se le evidenze recuperate non sono
adatte alla domanda, il sistema lo segnala senza forzare una spiegazione.

## Valutazione

La valutazione principale usa 50 domande realistiche e i primi cinque risultati.
E5-small con filtri e reranker ha ottenuto:

- Precision@5: 0,652;
- Hit Rate@5: 0,980;
- MRR@5: 0,791;
- nDCG@5: 0,842.

Quarantanove domande su cinquanta presentano almeno una recensione pertinente.

Per ricalcolare le metriche disponibili:

```powershell
.\scripts\evaluation\evaluate_retrieval_operational.ps1 -Mode score
.\scripts\evaluation\evaluate_themes.ps1 -Mode score
```

Nel confronto controllato E5-base ha migliorato la Precision@5 di 0,020, ma
richiede più memoria e produce vettori più grandi. E5-small rimane quindi il
modello predefinito.

## Scalabilità

Il benchmark locale usa fattori 1×, 5×, 10×, 25×, 50× e 100×:

```powershell
.\scripts\scalability\run_scalability.ps1 -Mode all
```

Ogni configurazione comprende una prova di riscaldamento e tre misurazioni su
aggregazione, join e join multipli. Al fattore 100× i quattro dataset
contengono complessivamente 34.048.000 righe e occupano circa 3,5 GiB in HDFS.

La pipeline Spark è stata verificata anche su Amazon EMR con dati in S3. Sulle
configurazioni confrontate, l'accelerazione osservata rispetto all'ambiente
locale è stata compresa tra 2,67× e 6,11×. L'esecuzione cloud è opzionale e
richiede AWS CLI, un profilo valido e un bucket dedicato:

```powershell
.\scripts\aws\run_aws_validation.ps1 `
  -Mode All `
  -Profile my-profile `
  -Bucket my-unique-bucket
```

## Struttura del repository

```text
config/                 configurazioni Spark, HDFS e ambienti
docker/                 immagini e configurazioni Docker
scripts/
  analytics/            analisi esplorative e query Spark
  rag/                  indicizzazione, retrieval e interfaccia
  evaluation/           benchmark e confronti
  scalability/          prove di scalabilità
  aws/                  esecuzione su Amazon EMR
src/ecommerce_rag/
  analytics/            KPI e analisi Spark
  app/                  applicazione Streamlit
  cloud/                supporto AWS
  common/               configurazione condivisa
  evaluation/           metriche di valutazione
  ingestion/            caricamento dei CSV
  preprocessing/        pulizia e costruzione dei dataset
  rag/                  documenti, temi, embedding e retrieval
  scalability/          benchmark Spark
tests/                  test unitari e di integrazione
```

Dati originali, credenziali, modelli scaricati e output generati non sono inclusi
nel repository.

## Test e problemi comuni

Suite completa:

```powershell
docker compose exec -T app python3 -m pytest -q
```

- Se Docker non parte, avviare Docker Desktop e ripetere `start.ps1`.
- Se una porta è occupata, modificarla in `.env`.
- Se un servizio non è pronto, usare `docker compose ps` e
  `docker compose logs <servizio>`.
- Se sono cambiate dipendenze o immagini, eseguire `start.ps1` senza
  `-NoBuild`.
- Una nuova indicizzazione aggiorna soltanto i documenti modificati.
