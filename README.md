# E-commerce RAG Analytics

Progetto Big Data sul dataset Olist. Spark quantifica i fenomeni osservati negli
ordini e nelle recensioni; il sistema RAG recupera evidenze testuali utili a
formulare spiegazioni supportate dai dati, senza presentarle come cause certe.

## Prerequisiti

- Windows con PowerShell;
- Docker Desktop con container Linux e Docker Compose v2;
- almeno 8 GB di memoria assegnata a Docker (consigliati);
- porte locali 7077, 8000, 9000, 9864-9865, 9870 e 18080-18082 libere.

Non è necessario installare Java, Hadoop, Spark o le dipendenze Python sul
computer: sono inclusi nei container.

## Primo avvio e ripresa del lavoro

Da PowerShell, nella directory del progetto:

```powershell
.\scripts\start.ps1
```

Al primo avvio lo script:

1. crea `.env` a partire da `.env.example`;
2. avvia Docker Desktop, se necessario;
3. costruisce il container applicativo;
4. avvia HDFS, Spark, Chroma e l'applicazione Python;
5. crea in HDFS `/data/raw`, `/data/processed`, `/data/curated`, `/data/scaled`
   e `/data/outputs`.

Negli avvii successivi gli stessi comandi riutilizzano i volumi persistenti. Per
evitare una ricostruzione dell'immagine applicativa quando le dipendenze non sono
cambiate:

```powershell
.\scripts\start.ps1 -NoBuild
```

Per controllare i servizi:

```powershell
.\scripts\status.ps1
```

Per verificare l'intera infrastruttura:

```powershell
.\scripts\smoke_test.ps1
```

Per terminare la sessione di lavoro conservando tutti i dati:

```powershell
.\scripts\stop.ps1
```

`stop.ps1` non elimina i volumi. Non eseguire `docker compose down -v` se si
vogliono conservare HDFS e Chroma.

## Servizi locali

| Servizio | Indirizzo dal computer host | Indirizzo dai container |
|---|---|---|
| HDFS NameNode UI | http://localhost:9870 | `namenode:9870` |
| HDFS RPC | `hdfs://localhost:9000` | `hdfs://namenode:9000` |
| DataNode 1 UI | http://localhost:9864 | `datanode-1:9864` |
| DataNode 2 UI | http://localhost:9865 | `datanode-2:9864` |
| Spark Master UI | http://localhost:18080 | `spark-master:8080` |
| Spark Worker 1 UI | http://localhost:18081 | `spark-worker-1:8081` |
| Spark Worker 2 UI | http://localhost:18082 | `spark-worker-2:8081` |
| Chroma | http://localhost:8000 | `chroma:8000` |

Il Driver Spark viene eseguito nel container `app`. I due Worker sono indipendenti
dai due DataNode: Spark decide dove eseguire i task e HDFS decide dove conservare
i blocchi.

## Configurazione

- `.env`: risorse, porte e versioni delle immagini locali; non versionato;
- `config/environments/local.yml`: URI e servizi usati con Docker Compose;
- `config/environments/aws.yml`: scheletro per EMR/S3 da completare nella Fase 10;
- `config/hdfs/`: configurazione HDFS, incluso replication factor 1;
- `config/spark/`: configurazione del Driver e del cluster Spark.

La logica Python legge il profilo indicato da `APP_ENV`; URI e hostname non devono
essere inseriti direttamente nella logica applicativa.

## Smoke test

Lo smoke test esegue due percorsi:

```text
file di prova -> HDFS -> Spark cluster -> Parquet -> HDFS
recensioni di prova -> embedding deterministici -> Chroma -> query -> risultato
```

Gli embedding dello smoke test servono solo a verificare il collegamento con
Chroma. Non sono il modello multilingue definitivo del RAG.

## Ingestion e preprocessing Olist

Collocare i nove CSV originali in `data/raw/olist/`, quindi eseguire:

```powershell
.\scripts\run_pipeline.ps1 -Pipeline phase1
```

Lo script verifica i file e li carica senza sovrascrivere contenuti raw gia
presenti in HDFS. Se un file con lo stesso nome ha un hash SHA-256 diverso,
l'esecuzione viene interrotta. La pipeline PySpark produce:

- tabelle tipizzate e aggregazioni in `/data/processed/olist`;
- `orders_enriched`, `reviews_enriched` e `review_order_links` in
  `/data/curated/olist`;
- metriche di qualita, profili dei null e contratti tecnici in
  `/data/outputs`.

Le granularita, le regole di deduplicazione e le feature sono descritte nel
[data contract della fase 1](docs/data-contracts/phase1.md).

Per verificare sia le guardie anti-fan-out sia i Parquet realmente persistiti:

```powershell
docker compose exec -T app python3 -m pytest -q `
  tests/preprocessing/test_pipeline_guards.py `
  tests/integration/test_phase1_outputs.py
```

## Dati e documenti locali

`data/`, `materiale_corso/`, `reports/`, `ROADMAP.md`, `report.md` e `AGENTS.md`
sono esclusi da Git. Il dataset resta quindi locale e non viene incorporato nelle
immagini Docker grazie anche a `.dockerignore`. Al termine del progetto si potrà
decidere quali risultati di `reports/` rendere pubblici.

## Risoluzione rapida dei problemi

- Se Docker non parte, avviare Docker Desktop e ripetere `start.ps1`.
- Se una porta è occupata, modificarla nel file `.env` locale.
- Se un servizio non è pronto, eseguire `docker compose ps` e
  `docker compose logs <servizio>`.
- Se sono state cambiate dipendenze o immagini, avviare senza `-NoBuild`.
