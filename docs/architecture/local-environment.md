# Ambiente locale

```text
Python Application / Spark Driver
        |                 |
        | spark-submit    | HTTP
        v                 v
Spark Master          Chroma
   |      |
Worker 1 Worker 2
   \      /
    HDFS client
        |
    NameNode
     /    \
DataNode 1 DataNode 2
```

Il Driver viene avviato nel container `app`. Master e Worker coordinano il
calcolo, mentre NameNode e DataNode coordinano lo storage: non esiste un legame
fisso fra Worker 1/DataNode 1 o Worker 2/DataNode 2.

HDFS usa replication factor 1 per scelta progettuale: ogni blocco ha una sola
copia. Questo riduce l'uso di spazio ma non offre tolleranza alla perdita di un
DataNode ed è adatto esclusivamente all'ambiente didattico locale.

I volumi Docker di HDFS e Chroma sono persistenti. Il profilo AWS sostituirà HDFS
con S3 come storage persistente, mantenendo invariata la logica PySpark.
