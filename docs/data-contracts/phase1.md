# Data contract della fase 1

La pipeline conserva separati il livello `raw`, il livello `processed` e il
livello `curated`. I CSV raw vengono copiati in HDFS senza modificarli e ne viene
verificato l'hash SHA-256. Tutti gli altri dataset sono scritti in Parquet.

Lo schema Spark completo, inclusi tipi e nullability, viene esportato a ogni
esecuzione in `/data/outputs/contracts/phase1`. Questo documento descrive le
regole semantiche che lo schema tecnico da solo non esprime.

## Dataset processed di base

| Dataset | Granularita di una riga | Chiave logica | Regola principale |
|---|---|---|---|
| `orders` | ordine | `order_id` | date convertite in timestamp |
| `order_items` | posizione di un ordine | `(order_id, order_item_id)` | prezzi e trasporto in `decimal(18,2)` |
| `payments` | sequenza di pagamento | `(order_id, payment_sequential)` | valore in `decimal(18,2)` |
| `reviews` | associazione recensione-ordine | `(review_id, order_id)` | titolo e messaggio originali preservati |
| `products` | prodotto | `product_id` | categoria inglese, con fallback portoghese |
| `customers` | cliente associato a un ordine | `customer_id` | stato normalizzato in maiuscolo |
| `sellers` | venditore | `seller_id` | stato normalizzato in maiuscolo |
| `geolocation` | coordinata osservata per CAP | nessuna chiave univoca | osservazioni raw conservate |
| `category_translation` | categoria portoghese | `product_category_name` | testo normalizzato in minuscolo |

Le chiavi dichiarate sono obbligatorie. I campi non chiave mantengono i null
presenti nel dataset sorgente; i profili dei null sono salvati in
`/data/outputs/quality/phase1/null_profile`.

## Aggregazioni processed

### `items_by_order`

Una riga per `order_id`. Contiene conteggi di item, prodotti, venditori e
categorie, liste distinte ordinate, somme di prezzo e trasporto e
`order_value = item_subtotal + freight_total`. `order_category` e valorizzata
solo se tutti gli item hanno una categoria nota e la categoria e unica.

### `payments_by_order`

Una riga per `order_id`. Riporta numero di sequenze, tipi distinti, lista dei
tipi, massimo numero di rate e somma dei valori di pagamento.

### `geolocation_by_zip`

Una riga per `geolocation_zip_code_prefix`. Latitudine e longitudine sono le
medie delle coordinate osservate. Citta e stato rappresentativi sono la coppia
piu frequente; i pareggi sono risolti in ordine alfabetico, quindi l'algoritmo e
deterministico.

### `reviews_canonical`

Una riga per `review_id`. Prima della canonicalizzazione la pipeline verifica
che tutte le associazioni con lo stesso identificatore abbiano punteggio, testo
e date coerenti. Titolo e messaggio originali restano in colonne separate;
`review_text` li combina soltanto per il futuro retrieval. Una recensione e
eleggibile se possiede almeno un titolo o un messaggio non vuoto, quindi anche
il solo titolo e sufficiente.

`review_language` e un metadata euristico (`pt`, `es`, `en`, `it` oppure
`unknown_or_ambiguous`), non un filtro. Testi brevi o senza segnali linguistici
sufficienti restano ambigui.

### `review_order_links`

Una riga per la coppia unica `(review_id, order_id)`. E il bridge molti-a-molti
tra recensioni canoniche e ordini; non replica il testo delle recensioni.

## Dataset curated

### `orders_enriched`

Una riga per `order_id`. Viene costruito con `LEFT JOIN` soltanto verso tabelle
gia ridotte a una riga per chiave. Il conteggio deve coincidere con `orders`.
Le principali feature derivate sono:

- `delivery_time_days`: giorni tra acquisto e consegna;
- `delivery_delay_days`: giorni tra data stimata e consegna, positivo se tardi;
- `is_late`: consegna successiva alla data stimata;
- `order_value`: prezzo degli item piu trasporto;
- `purchase_date`, `purchase_month`, `purchase_year`: derivati dall'acquisto.

### `reviews_enriched`

Una riga per `review_id`, con testo originale, punteggio, lingua stimata e
metadata di ordine non ambigui. Una categoria viene esposta soltanto se tutti
gli ordini collegati sono mono-categoria stretti e condividono la stessa
categoria. Stato cliente, stato ordine, anno e mese vengono esposti soltanto se
sono noti e uguali per tutti gli ordini collegati.

### `review_order_links`

Il bridge processed viene pubblicato anche nel curated mantenendo la chiave
composta unica. Consente analisi per ordine senza duplicare i documenti che in
seguito saranno indicizzati in Chroma.

## Invarianti bloccanti

- nessuna duplicazione delle chiavi dichiarate;
- nessuna foreign key orfana nelle relazioni principali;
- punteggio recensione compreso tra 1 e 5;
- nessun fan-out nei join di `orders_enriched`;
- stesso numero di righe tra `orders` e `orders_enriched`;
- stesso numero di righe tra `reviews_canonical` e `reviews_enriched`;
- unicita di CAP e delle associazioni recensione-ordine.

Una violazione interrompe la pipeline prima di pubblicare nuovi output.
