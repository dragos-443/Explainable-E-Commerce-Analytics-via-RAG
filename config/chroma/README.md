# Configurazione Chroma

In locale Chroma è raggiungibile dal container applicativo come `chroma:8000` e
dal computer host come `localhost:8000`. I dati sono conservati nel volume Docker
`chroma-data`.

La collezione definitiva e il modello di embedding verranno implementati nella
Fase 4. Lo smoke test della Fase 0 usa una collezione separata ed embedding
deterministici leggeri, così da verificare il collegamento senza scaricare modelli.
