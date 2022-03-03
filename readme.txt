##################################################################################################################################
######################################################LEGGIMI#####################################################################

IMPORT --> transformers, farm

##################################################################################################################################

Gli scripts 'FARM_model_training.py' e 'FARM_use_example,py' richiedono un impiego di ram eccessiva, dunque non giungono a compimento.
Ciò è dovuto al fatto che questi scripts addestrano localmente il modello, tale operazione è la causa dell'impiego eleveato della ram.

Contrariamente lo script 'TRANSFORMERS_use_example.py' utilizza un modello detto 'pretrained', appunto un modello già allenato e
pronto per la fase di predizione.

##################################################################################################################################

OBIETTIVI:

a) Analizzare se il tempo impiegato per una risposta (circa 0.22 secondi, con 16 gb di ram e 2.5 GHz) è riducibile
b) Analizzare i limiti di dimensioni del "contesto" e relative dipendenze
c) Analizzare la capacità di rintracciare sinonimi da parte del modello, individuare eventuali limiti

##################################################################################################################################

OSSERVAZIONI:

'haystack' è uno script che permette l'estensione' del QA mostrato in 'transformers' ad un insieme di documenti (anche ampio).
Applicato nella ricerca semantica di documenti, esso è adatto anche alla generazione della risposta ?
Il nostro sistema di indicizzazione con reti neurali può essere una valida alternativa al sistema 'haystack' ?

##################################################################################################################################
##################################################################################################################################