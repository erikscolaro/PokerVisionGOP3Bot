# RICORDA: tesseract deve essere installato a livello di sitema e il path deve essere specificato in image_text.py

## Analisi computazionale: confronto tra i vari metodi di template matching in open cv

Siano date una immagine detta target con le dimensioni \( W_T=1920 \) e \( H_T=1080 \)  e \(M\) immagini con dimensioni \(W_M\) e \(H_M\). Si definisca anche il fattore di scalamento del _pyramid scaling_ di cv2 come \(S=1/2\). Allora si hanno le seguenti complessità computazionali:

- Per il _template matching_ standard $$ \sum_{i=0}^{M} O(W_{Mi} \cdot H_{Mi} \cdot W_T \cdot H_T) $$

- Per il _template matching_ con _pyramid scaling_ , assumendo di aver già prescalato l'immagine target e le immagini template di un fattore di scalamento S è $$ \sum_{i=0}^{M} O( S^4 \cdot W_{Mi} \cdot H_{Mi} \cdot W_T \cdot H_T) $$ 

- Per il calcolo della convoluzione attraverso trasformata di fourier bidimensionale si seguono i seguenti passsaggi:
    - FFT2D dell'immagine target $$O(W_T \cdot H_T \cdot \log_2(W_T \cdot H_T))$$
    - FFT2D delle immagini template $$ \sum_{i=0}^{M} O(W_{Mi} \cdot H_{Mi} \cdot \log_2(W_{Mi} \cdot H_{Mi})) $$
    - Padding delle immagini template
    - Moltiplicazione delle FFT2D dell'immagine target con le FFT2d delle immagini template $$ \sum_{i=0}^{M} O(W_T \cdot H_T) = O(M \cdot W_T \cdot H_T)$$
    - IFFT2D delle moltiplicazioni precedenti $$\sum_{i=0}^{M} O(W_T \cdot H_T \cdot \log_2(W_T \cdot H_T)) = O(M \cdot W_T \cdot H_T \cdot \log_2(W_T \cdot H_T))$$
    - Se assumiamo di aver precalcolato le fft2d delle immagini di template e di averne già fatto il padding, allora la complessità esatta è pari a quella di moltiplicazione e inversa della trasformata. $$ O(M \cdot W_T \cdot H_T) + O(M \cdot W_T \cdot H_T \cdot \log_2(W_T \cdot H_T)) $$ 
    Dal momento che stiamo ragionando in o-grande e assumendo \( W_T \cdot H_T >> 2 \) il primo termine risulta trascurabile, quindi la complessità o-grande finale è $$ O(M \cdot W_T \cdot H_T \cdot \log_2(W_T \cdot H_T)) $$

Effettuiamo ora il confronto tra le varie tecniche. Il _template matching_ con fattore di scalamento risulta sempre più efficiente della tecnica senza scalamento. Riscrivendo la complessità del _template matching_ con scalamento:
$$ O( S^4 \cdot W_T \cdot H_T \cdot \sum_{i=0}^{M}(W_{Mi} \cdot H_{Mi})) $$ 
La trasformata di fourier risulta conveniente rispetto al _template matching_ con scalamento solo se:
$$ S^4 \cdot \sum_{i=0}^{M}(W_{Mi} \cdot H_{Mi}) > M \cdot \log_2(W_T \cdot H_T) $$
Dividendo per M, il termine risultante \(\sum_{i=0}^{M}(W_{Mi}\cdot H_{Mi})/M\) corrisponde all'area media delle immagini template. Assumiamo come peggiore il caso in cui le immagini template abbiano una superficie media di 16x16 pixel e con dimensioni dell'immagine target 1920x1080:

$$ S^4 \cdot (\sum_{i=0}^{M}(W_{Mi} \cdot H_{Mi}))/M > \log_2(W_T \cdot H_T) $$

$$ S^4 \cdot avg(Area(immagini_{template})) > S^4 \cdot 16 \cdot 16 > log_2 (1920\cdot1080)\\
 \iff S > \sqrt[4]{log_2(1920\cdot1080) / (16\cdot16)} \approx 0.4 $$
Se 1/S deve essere multiplo di due, Allora \(S>=1/(2^1)= 0.5 > 0.4\) l'unico fattore di scalamento accettabile nel caso peggiore è uno scalamento di fattore 2.

### Conlcusione 

Dal momento che useremo immagini di template maggiori di 16x16 in ogni caso e che un fattore di scalamento di 2 rappresenta una perdita di informazione trascurabile in questo caso, anzi potrebbe addirittura portare dei benefici per quanto riguarda la riduzione del rumore dell'immagine, utilizzerò nel progetto una tecnica di _template matching_ con fattore di scalamento pari a 2.