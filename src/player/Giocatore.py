
class Giocatore:
    def __init__(self, nome, saldo_iniziale):
        self.nome = nome
        self.saldo = saldo_iniziale
        self.carte = []  # Le due carte personali del giocatore
        self.is_dealer = False
        self.is_piccolo_buio = False
        self.is_grande_buio = False
        self.in_gioco = False  # Per tracciare se il giocatore è ancora attivo nel round
        
    def ricevi_carta(self, carta):
        """Aggiunge una carta alla mano del giocatore"""
        if len(self.carte) < 2:
            self.carte.append(carta)
            
    def set_dealer(self, is_dealer):
        self.is_dealer = is_dealer
        
    def is_dealer(self):
        return self.is_dealer
    
    def set_piccolo_buio(self, is_piccolo_buio):
        self.is_piccolo_buio = is_piccolo_buio

    def is_piccolo_buio(self):
        return self.is_piccolo_buio
    
    def set_grande_buio(self, is_grande_buio):
        self.is_grande_buio = is_grande_buio
        
    def is_grande_buio(self):
        return self.is_grande_buio
        
    def reset_mano(self):
        """Resetta la mano del giocatore per una nuova partita"""
        self.carte = []
        self.in_gioco = True
        
    def modifica_saldo(self, importo):
        """Modifica il saldo del giocatore (positivo per aggiunta, negativo per sottrazione)"""
        self.saldo += importo
        class RuoloGiocatore(Enum):
            NORMALE = 0
            DEALER = 1
            PICCOLO_BUIO = 2
            GRANDE_BUIO = 3
