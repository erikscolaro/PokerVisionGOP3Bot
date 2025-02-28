from transitions import Machine

class TexasHoldem:
    def __init__(self, image_reader):
        self.image_reader = image_reader  # Oggetto che legge l'immagine per ottenere le carte e scommesse
        self.players_info = {}  # Informazioni sui giocatori
        self.cards = []  # Carte comuni (flop, turn, river)
        self.state = 'INIZIO'  # Stato iniziale
        self.machine = self.create_machine()

    def create_machine(self):
        states = ['INIZIO', 'PRE_FLOP', 'FLOP', 'TURN', 'RIVER', 'SHOWDOWN', 'FINE']
        machine = Machine(model=self, states=states, initial='INIZIO')

        # Transizioni
        machine.add_transition('distribuisci_carte', 'INIZIO', 'PRE_FLOP')
        machine.add_transition('fase_scommesse', 'PRE_FLOP', 'FLOP')
        machine.add_transition('distribuisci_flop', 'FLOP', 'TURN')
        machine.add_transition('distribuisci_turn', 'TURN', 'RIVER')
        machine.add_transition('distribuisci_river', 'RIVER', 'SHOWDOWN')
        machine.add_transition('showdown', 'SHOWDOWN', 'FINE')

        return machine

    def log_cards_and_bets(self, image):
        # Usa l'oggetto PokerImageReader per ottenere carte e scommesse
        cards, bets = self.image_reader.read_image(image)
        self.cards.extend(cards)
        for player, bet in bets.items():
            if player not in self.players_info:
                self.players_info[player] = {}
            self.players_info[player]['bet'] = bet
        print(f"Carte: {cards}, Puntate: {bets}")

    def distribuisci_carte(self, image):
        self.log_cards_and_bets(image)
        self.state = 'PRE_FLOP'

    def fase_scommesse(self, image):
        self.log_cards_and_bets(image)
        self.state = 'FLOP'

    def distribuisci_flop(self, image):
        self.log_cards_and_bets(image)
        self.state = 'TURN'

    def distribuisci_turn(self, image):
        self.log_cards_and_bets(image)
        self.state = 'RIVER'

    def distribuisci_river(self, image):
        self.log_cards_and_bets(image)
        self.state = 'SHOWDOWN'

    def showdown(self, image):
        self.log_cards_and_bets(image)
        self.state = 'FINE'
