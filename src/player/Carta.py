class Carta:
    def __init__(self, seme, tipo):
        self.seme = seme
        self.tipo = tipo

    def __str__(self):
        return f"{self.tipo} di {self.seme}"

    def __repr__(self):
        return f"Carta(seme='{self.seme}', tipo='{self.tipo}')"