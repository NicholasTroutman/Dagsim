class block:
    def __init__(self, txs, agents, creation_time, blockCounter, numAgents): #list of txs and agents
        self.blockTransactions = txs
        self.creators = agents
        self.creation_time = creation_time
        self.id = blockCounter
        self.blockLinks  = []
        self.seen = [""]*numAgents
        
    def __str__(self):
        return str(self.id)

    def __repr__(self):
        return str(self.id)
   