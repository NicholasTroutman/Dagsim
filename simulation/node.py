#Node is a base class that mobile agents and immobile base stations inherit common functions from
class Node:
    def __init__(self, _counter):
        self.id = _counter
        #self._visible_transactions = [] ##visible blocks instead in this block setup
        #self.coordinates= [] #list of double x and y coordinates in double [x,y] #agent specifi
        #self.past_coordinates=[]#agent specific
        #self.destination=[]#agent specific
        #self.vector=[]#agent specific
        #self.prev_dest=[]#agent specific
        
        #self.speed=15 #agent specific


        self.coordinates= [] #list of double x and y coordinates in double [x,y]
        self.radius=60 #hardcoded radius of p2p connectivity

        #block variables
        self._visible_transactions=[]
        self.confirmed_transactions=[]
        
        #transaction variables
        self._visible_blocks = []
        self.confirmed_blocks = []
        
        #For analysis
        self.agent_average_confirmation_confidence = 0
        self.tips = []
        self.record_tips = []
        

    ##Transaction functions
    
    def get_visible_transactions(self): #return vis txs
        return self._visible_transactions
        
    def add_visible_transactions(self, new_txs, time):  #no return
        #print("\nadd_vis_trans begin: ", time)
        #print("new: ",new_txs)
        #print("old: ",self._visible_transactions)
        
        newest_txs = list(set(new_txs) - set(self._visible_transactions))
        #print("newest! :",newest_txs)
        for tx in newest_txs:
            #print(tx," ",tx.seen[self.id])
            if tx.seen[self.id] == "":
                #print("\nUNSEEN: ", tx,"\n")
                tx.seen[self.id] = time
                self._visible_transactions.append(tx) 
                #print("appended to vis_txs: ",self._visible_transactions)


    ##Block functions
    def add_visible_blocks(self, new_blocks, time): #no return
    #print("\nadd_vis_trans begin: ", time)
    #print("new: ",new_blocks)
    #print("old: ",self._visible_transactions)
    
        newest_blocks = list(set(new_blocks) - set(self._visible_blocks))
        #print("newest! :",newest_txs)
        for block in newest_blocks:
            #print(block," ",block.seen[self.id])
            if block.seen[self.id] == "":
                #print("\nUNSEEN: ", block,"\n")
                block.seen[self.id] = time
                self._visible_blocks.append(block) 
                #print("appended to vis_txs: ",self._visible_blocks)
    
    def get_visible_blocks(self): #return vis blocks
        return self._visible_blocks
        
        

    def __str__(self):
        return str(self.id)

    def __repr__(self):
        return str(self.id)
