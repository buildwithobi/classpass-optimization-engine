import numpy as np

class DynamicPricingEngine:
    def __init__(self, base_price = 10):
        self.base_price = base_price
    
    def adjust_price(self, demand_score, fill_rate):
        """

        demand_score: predicted demand  (0-1)
        fill_rate: % seats filled
        
        """

        if fill_rate > 0.9:
            multiplier = 1.3
        elif fill_rate < 0.5:
            multiplier = 0.8
        else:
            multiplier = 1 + demand_score * 0.2

        return round(self.base_price * multiplier, 2)