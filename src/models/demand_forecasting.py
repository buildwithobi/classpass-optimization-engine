import pandas as pd
from prophet import Prophet

class StudioDemandForecast:
    def __init__(self):
        self.model = None()

    def prepare_data(self, df):

        df["date"] = pd.to_datetime(df["date"])

        booked_df = df[df["booked"] == 1]

        daily_demand =(
            booked_df.groupby(booked_df["date"].dt.date)
            .size()
            .eset_index(name = "y")
        )

        daily_demand.columns = ["ds", "y"]

        daily_demand["ds"] = pd.to_datetime(daily_demand["ds"])

        return daily_demand
    
    def train(self, df):
        df_prepared = self.prepare_data(df)
        
        self.model = Prophet(
            daily_seasonality = True,
            weekly_seasonality = True,
            yearly_seasonality = False
        )
        
        self.model.fit(df_prepared)

    def forecast(self, periods = 30):
        if self.model is None:
            raise ValueError ("Model has not been trained yet.")
        
        future = self.model.make_future_dataframe(periods = periods)
        forecast = self.model.predict(future)
        
        return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]