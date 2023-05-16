import pandas
class Data:
    #initalizes class by setting the default variable and automatically loads all data
    def __init__(self):
        self.data = None
        self.file_name = "./Crude Oil WTI Futures Historical Data.csv"
        self.load_data_file(self.file_name)
    
    #configures all data that it is dates, integers and floats
    def load_data_file(self, file_path):
        
        #reads from file
        self.data = pandas.read_csv(file_path)
        
        #loads volume as integer
        self.data["Vol."] = self.data["Vol."].replace({'K': '*1e3', 'M': '*1e6', '-': '-1'}, regex=True)
        self.data["Vol."] = self.data["Vol."].map(pandas.eval)
        self.data["Vol."] = self.data["Vol."].astype(int)
        
        #loads change percent, high, low, price and open values as float 
        self.data["Change %"] = self.data["Change %"].replace("%", "", regex=True)
        self.data["Change %"] = self.data["Change %"].astype(float)
        self.data["High"] = self.data["High"].replace(",", "", regex=True)
        self.data["High"] = self.data["High"].astype(float)
        self.data["Low"] = self.data["Low"].replace(",", "", regex=True)
        self.data["Low"] = self.data["Low"].astype(float)
        self.data["Open"] = self.data["Open"].replace(",", "", regex=True)
        self.data["Open"] = self.data["Open"].astype(float)
        self.data["Price"] = self.data["Price"].replace(",", "", regex=True)
        self.data["Price"] = self.data["Price"].astype(float)
        
        #loads date as dateTime objects
        self.data["Date"] = pandas.to_datetime(self.data["Date"])
        self.data = self.data.iloc[::-1].reset_index(drop=True)

    #returns prices column
    def get_all_prices(self): return self.data["Price"]
    
    #returns low column
    def get_all_low(self): return self.data["Low"]
    #returns high column
    def get_all_high(self):return self.data["High"]
    #returns volume column
    def get_all_volume(self):return self.data["Vol."]
    #returns date column
    def get_all_date(self):return self.data["Date"]
    #returns change column
    def get_all_change_percent(self):return self.data["Change %"]
    #returns open column
    def get_all_open(self):return self.data["Open"]

    #returns certain record of price if exists
    def get_prices(self,index):
        if(index<len( self.data["Price"])):return self.data["Price"][index];return None;
    #returns certain record of open if exists
    def get_open(self, index):
        if(index<len( self.data["Open"])):return self.data["Open"][index];return None;
    #returns certain record of low if exists
    def get_low(self, index):
        if(index<len( self.data["Low"])):return self.data["Low"][index];return None;
    #returns certain record of high if exists
    def get_high(self, index):
        if (index < len(self.data["High"])): return self.data["High"][index];return None;
    #returns certain record of volume if exists
    def get_volume(self, index):
        if (index < len(self.data["Vol."])): return self.data["Vol."][index];return None;
    #returns certain record of date if exists
    def get_date(self, index):
        if (index < len(self.data["Date"])): return self.data["Date"][index];return None;
    #returns certain record of change percent if exists
    def get_change_percent(self, index):
        if (index < len(self.data["Change %"])): return self.data["Change %"][index];return None;
    
    #returns all values
    def get_all_values(self):return self.data
    #replaces values with new ones
    def set_all_values(self,values): self.data=values