import matplotlib.pyplot
import plotly.graph_objs
import plotly.offline
#enables the ability to plot graphs in the webpage
plotly.offline.init_notebook_mode(connected=True)
class Charts:
    #initializes the layout of the chart
    def __init__(self):
        self.view_layout = plotly.graph_objs.Layout(title="Oil Price",
            xaxis=dict(title="Date",
                       titlefont=dict(family="FreeSans, monospace",size=12,color="#FF0000")),
            yaxis=dict(title="Price",
                       titlefont=dict(family="FreeSans, monospace",size=12,color="#000000"))
        )
    #draws the first chart using plotly
    def draw_chart1(self,data_object):
        oil_prices = [{'x': data_object.get_all_date(), 'y': data_object.get_all_prices()}]
        graph = plotly.graph_objs.Figure(data=plotly.graph_objects.Bar(y=oil_prices[0]['y']), layout=self.view_layout)
        plotly.offline.iplot(graph)

    #draws the second chart using plotly
    def draw_chart2(self,trace0,trace1):
        oil_prices = [trace0, trace1]
        self.view_layout.xaxis.title.text = "per day"
        graph = plotly.graph_objs.Figure(data=oil_prices, layout=self.view_layout)
        plotly.offline.iplot(graph)

    #draws the third chart using pandas
    def draw_chart3(self,data_object):
        data_object.get_all_values()[[ "Low", "High","Price","Open"]].plot( kind="kde");

    #draws the fourth chart using matplotlib
    def draw_chart4(self,data_object):
        matplotlib.pyplot.stackplot(data_object.get_all_date(),data_object.get_all_high(), data_object.get_all_prices(), data_object.get_all_open(),
                       labels=['High','Price', 'Open' ])
        matplotlib.pyplot.legend(loc='upper left');
    #draws the fifth chart using pandas
    def draw_chart5(self, data_object):
        data_object.get_all_values().plot(x="Vol.", y="Change %", kind="hexbin",ylim=(-1, 1));
