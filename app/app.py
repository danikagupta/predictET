import streamlit as st
import pandas as pd
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

from kats.consts import TimeSeriesData
from kats.models.ensemble.ensemble import EnsembleParams, BaseModelParams
from kats.models.ensemble.kats_ensemble import KatsEnsemble
from kats.models import (
    #arima,
    #holtwinters,
    #linear_model,
    prophet,  # requires fbprophet be installed
    #quadratic_model,
    #sarima,
    theta,
)
from kats.utils.backtesters import BackTesterSimple
from kats.models.arima import ARIMAModel, ARIMAParams

####functions
def mapper(lat, lon, county):
  fig, ax = plt.subplots(figsize=(12,8))
  map = Basemap(width=240000,height=160000, resolution='c', projection='lcc', lat_0 =  lat, lon_0 = lon)
  map.bluemarble(scale=0.5,alpha=0.3)
  #map.etopo(scale=0.5, alpha=0.3)
  map.drawcounties(linewidth=0.5)

  x, y = map(lon, lat)
  plt.plot(x, y, 'ok', markersize=3)
  plt.text(x, y, county, fontsize=18);
  st.pyplot(fig)

def forecast_plot(fcst, etts, ettr, plot_cap=True, xlabel='Years', ylabel='ET Index',figsize=(10, 6)):
  fig = plt.figure(facecolor='w', figsize=figsize)
  ax = fig.add_subplot(111)

  fcst_t = fcst['time'].dt.to_pydatetime()
  ax.plot(etts.to_dataframe().DateTime, etts.to_dataframe().ET, 'r.')
  ax.plot(ettr.to_dataframe().DateTime, ettr.to_dataframe().ET, 'go')
  ax.plot(fcst_t, fcst['fcst'], ls='-', c='#0072B2')
  ax.fill_between(fcst_t, fcst['fcst_lower'], fcst['fcst_upper'],color='#0072B2', alpha=0.2)
  ax.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)
  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)
  fig.tight_layout()
  return fig

def add_background_img(img="https://raw.githubusercontent.com/danikagupta/et_data1/main/img2.png"):
  if img!="https://raw.githubusercontent.com/danikagupta/et_data1/main/img0.png":
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("{img}");
             background-attachment: fixed;
             background-size: cover;
             opacity: 1;
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

def add_background_color():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-color: lightblue;
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

def aggregateEnsemble(fcst1,fcst2,fcst3):
  fcst_all_drop=pd.concat([fcst1.reset_index(drop=True),fcst2.reset_index(drop=True),fcst3.reset_index(drop=True)],axis=1,copy=False)
  fcst_df = pd.DataFrame(
                {
                    "time": fcst1["time"],
                    "fcst": fcst_all_drop["fcst"].median(axis=1),
                    "fcst_lower": fcst_all_drop["fcst_lower"].median(axis=1),
                    "fcst_upper": fcst_all_drop["fcst_upper"].median(axis=1),
                }
            )
  return fcst_df

def displayQualityParams(backtestErrors):
  df=pd.DataFrame.from_dict(backtestErrors,orient='index',columns=['Value']).transpose()
  st.dataframe(df.style.set_precision(2))

locationData=pd.read_csv("https://raw.githubusercontent.com/danikagupta/et_data1/main/LocationData.csv")
cities=locationData["City"].values.tolist()
citiesNoHyphen=[x.replace("_"," ") for x in cities]
backtestMethods=["mape", "smape", "mae", "mase", "mse", "rmse"]
header_format = '<p style="font-family:Roboto; color:White; font-size: 32px;"> {} with {} </p>'
backgroundImages={
       "None":"https://raw.githubusercontent.com/danikagupta/et_data1/main/img0.png",
       "Green leaves":"https://raw.githubusercontent.com/danikagupta/et_data1/main/img1.png",
       "Leaves on water":"https://raw.githubusercontent.com/danikagupta/et_data1/main/img2.png",
       "Leaves in earth":"https://raw.githubusercontent.com/danikagupta/et_data1/main/img3.png",
       "More earth, less leaves":"https://raw.githubusercontent.com/danikagupta/et_data1/main/img4.png",
       "more leaves on earth":"https://raw.githubusercontent.com/danikagupta/et_data1/main/img5.png",
      }


with st.sidebar:
  displayMode=st.select_slider('Display Mode',options=['Two Column','Tabbed','Full Page'])
  backgroundChosen=st.selectbox("Background",list(backgroundImages.keys()))
  imageSelected=backgroundImages[backgroundChosen]
  
  cityChosen=st.selectbox("City",citiesNoHyphen)
  cityChosenHyphen=cityChosen.replace(" ","_")
  (lat,lon)=locationData.loc[locationData["City"] == cityChosenHyphen, ("Latitude", "Longitude")].iloc[0]
  mapper(lat,lon,cityChosen)

  #Show the raw data in the side-bar
  download_link=f"https://raw.githubusercontent.com/danikagupta/et_data1/main/{cityChosenHyphen}.csv"
  etdata=pd.read_csv(download_link)
  etdata.drop(etdata.filter(regex='Unnamed'), axis = 1, inplace = True)
  etdata.rename(columns = {'Ensemble ET':'ET'}, inplace = True)
  et_ts = TimeSeriesData(time=etdata.DateTime, value=etdata.ET) 
  monthsAvailable=len(et_ts)
  monthsKeep = st.slider('Months to use for forecast', 10, monthsAvailable, int((10+monthsAvailable)/2))
  monthsMore = st.slider('Additional months to forecast', 12, 60, 36)
  monthsForecast=monthsAvailable-monthsKeep+monthsMore
  trp=int(100*monthsKeep/monthsAvailable)
  tep=100-trp

  et_ts_short = et_ts[0:monthsKeep]
  et_ts_rest = et_ts[monthsKeep:]

  #show data as a graph
  df = et_ts.to_dataframe()
  fig = plt.figure(figsize = (12,8))#you can change the figure size from here
  plt.plot(df['DateTime'], df['ET'])
  plt.xlabel("Year")#change xaxis label
  plt.ylabel("ET")#change yaxis label
  plt.title("Time Series data of {}".format(cityChosen))#title
  plt.grid(True)#emove this to remove the grids in the figure

  #show raw data
  with st.expander("Source Data"):
    st.pyplot(fig)
    st.dataframe(etdata)

#setting display mode
add_background_img(img=imageSelected)

if displayMode=='Two Column':
  print("Display mode is two column")
  col1, col2 = st.columns(2)
  model1, model2, model3, model4 = col1.container(),col2.container(),col1.container(), col2.container()
if displayMode=='Tabbed':
  print("Display mode is tabbed")
  tab1, tab2, tab3,tab4 = st.tabs(["Prophet Model","SARIMA Model","Theta Model","ENSEMBLE"])
  model1, model2, model3, model4 = tab1.container(),tab2.container(),tab3.container(), tab4.container()
if displayMode=='Full Page':
  print("Display mode is full page")
  model1, model2, model3, model4 = st.container(),st.container(),st.container(), st.container()

with model1:
  # Prophet model
  from kats.models.prophet import ProphetModel, ProphetParams
  params = ProphetParams(seasonality_mode='multiplicative') # additive mode gives worse results
  m1 = ProphetModel(et_ts_short, params)
  m1.fit()
  # make prediction for additional month
  fcst1 = m1.predict(steps=monthsForecast, include_history=True)
  # Plot
  fig1=forecast_plot(fcst1,et_ts_short,et_ts_rest)
  st.markdown(header_format.format(cityChosen,'Prophet'), unsafe_allow_html=True)
  st.pyplot(fig1)
  backtester=BackTesterSimple(error_methods=backtestMethods,data=et_ts,params=params,train_percentage=trp,test_percentage=tep,model_class=ProphetModel)
  backtester.run_backtest()
  displayQualityParams(backtester.errors)

with model2:
  #SARIMA Model
  from kats.models.sarima import SARIMAModel, SARIMAParams
  warnings.simplefilter(action='ignore')
  params = SARIMAParams( p = 2, d=1, q=1, trend = 'ct', seasonal_order=(1,0,1,12))
  m2 = SARIMAModel(data=et_ts_short, params=params)
  m2.fit()
  fcst2 = m2.predict(steps=monthsForecast, include_history=True)
  fig2=forecast_plot(fcst2,et_ts_short,et_ts_rest)
  st.markdown(header_format.format(cityChosen,'SARIMA'), unsafe_allow_html=True)
  st.pyplot(fig2) 
  backtester=BackTesterSimple(error_methods=backtestMethods,data=et_ts,params=params,train_percentage=trp,test_percentage=tep,model_class=SARIMAModel)
  backtester.run_backtest()
  displayQualityParams(backtester.errors)

with model3:
  #Theta model
  from kats.models.theta import ThetaModel, ThetaParams
  params = ThetaParams(m=12)
  m3 = ThetaModel(data=et_ts_short, params=params)
  m3.fit()
  fcst3 = m3.predict(steps=monthsForecast, include_history=True, alpha=0.2)
  fig3=forecast_plot(fcst3,et_ts_short,et_ts_rest)
  st.markdown(header_format.format(cityChosen,'Theta'), unsafe_allow_html=True)
  st.pyplot(fig3) 
  backtester=BackTesterSimple(error_methods=backtestMethods,data=et_ts,params=params,train_percentage=trp,test_percentage=tep,model_class=ThetaModel)
  backtester.run_backtest()
  displayQualityParams(backtester.errors)

with model4:
  #ENSEMBLE reusing the work done previously
  fcst4=aggregateEnsemble(fcst1,fcst2,fcst3)
  fig4=forecast_plot(fcst4,et_ts_short,et_ts_rest)
  st.markdown(header_format.format(cityChosen,'ENSEMBLE'), unsafe_allow_html=True)
  st.pyplot(fig4) 
