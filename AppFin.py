import streamlit as st
import yfinance as yf
from datetime import date
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from plotly import graph_objs as go

Data_Ini = '2017-01-01'
Data_Fim = date.today().strftime('%Y-%m-%d')

st.title('Análise de ações')

#Criando a sidebar
st.sidebar.header('Escolha a ação')

n_dias = st.slider('Qtde de dias para previsão', 30, 365)

def pegar_acoes():
    path = 'D:\\__Coding\\Projetos\\Python\\web\\Stock Predictor w Prophet\\acoes.csv'
    return pd.read_csv(path, delimiter=';')

df = pegar_acoes()

acao = df['snome']
nome_acao_escolhida = st.sidebar.selectbox('Escolha uma ação:', acao)

df_acao = df[df['snome'] == nome_acao_escolhida]
acao_escolhida = df_acao.iloc[0]['sigla_acao']
acao_escolhida = acao_escolhida + '.SA'

@st.cache(allow_output_mutation=True) #coloca em cache os valores pegos -> não precisa pegar toda vez do yfinance
def pegar_valores_online(sigla_acao):
    df = yf.download(sigla_acao, Data_Ini, Data_Fim)
    df.reset_index(inplace = True)
    return df

df_valores = pegar_valores_online(acao_escolhida)

#def limpa_coluna_Dates(data_hora):
df_valores['Date'] = pd.to_datetime(df_valores['Date']).dt.strftime('%Y-%m-%d')


st.subheader('Tabela de valores - ' + nome_acao_escolhida)
st.write(df_valores.tail(10))


#Criar gráfico

st.subheader('Gráfico de preços')
fig = go.Figure()
fig.add_trace(go.Scatter(x = df_valores['Date'],
                         y = df_valores['Close'],
                         name='Preço Fechamento',
                         line_color = 'red'))
fig.add_trace(go.Scatter(x = df_valores['Date'],
                         y = df_valores['Open'],
                         name='Preço Abertura',
                         line_color = 'blue'))

st.plotly_chart(fig)

#Previsões

df_treino = df_valores [['Date', 'Close']]

#Renomear colunas para uso no prophet (Date e Close)
df_treino = df_treino.rename(columns = {"Date": 'ds', 'Close': 'y'})

modelo = Prophet()
modelo.fit(df_treino)

futuro = modelo.make_future_dataframe(periods=n_dias, freq='B') #freq='B' -> utiliza apenas business days no modelo
previsao = modelo.predict(futuro)

st.subheader('Previsão')
st.write(previsao[['ds', 'yhat', 'yhat_lower', 'yhat_upper' ]].tail(n_dias))


#Grafico1
graf1 = plot_plotly(modelo, previsao)
st.plotly_chart(graf1)

#Grafico2
graf2 = plot_components_plotly(modelo, previsao)
st.plotly_chart(graf2)