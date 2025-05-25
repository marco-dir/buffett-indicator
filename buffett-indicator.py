import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pandas_datareader.data as web

# Configurazione della pagina
st.set_page_config(
    page_title="Buffett Indicator Dashboard",
    page_icon="📈",
    layout="wide"
)

# Titolo e descrizione
st.title("Buffett Indicator - Analisi in Tempo Reale")
st.markdown("""
Il **Buffett Indicator** è un rapporto che confronta la capitalizzazione totale 
del mercato azionario con il PIL di un paese. È considerato uno strumento utile 
per valutare se il mercato azionario è sopravvalutato rispetto all'economia reale.
""")

# Sidebar con informazioni
st.sidebar.header("Informazioni")
st.sidebar.info("""
**Interpretazione del Buffett Indicator:**

- **> 120%**: Significativamente Sopravvalutato
- **100-120%**: Moderatamente Sopravvalutato
- **80-100%**: Correttamente Valutato
- **< 80%**: Sottovalutato

Warren Buffett ha descritto questo indicatore come "probabilmente il miglior indicatore 
singolo di dove si trovano le valutazioni in un dato momento."
""")

# Aggiungiamo informazione sull'aggiornamento
st.sidebar.info("⏱️ Aggiornamento automatico")

# Imposta il periodo massimo disponibile
period = "max"

# Usa solo Wilshire 5000
market_index = "^W5000"
index_name = "Wilshire 5000 Total Market"

# Funzione per ottenere dati del PIL statunitense
@st.cache_data(ttl=86400)  # cache per 24 ore
def get_gdp_data():
    try:
        # Quando utilizziamo period="max", dobbiamo ottenere dati PIL per un periodo più lungo
        gdp_start_date = datetime(1947, 1, 1)  # I dati PIL degli USA sono disponibili dal 1947
        
        # Tentiamo di ottenere i dati del PIL da FRED
        gdp = web.DataReader('GDP', 'fred', gdp_start_date, datetime.now())
        
        if gdp.empty:
            raise ValueError("Nessun dato GDP disponibile da FRED")
            
        return gdp
    except Exception as e:
        st.info("Utilizziamo dati del PIL simulati per la dimostrazione")
        
        # Generiamo dati simulati del PIL USA dal 1947 fino ad oggi
        start_date = datetime(1947, 1, 1)
        end_date = datetime.now()
        
        # Genera date trimestrali
        quarters = pd.date_range(
            start=start_date,
            end=end_date,
            freq='Q'
        )
        
        # Generiamo valori del PIL realistici
        # Nel 1947 il PIL USA era circa 243 miliardi
        # Oggi è circa 26,500 miliardi
        initial_gdp = 243
        final_gdp = 26500
        
        n_quarters = len(quarters)
        quarterly_growth = (final_gdp / initial_gdp) ** (1 / n_quarters)
        
        # Generiamo valori con crescita esponenziale e fluttuazioni
        gdp_values = []
        current_gdp = initial_gdp
        
        for _ in range(n_quarters):
            # Aggiungi fluttuazione trimestrale (±1%)
            quarterly_noise = np.random.normal(0, 0.01)
            current_gdp = current_gdp * (quarterly_growth + quarterly_noise)
            gdp_values.append(current_gdp)
        
        # Aggiungiamo recessioni periodiche
        # Circa ogni 8-10 anni
        recession_years = [1953, 1957, 1960, 1969, 1973, 1980, 1981, 1990, 2001, 2008, 2020]
        for year in recession_years:
            # Troviamo l'indice più vicino alla data della recessione
            recession_idx = np.abs([(date.year - year) for date in quarters]).argmin()
            if recession_idx < len(gdp_values) - 2:
                # Riduciamo il PIL per simulare la recessione (2-5 trimestri)
                recession_length = np.random.randint(2, 6)
                recession_severity = np.random.uniform(0.02, 0.1)  # 2-10% di calo
                
                for i in range(recession_length):
                    if recession_idx + i < len(gdp_values):
                        # Effetto decrescente della recessione
                        effect = recession_severity * (1 - i/recession_length)
                        gdp_values[recession_idx + i] *= (1 - effect)
        
        simulated_gdp = pd.DataFrame(gdp_values, index=quarters, columns=['GDP'])
        return simulated_gdp

# Funzione per ottenere dati di mercato da Yahoo Finance
@st.cache_data(ttl=3600)  # cache per 1 ora
def get_market_data(ticker, period_str):
    try:
        # Per il periodo massimo, specifichiamo una data di inizio
        if period_str == "max":
            start_date = "1971-01-01"  # Wilshire 5000 inizia dal 1971
            end_date = datetime.now().strftime("%Y-%m-%d")
            data = yf.download(ticker, start=start_date, end=end_date)
        else:
            data = yf.download(ticker, period=period_str)
        
        if data.empty or 'Close' not in data.columns:
            raise ValueError(f"Nessun dato disponibile per {ticker}")
            
        return data
    except Exception as e:
        st.warning("Utilizziamo dati simulati per la dimostrazione")
        
        # Per simulazione dati storici massimi
        end_date = datetime.now()
        start_date = datetime(1971, 1, 1)  # Wilshire 5000
        
        # Usa business days (giorni lavorativi, escludendo weekend)
        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        
        # Crea prezzi simulati per Wilshire 5000
        initial_value = 1400  # Valore iniziale nel 1971
        final_value = 46000
        
        # Crea un trend con crescita esponenziale e un po' di volatilità
        n_points = len(dates)
        daily_return = (final_value / initial_value) ** (1 / n_points) - 1
        
        prices = []
        current_price = initial_value
        
        for _ in range(n_points):
            # Aggiunge volatilità giornaliera
            daily_noise = np.random.normal(0, 0.01)  # 1% di volatilità standard
            current_price = current_price * (1 + daily_return + daily_noise)
            prices.append(current_price)
        
        # Crea un DataFrame con le colonne standard di yfinance
        simulated_data = pd.DataFrame({
            'Open': prices,
            'High': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
            'Low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
            'Close': prices,
            'Adj Close': prices,
            'Volume': [int(np.random.normal(5e9, 1e9)) for _ in range(len(prices))]
        }, index=dates)
        
        return simulated_data

# Funzione per stimare la capitalizzazione di mercato totale
def estimate_total_market_cap(index_data, index_symbol):
    """
    Stima la capitalizzazione di mercato totale basata sul valore dell'indice.
    
    Il Wilshire 5000 Total Market Index (^W5000) è progettato per tracciare il valore di 
    mercato di tutte le azioni USA attivamente negoziate, quindi è particolarmente adatto
    per stimare la capitalizzazione di mercato totale.
    """
    # Gestiamo la conversione manualmente
    market_cap_values = []
    
    # Valori attuali di riferimento per la calibrazione
    current_wilshire = 46000  # Valore approssimativo attuale
    current_total_market_cap = 80000  # in miliardi di dollari
    
    for i in range(len(index_data)):
        date = index_data.index[i]
        close_price = float(index_data['Close'].iloc[i])
        year = date.year
        
        # Per il Wilshire 5000, il valore dell'indice è approssimativamente
        # uguale alla capitalizzazione di mercato totale in miliardi
        # Il Wilshire 5000 è stato progettato in modo che 1 punto = 1 miliardo di dollari
        # Ma nel corso del tempo questo rapporto è cambiato
        if year < 1980:
            # Prima del 1980, utilizzare un fattore di correzione basato sul valore iniziale
            # Il Wilshire 5000 è stato creato con un valore base di 1404.60 al 31/12/1970,
            # che rappresentava una capitalizzazione di mercato di circa 1.4 trilioni
            factor = 1.0
        elif year < 1990:
            factor = 1.1  # Leggero adeguamento per gli anni '80
        elif year < 2000:
            factor = 1.2  # Anni '90
        elif year < 2010:
            factor = 1.5  # Anni 2000
        elif year < 2020:
            factor = 1.7  # Anni 2010
        else:
            # Il Wilshire 5000 Full Cap (WILL) rappresenta la capitalizzazione totale in miliardi
            factor = current_total_market_cap / current_wilshire  # ~1.75
            
        market_cap = close_price * factor
        market_cap_values.append(market_cap)
    
    # Creiamo una Serie pandas con i valori calcolati
    market_cap = pd.Series(market_cap_values, index=index_data.index)
    
    return market_cap

# Funzione per calcolare il Buffett Indicator
def calculate_buffett_indicator(market_cap, gdp_data):
    """
    Calcola il Buffett Indicator come rapporto tra capitalizzazione di mercato e PIL.
    Formula corretta: (Capitalizzazione di mercato / PIL) * 100
    """
    # Convertiamo il PIL trimestrale in dati giornalieri mediante interpolazione
    gdp_daily = gdp_data.resample('D').interpolate(method='linear')
    
    # Allineiamo le date tra capitalizzazione di mercato e PIL
    aligned_dates = market_cap.index.intersection(gdp_daily.index)
    if len(aligned_dates) == 0:
        st.error("Nessuna sovrapposizione tra le date dei dati di mercato e PIL")
        return pd.DataFrame()
    
    # Filtriamo i dati sulle date comuni
    aligned_market_cap = market_cap.loc[aligned_dates]
    aligned_gdp = gdp_daily.loc[aligned_dates]
    
    # Calcoliamo il Buffett Indicator: (Market Cap / GDP) * 100
    buffett_values = []
    
    # Iteriamo manualmente per evitare problemi di dimensionalità
    for i in range(len(aligned_dates)):
        cap = float(aligned_market_cap.iloc[i])  # Capitalizzazione in milioni/miliardi
        gdp_val = float(aligned_gdp['GDP'].iloc[i])  # PIL in miliardi
        
        # Formula: (Capitalizzazione / PIL) * 100
        indicator_val = (cap / gdp_val) * 100
        buffett_values.append(indicator_val)
    
    # Creiamo un DataFrame con i risultati
    result_df = pd.DataFrame({'Buffett Indicator': buffett_values}, index=aligned_dates)
    
    return result_df

try:
    # Otteniamo i dati di mercato
    with st.spinner('Scaricamento dati di mercato in corso...'):
        market_data = get_market_data(market_index, period)
        
        if market_data.empty:
            st.error("Impossibile ottenere i dati di mercato. Riprova più tardi.")
            st.stop()
    
    # Otteniamo i dati del PIL
    with st.spinner('Elaborazione dati PIL in corso...'):
        gdp_data = get_gdp_data()
        
        if gdp_data.empty:
            st.error("Impossibile ottenere i dati del PIL. Riprova più tardi.")
            st.stop()
    
    # Calcoliamo la capitalizzazione di mercato stimata
    with st.spinner('Calcolo della capitalizzazione di mercato...'):
        market_cap = estimate_total_market_cap(market_data, market_index)
    
    # Calcoliamo il Buffett Indicator
    with st.spinner('Calcolo del Buffett Indicator...'):
        buffett_indicator_data = calculate_buffett_indicator(market_cap, gdp_data)
        
        if buffett_indicator_data.empty:
            st.error("Impossibile calcolare il Buffett Indicator. Dati insufficienti.")
            st.stop()

    # Visualizziamo i risultati
    st.subheader(f"Andamento Storico del Buffett Indicator")
    
    # Verifichiamo che ci siano dati da visualizzare
    if buffett_indicator_data.empty or len(buffett_indicator_data) < 2:
        st.error("Dati insufficienti per generare il grafico")
        st.stop()
    
    # Creiamo il grafico con Plotly
    fig = go.Figure()
    
    # Aggiungiamo la linea del Buffett Indicator
    fig.add_trace(go.Scatter(
        x=buffett_indicator_data.index,
        y=buffett_indicator_data['Buffett Indicator'],
        mode='lines',
        name='Buffett Indicator',
        line=dict(color='blue', width=2)
    ))
    
    # Invece di linee orizzontali, creiamo curve esponenziali per i livelli di valutazione
    # Creiamo una serie di date uniformemente distribuite per le curve esponenziali
    exp_dates = pd.date_range(start=buffett_indicator_data.index[0], end=buffett_indicator_data.index[-1], periods=500)

    # Parametri per controllare la forma della curva esponenziale
    base_value = 60     # Valore base iniziale
    exp_scale = 50      # Fattore di scala per l'esponenziale
    exp_rate = 1.5      # Tasso di crescita esponenziale

    # Creiamo le curve esponenziali per ciascun livello
    time_points = np.linspace(0, 1, len(exp_dates))  # Normalizzato da 0 a 1

    # Funzione esponenziale: base_value + exp_scale * (e^(rate * t) - 1)
    exp_values_120 = base_value + exp_scale * (np.exp(exp_rate * time_points) - 1)  # Sopravvalutato (120%)
    exp_values_100 = base_value + exp_scale * (np.exp(exp_rate * 0.8 * time_points) - 1)  # Neutrale (100%)
    exp_values_80 = base_value + exp_scale * (np.exp(exp_rate * 0.6 * time_points) - 1)   # Sottovalutato (80%)

    # Aggiungiamo le curve esponenziali al grafico
    fig.add_trace(go.Scatter(
        x=exp_dates,
        y=exp_values_120,
        mode='lines',
        name='Sopravvalutato',
        line=dict(dash="dash", width=1, color="red"),
        showlegend=True
    ))

    fig.add_trace(go.Scatter(
        x=exp_dates,
        y=exp_values_100,
        mode='lines',
        name='Neutrale',
        line=dict(dash="dash", width=1, color="orange"),
        showlegend=True
    ))

    fig.add_trace(go.Scatter(
        x=exp_dates,
        y=exp_values_80,
        mode='lines',
        name='Sottovalutato',
        line=dict(dash="dash", width=1, color="green"),
        showlegend=True
    ))

    # Aggiungiamo annotazioni per i livelli di valutazione
    # Spostiamo le annotazioni alla fine delle curve esponenziali
    fig.add_annotation(
        x=exp_dates[-1], 
        y=exp_values_120[-1], 
        text="Sopravvalutato", 
        showarrow=False, 
        xshift=100, 
        font=dict(color="red")
    )
    fig.add_annotation(
        x=exp_dates[-1], 
        y=exp_values_100[-1], 
        text="Neutrale", 
        showarrow=False, 
        xshift=100, 
        font=dict(color="orange")
    )
    fig.add_annotation(
        x=exp_dates[-1], 
        y=exp_values_80[-1], 
        text="Sottovalutato", 
        showarrow=False, 
        xshift=100, 
        font=dict(color="green")
    )
    
    # Configuriamo il layout del grafico
    fig.update_layout(
        xaxis_title="Data",
        yaxis_title="Buffett Indicator (%)",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        height=600,
        margin=dict(l=20, r=20, t=30, b=20),
    )
    
    # Mostriamo il grafico Plotly
    st.plotly_chart(fig, use_container_width=True)
    
    # Mostriamo il valore attuale del Buffett Indicator
    if not buffett_indicator_data.empty:
        current_buffett = float(buffett_indicator_data['Buffett Indicator'].iloc[-1])
        current_market_cap = float(market_cap.iloc[-1])
        current_gdp = float(gdp_data['GDP'].iloc[-1])  # Già in miliardi
        
        # Creiamo tre colonne per mostrare le metriche
        col1, col2, col3 = st.columns(3)
        
        with col1:
            delta = current_buffett - 100
            delta_color = "inverse" if delta > 0 else "normal"  # Rosso se sopra 100, verde se sotto
            st.metric(
                label="Buffett Indicator", 
                value=f"{current_buffett:.2f}%",
                delta=f"{delta:.2f}%", 
                delta_color=delta_color
            )
        
        with col2:
            st.metric(
                label="Capitalizzazione di Mercato", 
                value=f"${current_market_cap/1000:.2f} T"  # In trilioni
            )
        
        with col3:
            st.metric(
                label="PIL USA", 
                value=f"${current_gdp/1000:.2f} T"  # Già in trilioni
            )
    
    # Determiniamo lo stato attuale del mercato basato sulla curva esponenziale più vicina
    # Usiamo i valori finali delle curve esponenziali per la data attuale
    current_time_point = 1.0  # L'ultima data normalizzata è 1.0
    current_exp_120 = exp_values_120[-1]
    current_exp_100 = exp_values_100[-1]
    current_exp_80 = exp_values_80[-1]

    # Determiniamo lo stato basato sulla distanza dalle curve esponenziali
    if current_buffett > current_exp_120:
        status = "🔴 Significativamente Sopravvalutato"
    elif current_buffett > current_exp_100:
        status = "🟠 Moderatamente Sopravvalutato"
    elif current_buffett >= current_exp_80:
        status = "🟢 Correttamente Valutato"
    else:
        status = "🔵 Sottovalutato"
    
    # Mostriamo lo stato attuale
    st.success(f"Stato attuale del mercato: {status}")
    
    # Mostriamo l'ultimo aggiornamento
    st.sidebar.text(f"Ultimo aggiornamento: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    st.sidebar.text(f"Prossimo aggiornamento: domani")
    
    # Aggiungiamo una nota sulla metodologia, sui dati storici e sulle curve esponenziali
    st.info("""
    ℹ️ **Nota sulla metodologia:** Il Buffett Indicator viene calcolato come rapporto tra la capitalizzazione 
    di mercato totale e il PIL degli Stati Uniti. La capitalizzazione di mercato è stimata partendo 
    dal valore dell'indice Wilshire 5000, che è concepito per rappresentare l'intero mercato azionario USA.
          
    **Nota sulle curve esponenziali:**
    Le linee di riferimento per i livelli di valutazione (sopravvalutato, neutrale, sottovalutato) 
    seguono un andamento esponenziale. Questo riflette meglio l'andamento storico 
    del mercato, che tende a mostrare una crescita esponenziale nel lungo periodo, con 
    accelerazioni più marcate in fase di euforia di mercato.
    """)

except Exception as e:
    st.error(f"Si è verificato un errore: {str(e)}")
    st.info("""
    Potrebbero esserci problemi temporanei con le API di Yahoo Finance o FRED. 
    Prova a ricaricare la pagina o a tornare più tardi.
    
    Per eseguire questa applicazione, assicurati di aver installato tutte le dipendenze necessarie:
    ```
    pip install streamlit yfinance pandas numpy plotly pandas-datareader
    ```
    """)
