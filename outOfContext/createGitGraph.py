import os
import subprocess
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go


def get_git_commits(repo_path, author=None, since=None, until=None):
    """
    Estrae i dati dei commit da una repository Git locale.

    Parametri:
    repo_path (str): Percorso alla repository Git locale
    author (str, optional): Filtra i commit per autore
    since (str, optional): Data di inizio nel formato 'YYYY-MM-DD'
    until (str, optional): Data di fine nel formato 'YYYY-MM-DD'

    Returns:
    pandas.DataFrame: DataFrame con date e numero di commit
    """
    # Salva la directory corrente
    current_dir = os.getcwd()

    try:
        # Cambia directory alla repository
        os.chdir(repo_path)

        # Se non è specificata una data di inizio, la impostiamo a una settimana fa
        if not since:
            since_date = datetime.now() - timedelta(days=7)
            since = since_date.strftime('%Y-%m-%d')

        # Costruiamo il comando git log
        cmd = ['git', 'log', '--pretty=format:%ad', '--date=short']

        if author:
            cmd.extend(['--author', author])
        if since:
            cmd.extend(['--since', since])
        if until:
            cmd.extend(['--until', until])

        # Eseguiamo il comando
        result = subprocess.run(cmd, capture_output=True, text=True)

        # Gestione degli errori
        if result.returncode != 0:
            print(f"Errore nell'esecuzione del comando git: {result.stderr}")
            return pd.DataFrame(columns=['date', 'commits'])

        commit_dates = result.stdout.strip().split('\n')

        # Contiamo i commit per data
        if commit_dates and commit_dates[0]:
            date_counts = pd.Series(commit_dates).value_counts().sort_index()
            date_df = pd.DataFrame({'date': date_counts.index, 'commits': date_counts.values})
            date_df['date'] = pd.to_datetime(date_df['date'])

            # Stampa i dati recuperati per debug
            print("Dati recuperati dalla repository:")
            for _, row in date_df.iterrows():
                print(f"{row['date'].strftime('%Y-%m-%d-%H')}: {row['commits']} commit")

            return date_df
        else:
            print("Nessun commit trovato con i criteri specificati.")
            return pd.DataFrame(columns=['date', 'commits'])

    except Exception as e:
        print(f"Errore durante l'estrazione dei commit: {str(e)}")
        return pd.DataFrame(columns=['date', 'commits'])

    finally:
        # Ripristina la directory originale
        os.chdir(current_dir)


def create_commit_plot_hours(commit_data, hours=168):
    """
    Crea un grafico dei commit per ora usando Plotly
    """
    # Prepara i dati per tutte le ore del periodo
    end_date = datetime.now().replace(minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(hours=hours - 1)

    # Crea un dataframe con tutte le ore
    all_hours = pd.date_range(start=start_date, end=end_date, freq='H')
    df = pd.DataFrame({'datetime': all_hours})

    # Aggiungi le etichette per l'asse x
    df['hour_label'] = df['datetime'].dt.strftime('%d/%m\n%H:00')

    # Aggiungi i conteggi dei commit, usando 0 per le ore senza commit
    if not commit_data.empty:
        # Assumiamo che commit_data abbia una colonna 'datetime' con timestamp
        # Se invece ha solo date, dobbiamo prima convertirla in timestamp con ora
        hourly_commits = commit_data.copy()
        if 'datetime' not in hourly_commits.columns:
            # Se abbiamo solo date senza ore, convertiamo al formato orario
            hourly_commits['datetime'] = pd.to_datetime(hourly_commits['date'])

        # Raggruppiamo per ora
        hourly_commits = hourly_commits.groupby(pd.Grouper(key='datetime', freq='H')).sum().reset_index()

        # Unisci con tutti i timestamp del periodo
        df = df.merge(hourly_commits[['datetime', 'commits']], on='datetime', how='left')
    else:
        df['commits'] = 0

    df['commits'] = df['commits'].fillna(0).astype(int)

    # Definiamo i colori in stile GitHub
    colors = ['#ebedf0', '#9be9a8', '#40c463', '#30a14e', '#216e39']

    # Trova il valore massimo per la scala dei colori
    max_val = df['commits'].max() if df['commits'].max() > 0 else 1

    # Assegna colori in base al valore
    def get_color(val):
        if val == 0:
            return colors[0]
        elif val <= max_val * 0.25:
            return colors[1]
        elif val <= max_val * 0.5:
            return colors[2]
        elif val <= max_val * 0.75:
            return colors[3]
        else:
            return colors[4]

    # Crea un array di colori per ogni barra
    bar_colors = [get_color(val) for val in df['commits']]

    # Crea il grafico a barre
    fig = go.Figure()

    # Aggiungi barre per i commit
    fig.add_trace(go.Bar(
        x=df['hour_label'],
        y=df['commits'],
        text=df['commits'],
        textposition='outside',
        marker_color=bar_colors,
        hovertemplate='Data/Ora: %{x}<br>Commit: %{y}<extra></extra>'
    ))

    # Configura il layout
    fig.update_layout(
        title='Commit orari',
        yaxis_title='Numero di commit',
        plot_bgcolor='white',
        xaxis=dict(
            title='',
            showgrid=False,
            tickangle=0,
            # Mostriamo solo alcuni tick per non sovraffollare l'asse X
            tickmode='array',
            tickvals=df['hour_label'][::6].tolist()  # Mostra un tick ogni 6 ore
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='lightgray',
            gridwidth=0.5
        ),
        margin=dict(l=40, r=40, t=50, b=40),
        height=500,  # Aumentiamo l'altezza per un grafico più leggibile
        width=max(800, hours * 5)  # Larghezza dinamica in base al numero di ore
    )

    return fig

def create_commit_plot(commit_data, days=7):
    """
    Crea un grafico dei commit usando Plotly
    """
    # Prepara i dati per tutti i giorni del periodo
    end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=days - 1)

    # Crea un dataframe con tutte le date
    all_dates = pd.date_range(start=start_date, end=end_date)
    df = pd.DataFrame({'date': all_dates})

    # Aggiungi le etichette per l'asse x
    df['day_label'] = df['date'].dt.strftime('%a\n%d/%m-%H')

    # Aggiungi i conteggi dei commit, usando 0 per i giorni senza commit
    if not commit_data.empty:
        df = df.merge(commit_data[['date', 'commits']], on='date', how='left')
    else:
        df['commits'] = 0

    df['commits'] = df['commits'].fillna(0).astype(int)

    # Definiamo i colori in stile GitHub
    colors = ['#ebedf0', '#9be9a8', '#40c463', '#30a14e', '#216e39']

    # Trova il valore massimo per la scala dei colori
    max_val = df['commits'].max() if df['commits'].max() > 0 else 1

    # Assegna colori in base al valore
    def get_color(val):
        if val == 0:
            return colors[0]
        elif val <= max_val * 0.25:
            return colors[1]
        elif val <= max_val * 0.5:
            return colors[2]
        elif val <= max_val * 0.75:
            return colors[3]
        else:
            return colors[4]

    # Crea un array di colori per ogni barra
    bar_colors = [get_color(val) for val in df['commits']]

    # Crea il grafico a barre
    fig = go.Figure()

    # Aggiungi barre per i commit
    fig.add_trace(go.Bar(
        x=df['day_label'],
        y=df['commits'],
        text=df['commits'],
        textposition='outside',
        marker_color=bar_colors,
        hovertemplate='Data: %{x}<br>Commit: %{y}<extra></extra>'
    ))

    # Configura il layout
    fig.update_layout(
        title='Commit giornalieri',
        yaxis_title='Numero di commit',
        plot_bgcolor='white',
        xaxis=dict(
            title='',
            showgrid=False,
            tickangle=0
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='lightgray',
            gridwidth=0.5
        ),
        margin=dict(l=40, r=40, t=50, b=40)
    )

    return fig

def create_commit_mosaic(commit_data, weeks=1):
    """
    Crea un grafico a mosaico dei commit in stile GitHub
    """
    # Definiamo il periodo da visualizzare
    end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=(weeks * 7) - 1)

    # Crea un dataframe con tutte le date nel periodo
    all_dates = pd.date_range(start=start_date, end=end_date)
    df = pd.DataFrame({'date': all_dates})

    # Aggiungiamo giorno della settimana (0=lunedì, 6=domenica) e numero della settimana
    df['weekday'] = df['date'].dt.weekday

    # Calcoliamo il numero della settimana relativo all'inizio del periodo
    df['week'] = ((df['date'] - start_date).dt.days // 7)

    # Uniamo con i dati dei commit, usando 0 per i giorni senza commit
    if not commit_data.empty:
        df = df.merge(commit_data[['date', 'commits']], on='date', how='left')
    else:
        df['commits'] = 0

    df['commits'] = df['commits'].fillna(0).astype(int)

    # Prepariamo i dati per il grafico a mosaico
    pivot_data = df.pivot(index='weekday', columns='week', values='commits').fillna(0)

    # Definiamo i colori in stile GitHub
    colors = ['#ebedf0', '#9be9a8', '#40c463', '#30a14e', '#216e39']

    # Normalizzazione per la colorazione
    max_val = df['commits'].max() if df['commits'].max() > 0 else 1

    # Creiamo il grafico a mosaico usando heatmap
    fig = go.Figure(data=go.Heatmap(
        z=pivot_data.values,
        colorscale=[
            [0, colors[0]],
            [0.001 if max_val > 1 else 0.25, colors[0]],  # 0 commits
            [0.001 if max_val > 1 else 0.25, colors[1]],  # 1-25% del max
            [0.25, colors[1]],
            [0.25, colors[2]],  # 25-50% del max
            [0.5, colors[2]],
            [0.5, colors[3]],  # 50-75% del max
            [0.75, colors[3]],
            [0.75, colors[4]],  # 75-100% del max
            [1.0, colors[4]]
        ],
        showscale=False,
        hovertemplate='Commit: %{z}<extra></extra>'
    ))

    # Aggiungiamo testo per mostrare il numero di commit
    annotations = []
    for i in range(pivot_data.shape[0]):
        for j in range(pivot_data.shape[1]):
            if pivot_data.iloc[i, j] > 0:
                annotations.append(dict(
                    x=j, y=i,
                    text=str(int(pivot_data.iloc[i, j])),
                    showarrow=False,
                    font=dict(color='black' if pivot_data.iloc[i, j] < max_val * 0.75 else 'white', size=10)
                ))

    # Configuriamo il layout
    weekday_labels = ['Lun', 'Mar', 'Mer', 'Gio', 'Ven', 'Sab', 'Dom']
    week_labels = []
    for w in range(weeks):
        first_day = start_date + timedelta(days=w * 7)
        week_labels.append(first_day.strftime('%d %b'))

    fig.update_layout(
        title='Mosaico commit in stile GitHub',
        xaxis=dict(
            title='Settimana',
            tickmode='array',
            tickvals=list(range(weeks)),
            ticktext=week_labels,
            side='top'
        ),
        yaxis=dict(
            title='',
            tickmode='array',
            tickvals=list(range(7)),
            ticktext=weekday_labels
        ),
        annotations=annotations,
        margin=dict(l=40, r=20, t=60, b=20),
        height=300
    )

    # Quadratini invece di rettangoli
    fig.update_traces(
        xgap=3,  # spazio tra i quadrati in orizzontale
        ygap=3  # spazio tra i quadrati in verticale
    )

    return fig


def create_commit_mosaic_byHours(commit_data, hours=336):  # Default a 168 ore (7 giorni)
    """
    Crea un grafico a mosaico dei commit in stile GitHub con raggruppamento ogni 24 ore
    """
    # Definiamo il periodo da visualizzare
    end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(hours=hours)

    # Crea un dataframe con tutte le ore nel periodo
    all_hours = pd.date_range(start=start_date, end=end_date, freq='H')
    df = pd.DataFrame({'datetime': all_hours})

    # Estraiamo la data e l'ora
    df['date'] = df['datetime'].dt.date
    df['hour'] = df['datetime'].dt.hour

    # Calcoliamo il numero del giorno relativo all'inizio del periodo
    df['day'] = ((df['datetime'] - start_date).dt.total_seconds() // (24 * 3600))

    # Uniamo con i dati dei commit, usando 0 per le ore senza commit
    if not commit_data.empty:
        # Assumiamo che commit_data abbia una colonna 'datetime' o convertiamo la colonna 'date' in datetime
        hourly_commits = commit_data.groupby(pd.Grouper(key='date', freq='H')).sum().reset_index()
        hourly_commits['datetime'] = hourly_commits['date']
        df = df.merge(hourly_commits[['datetime', 'commits']], on='datetime', how='left')
    else:
        df['commits'] = 0

    df['commits'] = df['commits'].fillna(0).astype(int)

    # Prepariamo i dati per il grafico a mosaico
    pivot_data = df.pivot(index='hour', columns='day', values='commits').fillna(0)

    # Definiamo i colori in stile GitHub
    colors = ['#ebedf0', '#9be9a8', '#40c463', '#30a14e', '#216e39']

    # Normalizzazione per la colorazione
    max_val = df['commits'].max() if df['commits'].max() > 0 else 1

    # Creiamo il grafico a mosaico usando heatmap
    fig = go.Figure(data=go.Heatmap(
        z=pivot_data.values,
        colorscale=[
            [0, colors[0]],
            [0.001 if max_val > 1 else 0.25, colors[0]],  # 0 commits
            [0.001 if max_val > 1 else 0.25, colors[1]],  # 1-25% del max
            [0.25, colors[1]],
            [0.25, colors[2]],  # 25-50% del max
            [0.5, colors[2]],
            [0.5, colors[3]],  # 50-75% del max
            [0.75, colors[3]],
            [0.75, colors[4]],  # 75-100% del max
            [1.0, colors[4]]
        ],
        showscale=False,
        hovertemplate='Commit: %{z}<extra></extra>'
    ))

    # Aggiungiamo testo per mostrare il numero di commit
    annotations = []
    for i in range(pivot_data.shape[0]):
        for j in range(pivot_data.shape[1]):
            if pivot_data.iloc[i, j] > 0:
                annotations.append(dict(
                    x=j, y=i,
                    text=str(int(pivot_data.iloc[i, j])),
                    showarrow=False,
                    font=dict(color='black' if pivot_data.iloc[i, j] < max_val * 0.75 else 'white', size=10)
                ))

    # Configuriamo il layout
    hour_labels = [f'{h}:00' for h in range(24)]
    day_labels = []
    for d in range(int(hours // 24) + 1):
        current_day = start_date + timedelta(days=d)
        day_labels.append(current_day.strftime('%d %b'))

    fig.update_layout(
        title='Mosaico commit per ora del giorno',
        xaxis=dict(
            title='Giorno',
            tickmode='array',
            tickvals=list(range(len(day_labels))),
            ticktext=day_labels,
            side='top'
        ),
        yaxis=dict(
            title='Ora',
            tickmode='array',
            tickvals=list(range(24)),
            ticktext=hour_labels
        ),
        annotations=annotations,
        margin=dict(l=40, r=20, t=60, b=20),
        height=600  # Altezza maggiore per visualizzare meglio tutte le 24 ore
    )

    # Quadratini invece di rettangoli
    fig.update_traces(
        xgap=3,  # spazio tra i quadrati in orizzontale
        ygap=3  # spazio tra i quadrati in verticale
    )

    return fig

def main():
    repo_path = "/Users/riccardo/Sources/GitHub/Untitled/reLuxStorm"  # <-- Modifica questo!

    # Puoi specificare questi parametri se necessario
    author = None  # Filtra per autore specifico
    since = None  # Data di inizio (default: 7 giorni fa)
    until = None  # Data di fine (default: oggi)

    # Ottieni i dati dei commit
    commit_data = get_git_commits(repo_path, author, since, until)

    if commit_data.empty:
        print("Nessun dato da visualizzare.")
        return

    # Mostra i giorni con più commit
    top_days = commit_data.sort_values('commits', ascending=False).head(5)
    print("\nGiorni con più commit:")
    for _, row in top_days.iterrows():
        print(f"{row['date'].strftime('%Y-%m-%d-%H')}: {row['commits']} commit")

    # Crea e salva il grafico
    #fig = create_commit_mosaic(commit_data, weeks=2)
    fig = create_commit_mosaic_byHours(commit_data)

    fig.write_html("git_commits.html")  # Salva come file HTML interattivo
    #fig.write_image("git_commits.png")  # Salva come immagine PNG

    print("\nGrafico salvato come git_commits.html e git_commits.png")

    # Apre il browser per mostrare il grafico (opzionale)
    import webbrowser
    webbrowser.open("git_commits.html")


# Esegui lo script
if __name__ == "__main__":
    import sys


    # Funzione per mostrare i dati di esempio
    def show_example():
        print("Utilizzo dati di esempio...")

        # Crea dati di esempio basati sui tuoi dati reali
        end_date = datetime.now()
        dates = []
        commits = []

        # I tuoi dati reali menzionati
        data_reali = [
            ("2025-04-06", 187),
            ("2025-04-13", 140),
            ("2025-04-08", 133),
            ("2025-04-02", 128),
            ("2025-04-03", 124)
        ]

        # Convertiamo le date in oggetti datetime
        data_reali_parsed = []
        for data_str, count in data_reali:
            data_reali_parsed.append((datetime.strptime(data_str, '%Y-%m-%d-%h'), count))

        # Creiamo un range di 7 giorni
        for i in range(7):
            current_date = end_date - timedelta(days=6 - i)  # partendo da 6 giorni fa

            # Vediamo se questa data è nei dati reali
            commit_count = 0
            for data, count in data_reali_parsed:
                if current_date.date() == data.date():
                    commit_count = count
                    break

            dates.append(current_date)
            commits.append(commit_count)

        # Creiamo il DataFrame
        df = pd.DataFrame({'date': dates, 'commits': commits})

        print("Dati di esempio:")
        for _, row in df.iterrows():
            print(f"{row['date'].strftime('%Y-%m-%d')}: {row['commits']} commit")

        # Crea e salva il grafico
        fig = create_commit_plot(df)
        fig.write_html("git_commits_example.html")
        fig.write_image("git_commits_example.png")

        print("\nGrafico di esempio salvato come git_commits_example.html e git_commits_example.png")

        # Apre il browser
        import webbrowser
        webbrowser.open("git_commits_example.html")


    # Controllo argomenti
    if len(sys.argv) > 1 and sys.argv[1] == '--example':
        show_example()
    else:
        main()