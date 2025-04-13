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
                print(f"{row['date'].strftime('%Y-%m-%d')}: {row['commits']} commit")

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
    df['day_label'] = df['date'].dt.strftime('%a\n%d/%m')

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
        print(f"{row['date'].strftime('%Y-%m-%d')}: {row['commits']} commit")

    # Crea e salva il grafico
    fig = create_commit_plot(commit_data)
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
            data_reali_parsed.append((datetime.strptime(data_str, '%Y-%m-%d'), count))

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