import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

df = pd.read_csv('data.csv')

df = df.dropna()

tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['text']).toarray()

knn = NearestNeighbors(n_neighbors=10, metric='cosine')
knn.fit(X)

def recommend_songs():
    text = lyrics_text.get('1.0', tk.END).strip()
    genre = genre_combo.get()
    mood = mood_combo.get()

    if not text:
        messagebox.showwarning('Advertencia', 'Por favor ingresa la letra de la canción.')
        return

    vec = tfidf.transform([text]).toarray()

    distances, indices = knn.kneighbors(vec, n_neighbors=50)
    
    recommendations = df.iloc[indices[0]][['artist_name', 'title', 'genre', 'mood_pred']]

    if genre != 'Todos':
        recommendations = recommendations[recommendations['genre'] == genre]
    if mood != 'Todos':
        recommendations = recommendations[recommendations['mood_pred'] == mood]

    recommendations.columns = ['Artista', 'Canción', 'Género', 'Mood']

    results_text.config(state=tk.NORMAL)
    results_text.delete('1.0', tk.END)
    results_text.insert(tk.END, recommendations.to_string(index=False))
    results_text.config(state=tk.DISABLED)

app = tk.Tk()
app.title('Sistema de Recomendación de Canciones')

screen_width = app.winfo_screenwidth()
screen_height = app.winfo_screenheight()

app_width = 900  # Ancho de la ventana
app_height = 600  # Altura de la ventana
x_position = (screen_width - app_width) // 2
y_position = (screen_height - app_height) // 2

app.geometry(f'{app_width}x{app_height}+{x_position}+{y_position}')

label_lyrics = ttk.Label(app, text='Ingresa la letra de la canción:')
label_lyrics.grid(row=0, column=0, padx=10, pady=10, sticky='w')

lyrics_text = scrolledtext.ScrolledText(app, width=120, height=10, wrap=tk.WORD)
lyrics_text.grid(row=1, column=0, columnspan=2, padx=10, pady=10)

label_genre = ttk.Label(app, text='Selecciona el género:')
label_genre.grid(row=2, column=0, padx=10, pady=10, sticky='w')

genres = ['Todos'] + df['genre'].unique().tolist()
genre_combo = ttk.Combobox(app, values=genres, width=20)
genre_combo.grid(row=2, column=1, padx=10, pady=10, sticky='w')
genre_combo.current(0)

label_mood = ttk.Label(app, text='Selecciona el estado de ánimo:')
label_mood.grid(row=3, column=0, padx=10, pady=10, sticky='w')

moods = ['Todos'] + df['mood_pred'].unique().tolist()
mood_combo = ttk.Combobox(app, values=moods, width=20)
mood_combo.grid(row=3, column=1, padx=10, pady=10, sticky='w')
mood_combo.current(0)

recommend_button = ttk.Button(app, text='Recomendar Canciones', command=recommend_songs)
recommend_button.grid(row=4, column=0, columnspan=2, padx=10, pady=10)

results_text = scrolledtext.ScrolledText(app, width=120, height=15, wrap=tk.WORD, state=tk.DISABLED)  # Aumentar el ancho del cuadro de texto
results_text.grid(row=5, column=0, columnspan=2, padx=10, pady=10)

app.mainloop()
