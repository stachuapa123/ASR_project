def load_word_list(path):
    """Wczytuje listę słów z pliku tekstowego (jedno słowo per linia)."""
    with open(path, 'r', encoding='utf-8') as f:
        words = [line.strip() for line in f if line.strip()]
    return words


def save_word_list(words, path):
    """Zapisuje listę słów do pliku tekstowego."""
    with open(path, 'w', encoding='utf-8') as f:
        for word in words:
            f.write(word + '\n')