from pathlib import Path 

PHONEME_TO_LETTERS = {
    # 1:1 mapping
    'a': 'a', 'e': 'e', 'o': 'o', 'u': 'u', 'i': 'i', 'i2': 'y',
    'p': 'p', 'b': 'b', 't': 't', 'd': 'd', 'k': 'k', 'g': 'g',
    'f': 'f', 'v': 'w', 's': 's', 'z': 'z', 'm': 'm', 'n': 'n',
    'l': 'l', 'r': 'r', 'j': 'j', 'w': 'ł', 'h': 'h',
    
    # specjalne polskie
    'S': 'sz', 'Z': 'ż', 'tS': 'cz', 'dZ': 'dż',
    'c': 'c', 'dz': 'dz',
    
    # miękkie — wymagają kontekstu (zobacz funkcję poniżej)
    'sj': 'ś', 'zj': 'ź', 'tsj': 'ć', 'dzj': 'dź', 'n~': 'ń',
    
    # nosówki
    'eo5': 'ę', 'oc5': 'ą',
    
    # cisza
    'sil': '', 'sp': '',
}
WLIST = ['alibaba', 'mysz', 'kotek', 'samochod', 'komputer', 'telefon', 'dom', 'drzewo', 'kwiat', 'lampa', 'banany', 'siema', 
         'cesarz', 'informatyka', 'programowanie', 'kot', 'kawa', 
         'herbata', 'rower', 'samolot', 'statek', 'góra', 'rzeka', 'las', 
         'miasto', 'piesek', 'jajeczko', 'matematyka', 'fizyka', 'chemia', 'biologia', 'historia',
           'geografia', 'filozofia', 'psychologia', 'sztuka', 'muzyka', 'sport', 'kino', 'barcelona', 'politechnika', 'marynarz'
           ,'kropelka', 'polska', 'niemcy', 'włochy', 'francja', 'hiszpania', 'anglia', 'rosja', 'usa', 'japonia', 'chiny', 'indie'
           'ukraina', 'siedemnaście', 'osiemnaście', 'dziewiętnaście', 'dwadzieścia', 'trzydzieści', 'czterdzieści', 'pięćdziesiąt', 'sześćdziesiąt',
           

          ]
WLIST1000 = [
    # Zwierzęta
    'pies', 'ptak', 'rybka', 'chomik', 'krowa', 'koń', 'świnia', 'owca', 'koza', 'kura',
    'kaczka', 'gęś', 'indyk', 'wąż', 'jaszczurka', 'żaba', 'pająk', 'mucha', 'komar', 'osa',
    'pszczoła', 'mrówka', 'motyl', 'tygrys', 'lew', 'słoń', 'żyrafa', 'małpa', 'niedźwiedź', 'wilk',
    'lis', 'zając', 'królik', 'jeleń', 'sarna', 'dzik', 'wiewiórka', 'jeż', 'nietoperz', 'rekin',
    'wieloryb', 'delfin', 'foka', 'pingwin', 'orzeł', 'sokół', 'gołąb', 'wróbel', 'sikorka', 'kruk',
    'sowa', 'bocian', 'łabędź', 'karp', 'szczupak', 'śledź', 'łoś', 'żubr', 'ryś', 'borsuk',
    'kret', 'bóbr', 'wydra', 'kuna', 'łasica', 'szop', 'hipopotam', 'nosorożec', 'zebra', 'krokodyl',
    'aligator', 'żółw', 'skorpion', 'stonoga', 'dżdżownica', 'ślimak', 'rak', 'krab', 'ośmiornica', 'meduza',
    'struś', 'paw', 'papuga', 'kanarek', 'kogut', 'bażant', 'kuropatwa', 'przepiórka', 'dzięcioł', 'kukułka',
    'słowik', 'skowronek', 'mewa', 'pelikan', 'flaming', 'anakonda', 'boa', 'kobra', 'pyton', 'żmija',

    # Jedzenie i napoje
    'chleb', 'masło', 'ser', 'mleko', 'woda', 'sok', 'jabłko', 'gruszka', 'śliwka', 'truskawka',
    'malina', 'jagoda', 'ziemniak', 'pomidor', 'ogórek', 'cebula', 'czosnek', 'marchew', 'pietruszka', 'seler',
    'por', 'kapusta', 'sałata', 'rzodkiewka', 'mięso', 'kurczak', 'wieprzowina', 'wołowina', 'ryba', 'sól',
    'pieprz', 'cukier', 'mąka', 'ryż', 'makaron', 'kasza', 'jajko', 'śniadanie', 'obiad', 'kolacja',
    'deser', 'zupa', 'ciasto', 'lody', 'czekolada', 'cukierek', 'lizak', 'ciastko', 'tort', 'wino',
    'piwo', 'wódka', 'szampan', 'koniak', 'likier', 'rumianek', 'kefir', 'kakao', 'kompot', 'lemoniada',
    'orzech', 'migdał', 'rodzynki', 'daktyle', 'figi', 'banan', 'pomarańcza', 'mandarynka', 'cytryna', 'grapefruit',
    'kiwi', 'ananas', 'mango', 'arbuz', 'melon', 'brzoskwinia', 'morela', 'wiśnia', 'czereśnia', 'agrest',
    'porzeczka', 'borówka', 'żurawina', 'papryka', 'dynia', 'cukinia', 'bakłażan', 'brokuł', 'kalafior', 'szpinak',
    'fasola', 'groch', 'soczewica', 'bób', 'koper', 'bazylia', 'oregano', 'tymianek', 'rozmaryn', 'cynamon',

    # Dom, budynki i przedmioty codziennego użytku
    'okno', 'drzwi', 'ściana', 'podłoga', 'sufit', 'dach', 'pokój', 'kuchnia', 'łazienka', 'sypialnia',
    'salon', 'korytarz', 'piwnica', 'strych', 'garaż', 'schody', 'balkon', 'taras', 'ogród', 'płot',
    'brama', 'klucz', 'zamek', 'klamka', 'dzwonek', 'stół', 'krzesło', 'fotel', 'kanapa', 'łóżko',
    'szafa', 'komoda', 'półka', 'biurko', 'dywan', 'obraz', 'lustro', 'zegar', 'telewizor', 'radio',
    'lodówka', 'pralka', 'zmywarka', 'kuchenka', 'mikrofalówka', 'piekarnik', 'odkurzacz', 'żelazko', 'deska', 'garnek',
    'patelnia', 'talerz', 'kubek', 'szklanka', 'sztućce', 'nóż', 'widelec', 'łyżka', 'miska', 'dzbanek',
    'długopis', 'ołówek', 'gumka', 'linijka', 'zeszyt', 'książka', 'kartka', 'papier', 'koperta', 'znaczek',
    'gazeta', 'czasopismo', 'notes', 'kalendarz', 'teczka', 'nożyczki', 'klej', 'taśma', 'spinacz', 'pinezka',
    'worek', 'torba', 'pudełko', 'karton', 'butelka', 'puszka', 'słoik', 'tuba', 'beczka', 'wiadro',
    'szczotka', 'miotła', 'mop', 'gąbka', 'ścierka', 'ręcznik', 'mydło', 'szampon', 'pasta', 'krem',

    # Ciało i ubrania
    'głowa', 'włosy', 'twarz', 'oko', 'ucho', 'nos', 'usta', 'ząb', 'język', 'warga',
    'szyja', 'ramię', 'ręka', 'palec', 'kciuk', 'paznokieć', 'klatka', 'pierś', 'brzuch', 'plecy',
    'kręgosłup', 'noga', 'kolano', 'stopa', 'pięta', 'skóra', 'kość', 'krew', 'mięsień', 'serce',
    'płuco', 'żołądek', 'wątroba', 'nerka', 'mózg', 'jelito', 'żyła', 'tętnica', 'nerw', 'staw',
    'czaszka', 'szczęka', 'broda', 'policzek', 'czoło', 'brew', 'rzęsa', 'powieka', 'łokieć', 'nadgarstek',
    'biodro', 'udo', 'łydka', 'kostka', 'gardło', 'ubranie', 'koszula', 'bluzka', 'sweter', 'bluza',
    'spodnie', 'dżinsy', 'spódnica', 'sukienka', 'kurtka', 'płaszcz', 'czapka', 'szalik', 'rękawiczka', 'but',
    'skarpetka', 'rajstopy', 'bielizna', 'majtki', 'biustonosz', 'krawat', 'pasek', 'torebka', 'plecak', 'portfel',
    'okulary', 'zegarek', 'pierścionek', 'naszyjnik', 'kolczyk', 'bransoletka', 'garnitur', 'kalesony', 'szlafrok', 'piżama',
    'kapelusz', 'kask', 'sandały', 'kozak', 'kalosz', 'kamizelka', 'kaptur', 'guzik', 'suwak', 'kieszeń',

    # Natura, czas i zjawiska geograficzne
    'słońce', 'księżyc', 'gwiazda', 'niebo', 'chmura', 'deszcz', 'śnieg', 'wiatr', 'burza', 'mgła',
    'lód', 'mróz', 'ciepło', 'zimno', 'ogień', 'ziemia', 'powietrze', 'piasek', 'kamień', 'skała',
    'trawa', 'liść', 'gałąź', 'korzeń', 'pień', 'krzew', 'mech', 'grzyb', 'morze', 'ocean',
    'jezioro', 'staw', 'strumień', 'dolina', 'pagórek', 'szczyt', 'wyspa', 'plaża', 'wybrzeże', 'pustynia',
    'dżungla', 'bór', 'piorun', 'błyskawica', 'grzmot', 'tęcza', 'huragan', 'tornado', 'powódź', 'trzęsienie',
    'sekunda', 'minuta', 'godzina', 'dzień', 'noc', 'rano', 'wieczór', 'południe', 'północ', 'tydzień',
    'miesiąc', 'rok', 'wiek', 'poniedziałek', 'wtorek', 'środa', 'czwartek', 'piątek', 'sobota', 'niedziela',
    'styczeń', 'luty', 'marzec', 'kwiecień', 'maj', 'czerwiec', 'lipiec', 'sierpień', 'wrzesień', 'październik',
    'listopad', 'grudzień', 'wiosna', 'lato', 'jesień', 'zima', 'wczoraj', 'dzisiaj', 'jutro', 'przedwczoraj',
    'pojutrze', 'teraz', 'zaraz', 'potem', 'nigdy', 'zawsze', 'często', 'rzadko', 'czasem', 'wkrótce',

    # Miasto, transport, praca i rodzina
    'auto', 'pociąg', 'tramwaj', 'autobus', 'trolejbus', 'metro', 'prom', 'łódź', 'żaglówka', 'helikopter',
    'rakieta', 'motocykl', 'skuter', 'hulajnoga', 'rolki', 'wrotki', 'deskorolka', 'bilet', 'stacja', 'przystanek',
    'dworzec', 'lotnisko', 'port', 'ulica', 'droga', 'autostrada', 'chodnik', 'ścieżka', 'skrzyżowanie', 'rondo',
    'most', 'tunel', 'wiadukt', 'wypadek', 'korek', 'sklep', 'apteka', 'piekarnia', 'rzeźnik', 'warzywniak',
    'market', 'galeria', 'teatr', 'muzeum', 'biblioteka', 'szkoła', 'przedszkole', 'uniwersytet', 'szpital', 'przychodnia',
    'bank', 'poczta', 'policja', 'straż', 'kościół', 'cmentarz', 'park', 'plac', 'pomnik', 'fontanna',
    'restauracja', 'kawiarnia', 'pub', 'hotel', 'basen', 'stadion', 'boisko', 'siłownia', 'klub', 'praca',
    'szef', 'pracownik', 'biuro', 'fabryka', 'firma', 'pensja', 'lekarz', 'pielęgniarka', 'nauczyciel', 'uczeń',
    'student', 'inżynier', 'architekt', 'prawnik', 'sędzia', 'policjant', 'strażak', 'żołnierz', 'rolnik', 'górnik',
    'rodzina', 'matka', 'ojciec', 'mama', 'tata', 'syn', 'córka', 'brat', 'siostra', 'dziadek',

    # Emocje, pojęcia abstrakcyjne, narzędzia i sztuka
    'babcia', 'wnuk', 'wnuczka', 'wujek', 'ciocia', 'kuzyn', 'kuzynka', 'mąż', 'żona', 'teść',
    'miłość', 'nienawiść', 'radość', 'smutek', 'strach', 'złość', 'gniew', 'zaskoczenie', 'zdziwienie', 'nadzieja',
    'wiara', 'szczęście', 'pech', 'ból', 'zdrowie', 'choroba', 'życie', 'śmierć', 'pokój', 'wojna',
    'wolność', 'niewola', 'prawda', 'kłamstwo', 'dobro', 'zło', 'piękno', 'brzydota', 'mądrość', 'głupota',
    'siła', 'słabość', 'odwaga', 'tchórzostwo', 'duma', 'wstyd', 'wina', 'kara', 'nagroda', 'cel',
    'sens', 'marzenie', 'pomysł', 'myśl', 'pamięć', 'uwaga', 'rozum', 'dusza', 'wola', 'charakter',
    'osobowość', 'los', 'przeznaczenie', 'przypadek', 'sukces', 'porażka', 'problem', 'rozwiązanie', 'pytanie', 'odpowiedź',
    'przyczyna', 'skutek', 'początek', 'koniec', 'środek', 'część', 'całość', 'różnica', 'podobieństwo', 'waga',
    'młotek', 'śrubokręt', 'wiertarka', 'piła', 'gwóźdź', 'śruba', 'klucz', 'obcęgi', 'kombinerki', 'gitara',
    'pianino', 'skrzypce', 'flet', 'bęben', 'trąbka', 'saksofon', 'wiolonczela', 'perkusja', 'rytm', 'malarz',

    # Czasowniki (część 1)
    'być', 'mieć', 'móc', 'chcieć', 'musieć', 'wiedzieć', 'mówić', 'robić', 'widzieć', 'iść',
    'dać', 'wziąć', 'spać', 'jeść', 'pić', 'stać', 'siedzieć', 'leżeć', 'biec', 'jechać',
    'latać', 'pływać', 'skakać', 'padać', 'rzucać', 'łapać', 'trzymać', 'nosić', 'ciągnąć', 'pchać',
    'otwierać', 'zamykać', 'zaczynać', 'kończyć', 'szukać', 'znajdować', 'gubić', 'chować', 'pokazywać', 'patrzeć',
    'słuchać', 'słyszeć', 'czuć', 'pachnieć', 'smakować', 'dotykać', 'myśleć', 'pamiętać', 'zapominać', 'rozumieć',
    'uczyć', 'studiować', 'czytać', 'pisać', 'liczyć', 'rysować', 'malować', 'śpiewać', 'tańczyć', 'grać',
    'pracować', 'odpoczywać', 'bawić', 'śmiać', 'płakać', 'cieszyć', 'martwić', 'złościć', 'bać', 'kochać',
    'lubić', 'nienawidzić', 'szanować', 'pomagać', 'przeszkadzać', 'pytać', 'odpowiadać', 'prosić', 'dziękować', 'przepraszać',
    'witać', 'żegnać', 'zapraszać', 'spotykać', 'czekać', 'spieszyć', 'spóźniać', 'zdążyć', 'trwać', 'zmieniać',
    'rosnąć', 'maleć', 'budować', 'niszczyć', 'tworzyć', 'kupować', 'sprzedawać', 'płacić', 'kosztować', 'kraść',

    # Czasowniki (część 2) i Przymiotniki (część 1)
    'oszukiwać', 'walczyć', 'bronić', 'atakować', 'uciekać', 'gonić', 'wygrywać', 'przegrywać', 'rodzić', 'umierać',
    'żyć', 'mieszkać', 'pochodzić', 'nazywać', 'wyglądać', 'znaczyć', 'wydawać', 'zgadzać', 'proponować', 'decydować',
    'dobry', 'zły', 'wielki', 'mały', 'nowy', 'stary', 'młody', 'długi', 'krótki', 'wysoki',
    'niski', 'szeroki', 'wąski', 'gruby', 'chudy', 'ciężki', 'lekki', 'gorący', 'ciepły', 'zimny',
    'chłodny', 'mokry', 'suchy', 'twardy', 'miękki', 'ostry', 'tępy', 'gładki', 'szorstki', 'jasny',
    'ciemny', 'czysty', 'brudny', 'ładny', 'piękny', 'brzydki', 'mądry', 'głupi', 'bogaty', 'biedny',
    'zdrowy', 'chory', 'silny', 'słaby', 'szybki', 'wolny', 'głośny', 'cichy', 'tani', 'drogi',
    'łatwy', 'trudny', 'prosty', 'krzywy', 'pełny', 'pusty', 'wesoły', 'smutny', 'grzeczny', 'niegrzeczny',
    'miły', 'niemiły', 'ciekawy', 'nudny', 'ważny', 'nieważny', 'prawdziwy', 'fałszywy', 'zajęty', 'gotowy',
    'zmęczony', 'głodny', 'spragniony', 'pijany', 'trzeźwy', 'śpiący', 'odważny', 'tchórzliwy', 'dumny', 'skromny',

    # Przymiotniki (część 2), Spójniki, Przyimki, Zaimki i Kolory
    'uczciwy', 'kłamliwy', 'leniwy', 'pracowity', 'spokojny', 'nerwowy', 'ostrożny', 'niebezpieczny', 'bezpieczny', 'dziwny',
    'normalny', 'śmieszny', 'poważny', 'łagodny', 'gorzki', 'słodki', 'kwaśny', 'słony', 'pyszny', 'ohydny',
    'świeży', 'zepsuty', 'wczesny', 'późny', 'pierwszy', 'ostatni', 'kolejny', 'następny', 'poprzedni', 'lewy',
    'prawy', 'górny', 'dolny', 'środkowy', 'główny', 'poboczny', 'biały', 'czarny', 'czerwony', 'niebieski',
    'zielony', 'żółty', 'brązowy', 'pomarańczowy', 'fioletowy', 'różowy', 'szary', 'złoty', 'srebrny', 'bardzo',
    'mało', 'dużo', 'trochę', 'wcale', 'też', 'także', 'oraz', 'lub', 'albo', 'czy',
    'jeśli', 'jeżeli', 'ponieważ', 'dlatego', 'więc', 'zatem', 'jednak', 'ale', 'lecz', 'chociaż',
    'mimo', 'tylko', 'nawet', 'już', 'jeszcze', 'znowu', 'przecież', 'chyba', 'tutaj', 'tam',
    'stąd', 'gdzie', 'kiedy', 'dlaczego', 'kto', 'który', 'czyj', 'mój', 'twój', 'jego',
    'jej', 'nasz', 'wasz', 'ich', 'siebie', 'sobie', 'mną', 'tobą', 'nim', 'nią'

    'alibaba', 'mysz', 'kotek', 'samochod', 'komputer', 'telefon', 'dom', 'drzewo', 'kwiat', 'lampa', 'banany', 'siema', 
    'cesarz', 'informatyka', 'programowanie', 'kot', 'kawa', 
    'herbata', 'rower', 'samolot', 'statek', 'góra', 'rzeka', 'las', 
    'miasto', 'piesek', 'jajeczko', 'matematyka', 'fizyka', 'chemia', 'biologia', 'historia',
    'geografia', 'filozofia', 'psychologia', 'sztuka', 'muzyka', 'sport', 'kino', 'barcelona', 'politechnika', 'marynarz',
    'kropelka', 'polska', 'niemcy', 'włochy', 'francja', 'hiszpania', 'anglia', 'rosja', 'usa', 'japonia', 'chiny', 'indie'
    'ukraina', 'siedemnaście', 'osiemnaście', 'dziewiętnaście', 'dwadzieścia', 'trzydzieści', 'czterdzieści', 'pięćdziesiąt', 'sześćdziesiąt',
    ]
def phonemes_to_text(phonemes, after_silence = False):
    """Konwertuje listę fonemów na tekst z polskimi zasadami."""
    out = []
    for i, ph in enumerate(phonemes):
        next_ph = phonemes[i+1] if i+1 < len(phonemes) else None
          
        # miękkie spółgłoski przed samogłoską: "ś" → "si" + samogłoska
        # ale na końcu lub przed spółgłoską: zostają jako "ś"
        if ph in ('sj', 'zj', 'tsj', 'dzj', 'n~'):
            soft_map = {'sj': 'ś', 'zj': 'ź', 'tsj': 'ć', 'dzj': 'dź', 'n~': 'ń'}
            if next_ph == 'i':
                # "ś" + "i" → "si" (na piśmie)
                base = {'sj': 's', 'zj': 'z', 'tsj': 'c', 'dzj': 'dz', 'n~': 'n'}[ph]
                out.append(base)
            elif next_ph in ('a', 'e', 'o', 'u', 'eo5', 'oc5'):
                # "ś" + "a" → "sia" (na piśmie)
                base = {'sj': 's', 'zj': 'z', 'tsj': 'c', 'dzj': 'dz', 'n~': 'n'}[ph]
                out.append(base + 'i')
                # samogłoska zostanie dodana w następnej iteracji
            else:
                out.append(soft_map[ph])
        else:
            out.append(PHONEME_TO_LETTERS.get(ph, ph))
        if after_silence == False and next_ph == 'sil' and i > 0: #po sil juz nic nie liczy
            break
    return ''.join(out)

def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def parse_words(text_grid, silences_same=True, verbose=False):
    
    words = []
    in_words = False
    xmin = xmax = None
    for line in text_grid.split("\n"):
        if verbose:
            print(line)
        line = line.strip()
        if 'name = "words"' in line:
            in_words = True
            continue
        if in_words and line.startswith("name =") and "words" not in line:
            break
        if not in_words:
            continue
        if line.startswith("xmin =") and "intervals" not in line:
            xmin = float(line.split("=")[1].strip())
        elif line.startswith("xmax =") and "intervals" not in line:
            xmax = float(line.split("=")[1].strip())
        elif line.startswith("text ="):
            text = line.split("=", 1)[1].strip().strip('"')
            if text == "sp" and silences_same:
                text = "sil"
            if xmin is not None and xmax is not None:
                words.append((xmin, xmax, text))
            xmin = xmax = None
    return words


def proba_predict(result, p=1, longer_reg = False, noise_v=True, aeo_reg = True, verbose=False ):
    tab = result['probas']
    longer = ['a', 'e', 'sil']
    longvowels = ['a', 'e', 'o', 'oc5']
    k = len(tab[0])
    dur = len(tab)
    word = ['sil']
    wordprob = [1]
    phonemes = list(tab[0].items())
    candidates = phonemes
    for i in range(1, dur):

        phonemes = list(tab[i].items())
        #if aeou_reg  and len(candidates) > 1 and candidates[0][0] in longvowels and candidates[1][0] in longvowels:
            #candidates.pop(1)
        #print(phonemes)
            
        lasting = [0] * len(candidates) #lista do sprawdzania czy dany kandydat jest nadal aktualny
        plist = [pho for pho, va in phonemes]
        idown = 0
        for ic in range(len(candidates)):
            if candidates[ic][0] not in plist:
                lasting[ic] = 1
        for ic in range(len(candidates)):
            if lasting[ic] == 1:
                candidates.pop(ic-idown)
                idown += 1

        for i_new in range(k):
            ph = phonemes[i_new][0]
            val = phonemes[i_new][1]
            NEW = True

            for i_old in range(len(candidates)):
                if(ph == candidates[i_old][0]):
                    old_p = candidates[i_old][1]
                    candidates[i_old] = ((ph, val+old_p))
                    NEW = False
            if(NEW and val > 0.02):
                candidates.append((ph, val))
            candidates.sort(key=lambda item: item[1], reverse=True)
        


        if(candidates[0][1]>p):
            if (longer_reg and candidates[0][0] in longer):
                if(candidates[0][1] < 2*p):
                    continue
            if(candidates[0][0] != word[-1]):
                word.append(candidates[0][0])
                wordprob.append(candidates[0][1])
            candidates = candidates[1:]
            #candidates = []
        #print(candidates)


    if noise_v:
        for i in range(len(word)-2, 0, -1):
            if (word[i-1] == 'sil' and word[i+1] == 'sil'):
                word.pop(i)
                word.pop(i)
                wordprob.pop(i)
                wordprob.pop(i)
    
    if aeo_reg:
        for i in range(len(word)-1, 0, -1):
                if (word[i] in longvowels and word[i-1] in longvowels):
                    word.pop(i)
                    wordprob.pop(i)
    if(verbose):
        print("here is word: ")
        for i in range(len(word)):
            print(word[i], round(wordprob[i], 2))
    
    return word
    #print(word)
    #print(wordprob)

def list_add(word_list, word_grid):
    for word in word_grid:
        if word[2] not in word_list and word[2] != "sil":
            word_list.append(word[2])
    return word_list       

def dictionary_extend(word_list, data_dir):
    tg_paths = sorted(str(p) for p in Path(data_dir).rglob("*.TextGrid"))
    print(f"znaleziono {len(tg_paths)} plików")
    for tg_path in tg_paths:
        with open(tg_path, "r", encoding="utf-8") as f:
            file_words = parse_words(f.read())
        word_list = list_add(word_list, file_words)
    return word_list